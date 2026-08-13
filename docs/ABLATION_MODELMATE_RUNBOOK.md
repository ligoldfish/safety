# ModelMate 全量消融实验运行手册

本手册对应 `configs/ablations/catalog.yaml` 和
`scripts/33_modelmate_ablation_job.py`。代码枚举得到的正式矩阵是 **497 个单元**：

| 波次 | 单元数 | 内容 |
|---|---:|---|
| `core-train` | 175 | P0-02 六语料三方法三 seed、主文组件与敏感性消融、P2 常规训练；不含 WJB 边界和公平性 |
| `wjb` | 90 | P0-06：5 模型对 × 2 配置 × 3 curation × 3 方法 |
| `fairness` | 12 | P0-07：6 数据集 × 2 配置 |
| `evaluate` | 31 | P0-08、P1-19、P2-02、P2-05 |
| `analyze` | 186 | provenance、统计、机制、效率、伦理分析 |
| `manual` | 3 | P0-03 三个分层的双人标注审计 |

即 **277 个训练、31 个评测、186 个分析、3 个人工单元**。主表本身是
5 模型对 × 6 语料 × 5 方法 = 150 个 provenance 单元；它不是 150 次额外训练。

## 1. 持久目录

代码快照可以位于 `/home/work/user-job-dir/app/`，模型、数据、状态和正式输出不能位于
作业快照、`/tmp` 或 `/cache`。建议：

```bash
export SAFETY_MODEL_ROOT=/opt/dpcvol/models/safetytransfer
export SAFETY_DATA_ROOT=/opt/dpcvol/datasets/safetytransfer
export SAFETY_OUTPUT_ROOT=/opt/dpcvol/datasets/safetytransfer/ablation-outputs/$(date +%Y%m%d)
export HF_HOME="$SAFETY_DATA_ROOT/_hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$SAFETY_OUTPUT_ROOT" "$SAFETY_OUTPUT_ROOT/registries"
```

如平台为正式输出提供了其它 DPC/OBS 授权目录，只替换
`SAFETY_OUTPUT_ROOT`；不要假定存在 `/opt/dpcvol/outputs`。不要在提交目录建立指向 DPC 的
符号链接，也不要覆盖平台注入的 `ASCEND_RT_VISIBLE_DEVICES`。

## 2. 运行前资产

复制模板并仅修改路径，不要修改键名或类型：

```bash
cd /home/work/user-job-dir/app/safety-src-b9b8184f-clean
mkdir -p "$SAFETY_DATA_ROOT/manifests"
cp configs/ablations/assets.modelmate.template.json \
  "$SAFETY_DATA_ROOT/manifests/assets.ablation.json"
export ASSET_MANIFEST="$SAFETY_DATA_ROOT/manifests/assets.ablation.json"
```

训练单元的模型和数据不写进这个 JSON：预检会从实际 pair/dataset YAML 精确解析。

- 模型：Qwen3.5-9B、Qwen3.5-0.8B、Llama-3.1-8B-Instruct、
  Llama-3.2-1B-Instruct、Qwen3-8B、Qwen3-4B、Qwen3-0.6B；
  P2-03 另需 `teacher-controls/same-size-base` 与 `teacher-controls/safety-tuned`。
- PAN：`$SAFETY_DATA_ROOT/external/safety-residual-space/data/` 下必须有
  `toxicity.csv`、`safety.csv`、`add_moderation.csv`、`sr_moderation.csv`。
- 安全语料：`processed/safety/*.jsonl`、`processed/eval/*.jsonl` 以及
  `processed/pan_test_set.jsonl`。正式预检拒绝空文件、ID 问题、train/eval prompt 重复或泄漏。
- `common_test/`：必须有 `pan_heldout.jsonl` 与 `common_safety.jsonl`。
- `wildguard` 是完整离线 Hugging Face 模型目录。

模板中的 21 个键是外部或上游产物，不会自动伪造：

- `search_ledger` 在 P0-07 前准备，必须是 validation-only 且各方法搜索预算相同；
- `checkpoint_registry`、`model_registry`、各 prediction registry、
  `phase_runtime_logs` 在相应训练/评测结果汇总后准备；
- `human_annotations` 必须包含 `judge_predictions.jsonl`、`blind_key.json`、
  `rater_a.jsonl`、`rater_b.jsonl`；
- `benchmark_assets` 是含 `run.py` 或 `tools/run.py` 的 OpenCompass checkout；
- `dataset_registry`/`split_manifests` 必须记录数据来源、许可和 split 审计；
- `trained_checkpoints` 是一个被 P1-19/P2-05 明确选定的参考模型，不会把不同模型混在同一 cell。

缺少哪一项，对应 cell 会在 `preflight` 阶段以结构化原因进入 `BLOCKED`，不会输出假结果。

## 3. 先做 dry-run

每个平台作业只跑一个 cell，最稳妥。下面以 175 个常规训练单元的第 0 个为例：

```bash
python scripts/33_modelmate_ablation_job.py \
  --wave core-train \
  --model-root "$SAFETY_MODEL_ROOT" \
  --data-root "$SAFETY_DATA_ROOT" \
  --output-root "$SAFETY_OUTPUT_ROOT" \
  --asset-manifest "$ASSET_MANIFEST" \
  --shard-index 0 --shard-count 175 --max-cells 1 \
  --device npu --device-id 0 --num-devices 1 \
  --dry-run
```

dry-run 状态写到 `dry-run-state`，正式状态写到 `run-state`，二者永不混用。入口内部按相同
`--shard-index`、`--shard-count`、`--max-cells` 先生成计划和做精确分片预检，再执行。

## 4. 正式作业

确认 dry-run 后去掉 `--dry-run`：

```bash
python scripts/33_modelmate_ablation_job.py \
  --wave core-train \
  --model-root "$SAFETY_MODEL_ROOT" \
  --data-root "$SAFETY_DATA_ROOT" \
  --output-root "$SAFETY_OUTPUT_ROOT" \
  --asset-manifest "$ASSET_MANIFEST" \
  --shard-index 0 --shard-count 175 --max-cells 1 \
  --device npu --device-id 0 --num-devices 1
```

其它波次只替换 wave 和总分片数：

```text
core-train  175
wjb          90   # P0-06；Llama 在此非常重要，因为失败边界要求 5/5 模型对
fairness     12   # P0-07；先准备 search_ledger
evaluate     31
analyze     186
manual        3
```

可在平台分别提交索引 `0..N-1`。每个 job 都必须保留平台自己的
`ASCEND_RT_VISIBLE_DEVICES`，进程内只使用逻辑 `npu:0`。不要在一个容器里后台启动作业后立刻退出。

## 5. 直接使用底层 CLI

入口等价于以下四步；诊断时可以手动执行：

```bash
python scripts/30_ablation.py plan --scope all --execution-kind train \
  --exclude-experiment-id P0-06 --exclude-experiment-id P0-07 \
  --output-root "$SAFETY_OUTPUT_ROOT/cell-outputs" --output /tmp/core-plan.jsonl

python scripts/30_ablation.py preflight --plan /tmp/core-plan.jsonl \
  --asset-manifest "$ASSET_MANIFEST" --shard-index 0 --shard-count 175 \
  --max-cells 1 --device npu --output "$SAFETY_OUTPUT_ROOT/preflight.json"

python scripts/30_ablation.py run --plan /tmp/core-plan.jsonl \
  --asset-manifest "$ASSET_MANIFEST" --shard-index 0 --shard-count 175 \
  --max-cells 1 --state-root "$SAFETY_OUTPUT_ROOT/run-state" \
  --device npu --device-id 0 --num-devices 1

python scripts/30_ablation.py status --plan /tmp/core-plan.jsonl \
  --state-root "$SAFETY_OUTPUT_ROOT/run-state" \
  --output "$SAFETY_OUTPUT_ROOT/status.json"
```

相同配置重复提交时，完成单元会根据账本与指纹直接恢复；配置、资产 manifest 或 cell 定义变化
会拒绝误复用。运行成功还必须通过 completion artifact 存在、非空、JSON/JSONL 可解析和哈希落盘检查。

## 6. 推荐顺序

1. 先修完数据 overlap 审计，并运行 `core-train`、`wjb`、`fairness`；
2. 从训练输出建立 immutable `checkpoint_registry`、`model_registry` 和 prediction registries；
3. 运行 `evaluate`；
4. 聚合逐样本结果、阶段日志、hidden states 与 split manifests；
5. 运行 `analyze`；
6. 导出 P0-03 盲化包，双人标注回传后再运行 `manual`。

P0-06 是最优先且 Llama 重要性高：设计要求验证 WJB 的负增益是否在 5/5 模型对上成立；缺少两个
Llama 权重会使该结论只能退化为 Qwen 家族内部结论。P2-04 的跨 tokenizer 实验也依赖
Llama-3.2-1B-Instruct。

## 7. 提交包与依赖

不要直接检查或打包开发仓库根目录：仓库中保留的 `data/` 与 `external/` 是历史开发输入，正式作业
已经通过 `SAFETY_DATA_ROOT` 和 typed asset manifest 读取 DPC 资产。先制造只含运行源码的 staging：

```bash
set -euo pipefail
SRC_ROOT="${SRC_ROOT:?export SRC_ROOT first}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$HOME/work/safety-ablation-bundles}"
SOURCE_ID="$(git -C "$SRC_ROOT" rev-parse HEAD)"
STAGE="$BUNDLE_ROOT/source-$SOURCE_ID"
ARCHIVE="$BUNDLE_ROOT/source-$SOURCE_ID.tar.gz"

test ! -e "$STAGE"
mkdir -p "$STAGE"
for item in README.md configs docs scripts src run_experiments.sh; do
  cp -a "$SRC_ROOT/$item" "$STAGE/"
done

STAGE="$STAGE" python - <<'PY'
import os
from src.ablations.preflight import inspect_submission_package
report = inspect_submission_package(os.environ["STAGE"])
print(report.to_dict())
raise SystemExit(0 if report.status == 'READY' else 3)
PY

tar -C "$BUNDLE_ROOT" -czf "$ARCHIVE" "source-$SOURCE_ID"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
du -h "$ARCHIVE"
```

`STAGE` 和压缩包都不得包含顶层 `data/`、`external/`、模型、checkpoint、输出、缓存或符号链接。
每次源码变化使用新的 commit/hash 名，不能原地覆盖旧 bundle。作业解压后把该目录作为项目根，并从
`scripts/33_modelmate_ablation_job.py` 启动。各 shard 的 plan/preflight/status 写入独立目录；同一 wave
的正式 `run-state` 共享，以便统一断点恢复且避免报告覆盖。

正式作业禁止对平台已有环境做无约束升级。`torch`、`torch_npu`、CANN、Transformers、PEFT、
Safetensors 等应来自平台已验证镜像或固定 DPC wheelhouse；启动后打印版本并执行最小 NPU 算子。
入口不会自行联网安装依赖，也不会写入访问令牌。
