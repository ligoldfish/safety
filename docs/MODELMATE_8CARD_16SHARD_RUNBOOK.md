# ModelMate 8 卡 / 16 逻辑切片消融运行手册

## 1. 固定执行语义

本项目的正式调度方式是：**8 张 910B 作为 8 个并发槽位，509 个实验单元按轮次拆成最多 16 个逻辑切片；每个实验单元仍使用 1 张 NPU。**

不要把 `--num-devices` 改成 8。当前训练入口已经验证且强制单设备语义；把独立实验并行到 8 张卡，既能保持实验定义不变，也能避免错误的数据并行、重复样本和不可比的训练预算。

池调度器具有以下约束：

- 一个设备同一时刻只运行一个 shard；
- 最多 8 个 shard 并行，16 个 shard 动态领取，先结束的卡继续领取下一片；
- 每条下游正式轮次必须看到上一轮的真实 `READY` 汇总；
- `--dry-run` 和 `--preflight-only` 不能满足正式依赖；
- 任一 shard 失败后不再领取新 shard，已运行的结果和失败日志保留在 DPC 输出目录；
- NPU 轮次在启动前逐卡执行真实 forward/backward 探针；
- 最终门禁逐个核对 509 个 cell 均为 `COMPLETED`。

## 2. 固定目录

```bash
export SAFETY_SOURCE_ROOT=/home/work/user-job-dir/app/safety-src-886760f
export SAFETY_MODEL_ROOT=/opt/dpcvol/models/safetytransfer
export SAFETY_DATA_ROOT=/opt/dpcvol/datasets/safetytransfer
export SAFETY_OUTPUT_ROOT=/opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f
export SAFETY_ASSET_MANIFEST="$SAFETY_SOURCE_ROOT/configs/ablations/assets.modelmate.template.json"
export SAFETY_ACTIVATE="$SAFETY_MODEL_ROOT/_runtime_assets/activate-safety-ablation.sh"
```

模型、数据、运行环境和所有训练输出必须位于 `/opt/dpcvol`。Notebook 工作区只保存精简代码和两个启动文件。

## 3. 平台表单

提交前，精简源码快照中必须至少新增/更新这些文件：

```text
src/ablations/modelmate_pool.py
scripts/35_modelmate_8card_pool.py
scripts/36_modelmate_ablation_final_gate.py
configs/ablations/catalog.yaml
configs/ablations/assets.modelmate.template.json
```

Notebook 工作区根目录必须放置：

```text
run_safety_8card.py
boot_safety_8card.sh
```

两者应为普通文件，源码快照内也不应有符号链接。`SAFETY_SOURCE_ROOT` 必须指向实际包含上述 `src/`、`scripts/` 和 `configs/` 的精简快照；若平台提交前给快照改了目录名，只改这一项环境变量，不要改代码。

每一轮创建一个训练作业：

- Worker 节点：`1`
- 规格：`8*ASCEND_910B | 176U | 880G`
- 启动文件：`/home/work/user-job-dir/app/run_safety_8card.py`
- 训练数据集：平台若要求必选，可选择项目已有的小型占位数据集；正式数据由 `SAFETY_DATA_ROOT` 读取
- 环境变量：至少填写上一节的 6 个变量
- 运行参数：按下一节只填写当前轮次，例如 `--round p0-smoke`

启动脚本默认使用 `--devices 8 --logical-shards 16`。父进程不会覆盖平台下发的
`ASCEND_RT_VISIBLE_DEVICES`；调度器会为每个 worker 从该列表中只选择一个条目，使现有单卡入口的
逻辑 `npu:0` 分别映射到 8 张不同物理卡，而不是让 8 个进程挤在第一张卡上。

## 4. 按重要性提交的 14 个轮次

先只提交 `p0-smoke`。它成功后再逐轮提交，不能一次创建全部作业。

| 顺序 | 运行参数 | 单元数 | 设备 | 实际切片数 | 每片上限 |
|---:|---|---:|---|---:|---:|
| 1 | `--round p0-smoke` | 8 | NPU | 8 | 1 |
| 2 | `--round p0-core` | 54 | NPU | 16 | 4 |
| 3 | `--round p0-wjb` | 90 | NPU | 16 | 6 |
| 4 | `--round p0-fairness` | 24 | NPU | 16 | 2 |
| 5 | `--round p0-evaluate` | 2 | NPU | 2 | 1 |
| 6 | `--round p0-analyze` | 154 | CPU | 16 | 10 |
| 7 | `--round p0-manual` | 3 | CPU | 3 | 1 |
| 8 | `--round p1-mechanism` | 99 | NPU | 16 | 7 |
| 9 | `--round p1-data` | 16 | NPU | 16 | 1 |
| 10 | `--round p1-evaluate` | 5 | NPU | 5 | 1 |
| 11 | `--round p1-analyze` | 5 | CPU | 5 | 1 |
| 12 | `--round p2-generalization` | 6 | NPU | 6 | 1 |
| 13 | `--round p2-evaluate` | 24 | NPU | 16 | 2 |
| 14 | `--round p2-analyze` | 27 | CPU | 16 | 2 |

`p0-smoke` 是 `p0-core` 的前 8 个 cell，并共享真实 ledger。第二轮会复用已完成状态，不会重新训练这 8 个 cell。最终唯一实验总数仍是 509。

`p0-manual` 要求 `human_annotations` 中已有盲化双人标注和 WildGuard 预测。若人工标注尚未完成，该轮应按预期阻塞；不要伪造 READY，也不要用 dry-run 越过它。

## 5. 每轮提交前预检

可以在训练作业运行参数后加 `--preflight-only`。预检会检查该轮资产，并对 NPU 轮次逐卡运行真实张量探针，但不会执行 cell，也不会覆盖已有正式 `pool-summary.json`。

```text
--round p0-smoke --preflight-only
```

预检报告：

```text
$SAFETY_OUTPUT_ROOT/jobs/<round>/preflight-summary.json
$SAFETY_OUTPUT_ROOT/jobs/<round>/device-preflight.json
```

正式启动时删除 `--preflight-only`。

## 6. 每轮完成判定与续跑

只有下面文件同时满足条件才算轮次成功：

```text
$SAFETY_OUTPUT_ROOT/jobs/<round>/pool-summary.json
$SAFETY_OUTPUT_ROOT/jobs/<round>/status.json
```

要求：

- `pool-summary.json.status == "READY"`
- `pool-summary.json.dry_run == false`
- `failed_shards == []`
- `pending_shards == []`
- `status.json` 中本轮全部 cell 为 `COMPLETED`

作业异常终止后，用完全相同的 `--round` 重提即可。每个 cell 有持久 ledger，已完成 cell 会跳过；不要删除 `run-state`，也不要改 plan、资产清单或输出根目录。

单 shard 日志位于：

```text
$SAFETY_OUTPUT_ROOT/jobs/<round>/shards/shard-*/worker.log
```

## 7. 全部实验的最终门禁

14 轮结束后，在已经激活正式环境的 Notebook 终端执行：

```bash
source /opt/dpcvol/models/safetytransfer/_runtime_assets/activate-safety-ablation.sh
cd /home/work/user-job-dir/app/safety-src-886760f
python3 scripts/36_modelmate_ablation_final_gate.py \
  --output-root /opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f
```

只有输出同时出现以下内容，才能声明消融矩阵完成：

```text
"status": "READY"
"expected_cells": 509
"covered_cells": 509
```

审计文件固定保存为：

```text
/opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f/final-completion.json
```

若报告为 `BLOCKED`，根据 `blockers` 指向的轮次原样续跑；不要直接修改 JSON 状态文件。
