# ModelMate 8 卡 / 16 逻辑切片 / 6 波次运行手册

## 1. 固定执行语义

正式调度使用 8 张 910B 作为 8 个并发槽位。509 个正式实验单元按 6 个用户可见波次提交；每个内部轮次最多拆为 16 个逻辑切片，每个实验 cell 始终只占用 1 张 NPU。

不要把单 cell 的 `--num-devices` 改成 8。这里的 8 卡用于并行运行 8 个独立实验，不是把同一个实验错误地复制到 8 卡。

调度器保证：

- 一个设备同一时刻只运行一个 shard；
- 最多 8 个 shard 并行，先结束的设备动态领取下一片；
- 任一 shard 失败后停止派发，并终止同轮仍在运行的兄弟进程；
- 每个下游正式内部轮次必须看到上一轮真实 `READY`；
- `--dry-run` 和 `--preflight-only` 不能满足正式依赖；
- NPU 轮次启动前逐卡执行真实 forward/backward 探针；
- 最终门禁逐个核对 509 个正式 cell 均为 `COMPLETED`。

P0-02 另外启用 Phase1 基础产物缓存。同一模型对、数据集和 Phase1 配置只计算一次；`ours/random/sft1` 和不同种子的 PhaseF 训练、评测、ledger 仍逐 cell 独立。缓存减少重复计算，不改变论文实验数与正式训练预算。

## 2. 固定路径

```bash
export SAFETY_SOURCE_ROOT=/home/work/user-job-dir/app/safety-src-886760f
export SAFETY_MODEL_ROOT=/opt/dpcvol/models/safetytransfer
export SAFETY_DATA_ROOT=/opt/dpcvol/datasets/safetytransfer
export SAFETY_OUTPUT_ROOT=/opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f
export SAFETY_ASSET_MANIFEST="$SAFETY_SOURCE_ROOT/configs/ablations/assets.modelmate.template.json"
export SAFETY_ACTIVATE="$SAFETY_MODEL_ROOT/_runtime_assets/activate-safety-ablation.sh"
```

模型、数据、环境、缓存和训练输出都必须位于 `/opt/dpcvol`。Notebook 工作区仅保存精简代码和启动文件。

源码快照至少包含：

```text
src/ablations/modelmate_pool.py
src/ablations/foundation_cache.py
scripts/35_modelmate_8card_pool.py
scripts/36_modelmate_ablation_final_gate.py
scripts/37_modelmate_ablation_campaign.py
configs/ablations/catalog.yaml
configs/ablations/assets.modelmate.template.json
```

Notebook 工作区根目录包含普通文件：

```text
run_safety_8card.py
boot_safety_8card.sh
```

## 3. ModelMate 表单

- Worker 节点：`1`
- 规格：`8*ASCEND_910B | 176U | 880G`
- 启动文件：`/home/work/user-job-dir/app/run_safety_8card.py`
- 训练数据集：若平台强制必选，可选已有小型占位数据集；正式数据由 `SAFETY_DATA_ROOT` 读取
- 环境变量：填写第 2 节 6 个变量
- 运行参数：只填写一个当前波次，例如 `--wave canary`

启动脚本默认使用 `--devices 8 --logical-shards 16`。父进程保留平台下发的 `ASCEND_RT_VISIBLE_DEVICES`，每个 worker 只获得其中一个设备条目，现有 `npu:0` 会映射到不同物理卡。

## 4. 六个波次

| 顺序 | 运行参数 | 内部内容 | 正式 cell |
|---:|---|---|---:|
| 1 | `--wave canary` | 8 卡 PAN 短预算连通性测试 | 0 |
| 2 | `--wave p0` | core、WJB、fairness、evaluate、analyze | 324 |
| 3 | `--wave p0-manual` | 盲化人工 judge agreement | 3 |
| 4 | `--wave p1` | mechanism、data、evaluate、analyze | 125 |
| 5 | `--wave p2` | generalization、evaluate、analyze | 57 |
| 6 | `--wave final` | 509 cell 最终门禁 | 0 |

`canary` 与正式 ledger 隔离，只使用 PAN 和 Qwen3.5 9B→0.8B。8 张卡各运行一个短序列、每标签 8 样本、1 epoch 的端到端检查。它不是所有模型或数据集的 Phase1，也不能作为论文结果。

`p0` 仍完整覆盖 P0-02 的 6 个数据集、3 种方法和 3 个种子，并执行其余 P0 实验。一个波次内部会按原依赖顺序串行启动多个内部轮次；某一轮失败后不会继续后面的轮次。

`p0-manual` 需要 `human_annotations` 中已有盲化双人标注和 WildGuard 预测。若尚未完成，应按预期阻塞，不能伪造 READY。

## 5. 预检与正式启动

先提交：

```text
--wave canary --preflight-only
```

预检成功后删除 `--preflight-only`，正式提交：

```text
--wave canary
```

canary 成功后，按表中顺序逐个提交其余波次。不要同时创建相互依赖的多个波次作业。

内部轮次的预检报告位于：

```text
$SAFETY_OUTPUT_ROOT/jobs/<internal-round>/preflight-summary.json
$SAFETY_OUTPUT_ROOT/jobs/<internal-round>/device-preflight.json
```

波次汇总位于：

```text
$SAFETY_OUTPUT_ROOT/jobs/campaign-<wave>/wave-summary.json
```

## 6. 成功、失败与续跑

内部轮次只有同时满足以下条件才成功：

- `pool-summary.json.status == "READY"`
- `pool-summary.json.dry_run == false`
- `failed_shards == []`
- `pending_shards == []`
- `status.json` 中该轮全部 cell 为 `COMPLETED`

单 shard 日志：

```text
$SAFETY_OUTPUT_ROOT/jobs/<internal-round>/shards/shard-*/worker.log
```

作业异常终止时，用完全相同的 `--wave` 重提。已完成 cell 会由持久 ledger 跳过；完整 Phase1 会由 `foundation-cache` 复用。不要删除 `run-state` 或 `foundation-cache`，不要改 plan、资产清单、源代码版本或输出根目录。

## 7. 最终门禁

前 5 个波次成功后提交：

```text
--wave final
```

只有最终报告同时包含以下值，才能声明全部消融完成：

```text
"status": "READY"
"expected_cells": 509
"covered_cells": 509
```

审计文件：

```text
/opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f/final-completion.json
```
