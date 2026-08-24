# ICLR 全量消融运行手册

统一入口是 `scripts/30_ablation.py`，ModelMate 单作业包装器是
`scripts/33_modelmate_ablation_job.py`。平台持久目录、资产清单、分片命令和执行波次见
[`../ABLATION_MODELMATE_RUNBOOK.md`](../ABLATION_MODELMATE_RUNBOOK.md)，本文件定义所有环境通用的
生命周期与完成门禁。

## 固定规模

- 35/35 个 P0/P1/P2 实验 ID；
- 主表 provenance 精确 150 个单元（5 模型对×6 数据集×5 方法）；
- Full 正式计划 360 个单元：140 train、31 evaluate、186 analyze、3 manual；
- Extended 扩展计划仍可显式生成 509 个单元：289 train、31 evaluate、186 analyze、3 manual；
- Full 的 P0-07 聚焦 WJB/WGM，2 配置 × 3 方法，共 12 个公平预算单元；
- P0-07 之前另有 12 个验证集搜索候选（不计入正式 360 单元），由
  `scripts/34_fairness_search.py` 逐任务训练、WildGuard 验证并生成不可手写的证据账本；
- P0-02 为 6 数据集×3 matched 方法×3 seed，共 54 个训练单元；
- P1-01～04 保留主文组件，随机子空间与层选择的多次随机对照聚焦 PAN；其余 Full 机制实验显式限定 PAN。
- `P1-06/10/12/14/15/17` 属于 Extended；它们有完整实现，但不计入 Full 的 140 次正式训练。

## 生命周期

```bash
python scripts/30_ablation.py catalog
python scripts/30_ablation.py plan --scope full --output-root "$SAFETY_OUTPUT_ROOT/cell-outputs" --output plan.jsonl
python scripts/30_ablation.py preflight --plan plan.jsonl --asset-manifest assets.json --output preflight.json
python scripts/30_ablation.py run --plan plan.jsonl --cell-id CELL_ID --state-root "$SAFETY_OUTPUT_ROOT/state" --asset-manifest assets.json --device npu
python scripts/30_ablation.py status --plan plan.jsonl --state-root "$SAFETY_OUTPUT_ROOT/state" --output status.json
python scripts/30_ablation.py summarize --plan plan.jsonl --state-root "$SAFETY_OUTPUT_ROOT/state" --output summary.json
```

正式 `run` 必须显式指定单 cell 或有界 shard；禁止一次隐式启动全部训练。缺模型、缺数据、split
污染、历史 checkpoint/人工标注未准备好时，`preflight` 必须报告 `BLOCKED`，不能创建虚假
`COMPLETED`。dry-run 使用独立状态根，不能与正式运行复用。

## 顺序

1. 修复 split overlap 后运行 `core-train`、`wjb`，再完成 12 个 P0-07 validation-only
   搜索候选并收集账本，最后运行 12 个正式 `fairness` 单元；
2. 生成不可变 checkpoint/model/prediction registries；
3. 运行 `evaluate`；
4. 汇集逐样本结果、日志、hidden states 与 manifests；
5. 运行 `analyze`；
6. P0-03 盲化导出、双人标注回传后运行 `manual`；
7. 只有 completion artifacts、哈希和解析校验全部通过，cell 才算完成。

Full 的 P0-06 先在主 Qwen3.5 9B→0.8B 对上验证 2 配置×3 curation 的失败边界；五模型对
全因子复现属于 Extended。P2-04 跨 tokenizer 仍依赖 Llama-3.2-1B-Instruct。
