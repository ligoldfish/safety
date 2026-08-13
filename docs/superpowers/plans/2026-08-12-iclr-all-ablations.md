# ICLR 全量消融实验系统实施计划

> **执行要求：** 严格测试先行。每个任务先运行聚焦测试并观察预期失败，再做最小实现，随后运行聚焦测试和相关回归。所有昂贵训练仅生成/验证计划，不在本地 CPU 伪运行。

**目标：** 为 `ICLR_细分消融实验设计.html` 的 35 项 P0/P1/P2 实验提供完整、可审计的统一代码路径，并保持现有 Phase A–F 流水线兼容。

**架构：** 新增 `src/ablations` 声明式实验层，负责 schema、catalog、规划、preflight、artifact/ledger、策略、统计、分析和 runner；现有脚本继续承担真实模型计算。新增 `scripts/30_ablation.py` 作为唯一编排入口，策略字段通过配置覆盖进入既有 Phase 脚本。

**技术栈：** Python 3.10+、PyTorch、PyYAML、NumPy（若可用）；测试使用 `unittest`，不要求网络或真实模型。

---

## Task 1：严格 schema、35 项 catalog 与 150 单元主矩阵

**文件：**

- 新增：`src/ablations/__init__.py`
- 新增：`src/ablations/schema.py`
- 新增：`src/ablations/catalog.py`
- 新增：`src/ablations/planner.py`
- 新增：`configs/ablations/catalog.yaml`
- 新增：`tests/test_ablation_catalog.py`
- 新增：`tests/test_ablation_planner.py`

**RED：**

1. 写测试断言 catalog ID 集合精确等于 P0-01..08、P1-01..20、P2-01..07。
2. 写测试断言正式 pair=5、dataset=6、method=5，主表精确 150 个唯一 cell。
3. 写测试断言 canonical cell ID 与 mapping 字段顺序无关。
4. 写测试断言重复 ID、未知策略、空轴、重复输出目录和依赖环被拒绝。
5. 运行：

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe -m unittest tests.test_ablation_catalog tests.test_ablation_planner -v
```

预期：模块不存在或断言失败。

**GREEN：**

1. 用 frozen dataclass/枚举实现严格 schema；禁止未知字段和 bool 冒充 int。
2. catalog YAML 完整编码 35 项定义、轴、策略、依赖、metrics 和 completion contract。
3. planner 实现笛卡尔展开、canonical JSON、SHA-256 cell ID、输出目录唯一性、依赖 DAG 校验。
4. 只把 `P0-01/main_table` 展开为 150 主表单元；其余按各自轴展开，防止把 35 项与主表重复相乘。
5. 复跑聚焦测试。

## Task 2：artifact provenance、状态账本与断点续跑

**文件：**

- 新增：`src/ablations/artifacts.py`
- 新增：`src/ablations/ledger.py`
- 新增：`tests/test_ablation_artifacts.py`
- 新增：`tests/test_ablation_ledger.py`

**RED：**

1. 测试文件 SHA-256、目录 manifest、canonical config hash。
2. 测试 Phase1 key 对模型/tokenizer/data/position/subspace/bridge/pairing/commit 任一变化敏感。
3. 测试合法状态转换；dry-run 不能变为 `COMPLETED`。
4. 测试原子 JSON、锁冲突、stale RUNNING 检测、artifact 指纹不匹配拒绝恢复。

**GREEN：**

实现 streaming hash、非敏感 provenance、`PLANNED/BLOCKED/READY/RUNNING/COMPLETED/FAILED` 状态机、原子 replace 和跨平台 lock 文件。

## Task 3：表示位置与层选择策略

**文件：**

- 新增：`src/ablations/strategies/__init__.py`
- 新增：`src/ablations/strategies/representation.py`
- 新增：`src/ablations/strategies/layers.py`
- 修改：`src/features/first_gen_token.py`
- 修改：`scripts/01_extract_hidden_states.py`
- 修改：`scripts/02_analyze_teacher_layers.py`
- 新增：`tests/test_ablation_representation.py`
- 新增：`tests/test_ablation_layers.py`

**RED：**

1. 用带 padding 的小 tensor 验证 `last_prompt`、`mean_prompt`、`first_generated`、`first_4_generated_mean`。
2. 验证 generated 模式不足 token 时使用有效 token，零有效 token fail-fast。
3. 验证 effect/probe/sum、random×seed、evenly-spaced、last-K，K 边界和唯一性。

**GREEN：**

实现共享 position extractor 和 layer selector；Phase1 manifest 保存模式、位置和 draw seed；现有默认保持 `last_prompt` + sum score。

## Task 4：子空间与 bootstrap 稳定性

**文件：**

- 新增：`src/ablations/strategies/subspace.py`
- 修改：`src/features/subspace.py`
- 修改：`scripts/03_build_teacher_safe_subspace.py`
- 新增：`src/ablations/stability.py`
- 新增：`tests/test_ablation_subspace.py`
- 新增：`tests/test_ablation_stability.py`

**RED：**

1. learned/no-projection/random-orthogonal 的维度、正交性、rank、seed 性质。
2. principal angle、projection overlap 和 layer Jaccard 在相同/正交子空间上的已知值。
3. 20 bootstrap draw 的稳定输出 schema 和独立 seed。

**GREEN：**

实现 QR-based 随机 U、无投影 identity contract、bootstrap resampling 和稳定性指标；manifest 区分真实 U 与控制 U。

## Task 5：语义桥与跨 tokenizer 支持

**文件：**

- 新增：`src/ablations/strategies/bridge.py`
- 修改：`src/features/semantic_basis.py`
- 修改：`src/features/semantic_recompose.py`
- 修改：`scripts/05_build_semantic_bases.py`
- 修改：`scripts/08_recompose_student_targets.py`
- 新增：`tests/test_ablation_bridge.py`

**RED：**

1. ridge 在已知线性映射上恢复解，并只使用 alignment 数据。
2. orthogonal Procrustes 恢复已知正交映射。
3. token-string 处理不同 ID、重复字符串、special token 和未匹配 token。
4. embedding-nearest 输出覆盖率/冲突率/未匹配率并有阈值门禁。
5. 跨 tokenizer 模式禁止 vocabulary-index 直连。

**GREEN：**

实现五种 bridge，统一 `BridgeArtifact` 和匹配审计；现有共享词表默认行为不变。

## Task 6：层配对、target controls 与四种 loss

**文件：**

- 新增：`src/ablations/strategies/pairing.py`
- 新增：`src/ablations/strategies/targets.py`
- 新增：`src/ablations/strategies/losses.py`
- 修改：`src/features/layer_pairing.py`
- 修改：`scripts/04_pair_layers.py`
- 修改：`src/training/losses.py`
- 修改：`src/training/trainer_phase1.py`
- 修改：`scripts/09_train_student_semalign.py`
- 新增：`tests/test_ablation_pairing.py`
- 新增：`tests/test_ablation_targets.py`
- 新增：`tests/test_ablation_losses.py`

**RED：**

1. relative/CKA/random/same-index 配对数值与 seed。
2. `random_same_norm`、`within_label_permutation`（类内置乱）、`cross_label_permutation`（跨类置乱）保持/改变标签约束，stable sample ID 可复现。
3. cosine/normalized MSE/raw MSE/margin contrastive 的已知值、mask、零向量、单类和 fp16 边界。
4. harmful-only/all/label-weighted/harmless-anchor 权重行为。

**GREEN：**

实现并接入现有 trainer；默认 cosine + harmful-only 保持兼容。permutation manifest 保存 sample ID 映射，禁止按文件行号隐式置乱。

## Task 7：统一 preflight、模型/数据路径与平台提交检查

**文件：**

- 新增：`src/ablations/preflight.py`
- 新增：`src/ablations/platform.py`
- 修改：`src/models/hf_loader.py`
- 修改：`src/utils/config.py`
- 修改：`src/baselines/config.py`
- 新增：`tests/test_ablation_preflight.py`
- 新增：`tests/test_ablation_platform.py`

**RED：**

1. `SAFETY_MODEL_ROOT/DATA_ROOT/OUTPUT_ROOT` 路径覆盖，不修改源 YAML。
2. 模型 config/tokenizer/weights、数据文件和依赖 checkpoint 缺失进入结构化 `BLOCKED`。
3. 不覆盖已有 `ASCEND_RT_VISIBLE_DEVICES`。
4. 提交包检测模型/数据/输出、符号链接、单文件/总大小超限。
5. token/密码不进入错误文本或 manifest。

**GREEN：**

实现只读预检和路径解析；所有错误具有 cell ID、类别和建议，不执行网络下载。

## Task 8：数据 split 泄漏修复与伦理审计

**文件：**

- 新增：`src/ablations/data_audit.py`
- 修改：`src/data/dataset_io.py`
- 修改：`src/data/safety_datasets.py`
- 修改：`scripts/19_prepare_safety_data.py`
- 修改：`scripts/21_build_baseline_eval_jsonls.py`
- 新增：`tests/test_ablation_data_audit.py`
- 修改：`tests/test_baseline_eval_jsonls_smoke.py`
- 修改：`tests/test_safety_datasets_smoke.py`

**RED：**

1. PAN 原始重复 prompt 不能跨 train/test。
2. STL eval 样本必须从 train 移除；不再允许 496/496 泄漏。
3. WGM 官方 split prompt 重叠必须从 train 排除。
4. 数据内部重复与跨 split overlap 分开报告。
5. license、用途、target_source、template diversity 和 drop reason schema。

**GREEN：**

以规范化 prompt SHA-256 做稳定 split exclusion；写 split manifest 和审计 JSON；不静默改变非泄漏的内部重复策略。

## Task 9：统计、judge agreement、ISO-HR 与分组分析

**文件：**

- 新增：`src/ablations/statistics.py`
- 新增：`src/ablations/analysis.py`
- 修改：`src/eval/iso_hr.py`
- 修改：`scripts/23_iso_hr_compare.py`
- 新增：`scripts/31_export_manual_audit.py`
- 新增：`scripts/32_import_manual_audit.py`
- 新增：`tests/test_ablation_statistics.py`
- 新增：`tests/test_ablation_analysis.py`
- 新增：`tests/test_ablation_manual_audit.py`

**RED：**

1. paired bootstrap known sample、固定 seed、样本 ID 错位拒绝。
2. McNemar、Holm 和 Cohen's κ 对已知列联表。
3. validation-only checkpoint/ISO-HR 选择；传入 test selector fail-fast。
4. PAN jailbreak/benign bucket、train-corpus×common-test、pre/post cosine correlation。
5. 人工审计盲化、分层抽样、双标注回传和缺失/重复 ID 校验。

**GREEN：**

实现统计与分析纯函数，结果保存 seed、N、CI 和 parse rate；人工标签只从校验后的回传读取。

## Task 10：runner、CLI、外部 benchmark 与效率采集

**文件：**

- 新增：`src/ablations/runner.py`
- 新增：`src/ablations/efficiency.py`
- 新增：`src/ablations/benchmarks.py`
- 新增：`scripts/30_ablation.py`
- 新增：`tests/test_ablation_runner.py`
- 新增：`tests/test_ablation_cli.py`
- 新增：`tests/test_ablation_efficiency.py`

**RED：**

1. catalog/plan/preflight/status/summarize 无模型运行。
2. fake runner 完成完整状态机；失败、中断和恢复。
3. 每个策略 cell 翻译为正确 Phase 命令/override；参数含空格时 argv 安全。
4. dry-run 不写完成；未知 cell/重复 writer 拒绝。
5. benchmark 缺资产为 BLOCKED；明确 runner 时生成相同 decode 配置。
6. 阶段 wall time、peak memory、disk delta 和 device-hours schema。

**GREEN：**

实现 subprocess argv runner（不使用 shell 拼接）、阶段 hook、结构化日志和原子 result。真实 run 默认要求 `--cell-id` 或显式有限选择，防止误启动全矩阵。

## Task 11：35 项 completion contract 与端到端覆盖门禁

**文件：**

- 新增：`tests/test_ablation_coverage.py`
- 新增：`tests/test_ablation_e2e.py`
- 新增：`tests/fixtures/ablations/` 下小型 catalog/artifacts
- 修改：`configs/ablations/catalog.yaml`
- 修改：`src/ablations/catalog.py`
- 修改：`src/ablations/runner.py`

**RED：**

1. 每个 P0/P1/P2 项必须有非空 requires、metrics、completion artifacts 和 runner/analysis handler。
2. 禁止 `document_only` 或空 handler 假覆盖。
3. 35 项全计划 dry-run；主表 150；输出目录全唯一。
4. fake 资产端到端覆盖 READY/BLOCKED/COMPLETED/FAILED。

**GREEN：**

补齐所有 handler 和 completion validator；输出机器可读 coverage report，要求 35/35 executable contract。

## Task 12：回归、三轮审查、文档与整合

**文件：**

- 修改：`README.md`
- 新增：`docs/ablations/ICLR_ALL_ABLATIONS_RUNBOOK.md`
- 新增：`docs/ablations/EXPERIMENT_COVERAGE.md`
- 按审查结果修改相关源码和测试

**验证顺序：**

1. 聚焦全部新增测试：

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe -m unittest discover -s tests -p "test_ablation*.py" -v
```

2. 原测试按文件分组，避免 120 秒外层超时；每组保存退出码。
3. 完整 tests，使用更高但有限的外层超时。
4. `catalog validate`、`plan --scope main-table`、`plan --scope all`、`preflight` fake 资产。
5. `git diff --check`、敏感字段扫描、工作树状态和提交内容审查。

**三轮审查：**

- 规格审查：逐项对照 HTML 35 项，验证 35/35 和 150；
- 算法审查：张量形状、随机性、统计、split、validation/test 边界、artifact reuse；
- 工程审查：状态机、锁、错误、路径、提交包、日志敏感信息、旧入口兼容。

每个发现先新增失败回归测试，再修复，直至无 Critical/Important。

**最终交付条件：**

- 所有新增与旧测试有新鲜通过证据；
- 全部 35 项都有真实 handler 和完成契约；
- 主表精确 150；
- 缺资产正确 BLOCKED；
- 无真实 NPU 结果被伪造；
- 分支提交清晰，最终整合回 `D:\code-safety\code-safety` 时保留用户未跟踪文件。
