# ICLR 全量消融实验系统设计

日期：2026-08-12  
状态：已获用户批准  
权威需求：`ICLR_细分消融实验设计.html` 中 35 项 P0/P1/P2 实验目录

## 1. 目标

在不推翻现有 Phase A–F 训练流水线的前提下，将 HTML 中全部 35 项消融变成可规划、可预检、可运行、可断点续跑、可审计和可汇总的代码路径。

“代码完成”不等于“论文结论成立”。本项目交付的是完整实验能力及验证门禁：

- 需要模型、数据、人工标注或外部 benchmark 的单元，在资产齐全时可执行；
- 资产不齐时必须在预检阶段显示 `BLOCKED` 和精确原因；
- dry-run、预检或缺少真实训练结果的单元不得标记为 `COMPLETED`；
- 只有满足输出契约、指纹契约和评测契约的真实运行才可进入汇总。

## 2. 范围

### 2.1 正式模型对

1. `qwen35_9b_to_08b`
2. `llama31_8b_to_1b`
3. `qwen3_8b_to_06b`
4. `qwen3_8b_to_4b`
5. `qwen3_4b_to_06b`

跨家族实验允许额外模型对，但必须显式声明 tokenizer bridge 和所需资产，不能混入上述五个主表模型对。

### 2.2 正式数据集

1. `pan`
2. `safety_tuned_llamas`
3. `coconot`
4. `c5`
5. `wildjailbreak`
6. `wildguardmix`

BeaverTails、TuluSafety、HH-RLHF 等历史配置不进入正式 5×6 主矩阵，但保留兼容性。

### 2.3 主表方法

- `ours`
- `sft1`
- `sft`
- `distill`
- `nosft`

因此主表基础矩阵必须精确展开为 `5 × 6 × 5 = 150` 个唯一单元。

### 2.4 完整消融目录

系统必须覆盖以下 35 个实验 ID，且 catalog 覆盖测试要求集合完全相等：

- P0：`P0-01` 至 `P0-08`
- P1：`P1-01` 至 `P1-20`
- P2：`P2-01` 至 `P2-07`

## 3. 方案选择

采用“声明式实验目录 + 统一规划器 + 可组合策略组件”，而不是继续向 `15_run_oneclick.py` 和 shell 脚本堆叠条件分支，也不在投稿周期内迁移到新的工作流框架。

理由：

- 35 项实验跨越数据、表示、层选择、子空间、语义桥、配对、损失、评测和统计，简单分支难以证明覆盖完整；
- 同一 Phase1 artifact 只可被兼容配置复用，必须有规范化 artifact key；
- NPU 平台任务需要可拆分的计划清单、单写者输出目录和断点续跑；
- 外部/人工依赖必须在统一预检中显式处理。

## 4. 总体架构

### 4.1 统一入口

新增：

```bash
python scripts/30_ablation.py catalog
python scripts/30_ablation.py plan --scope all --output <plan.jsonl>
python scripts/30_ablation.py preflight --plan <plan.jsonl>
python scripts/30_ablation.py run --plan <plan.jsonl>
python scripts/30_ablation.py status --plan <plan.jsonl>
python scripts/30_ablation.py summarize --plan <plan.jsonl>
```

命令语义：

- `catalog`：验证并展示实验定义；
- `plan`：只做纯函数式矩阵展开，不加载模型；
- `preflight`：检查资产、配置、数据隔离、设备、输出冲突和外部依赖；
- `run`：执行指定 cell，默认不隐式运行整个昂贵矩阵；
- `status`：从不可变计划和单元状态文件汇总状态；
- `summarize`：只读取已完成且契约有效的结果。

### 4.2 包结构

新增 `src/ablations/`：

- `schema.py`：严格 dataclass/枚举和验证错误；
- `catalog.py`：加载并验证 `configs/ablations/catalog.yaml`；
- `planner.py`：展开实验矩阵、canonical JSON 和稳定 ID；
- `artifacts.py`：artifact key、文件哈希、模型/数据 provenance；
- `preflight.py`：环境、资产、数据泄漏和输出目录检查；
- `runner.py`：将声明式 cell 翻译为现有 Phase A–F 命令；
- `ledger.py`：原子状态更新、断点续跑和单写者锁；
- `statistics.py`：paired bootstrap、McNemar、Holm、Cohen's kappa；
- `analysis.py`：分组分析、跨语料矩阵、表示相似性、效率汇总；
- `strategies/`：表示、层选择、子空间、桥接、配对、目标和损失策略。

现有 Phase A–F 脚本继续作为执行后端；新增策略通过明确配置字段进入后端，而不是复制整条流水线。

## 5. 数据模型

### 5.1 ExperimentDefinition

每个 catalog 项必须包含：

- `id`、`priority`、`family`；
- `question`、`hypothesis`；
- `execution_kind`：`train`、`evaluate`、`analyze`、`manual`；
- `axes`：模型对、数据集、方法、seed、draw 和 sweep；
- `overrides`：各阶段策略覆盖；
- `requires`：模型、数据、benchmark、人工输入、历史 checkpoint；
- `metrics`；
- `completion_contract`；
- `blocked_policy`。

未知字段、未知策略、空轴、重复 sweep 值和不合法 ID 均 fail-fast。

### 5.2 ExperimentCell

计划展开后每个 cell 包含：

- 稳定 `cell_id`；
- `experiment_id`、模型对、数据集、方法、seed、draw；
- 规范化后的全部配置；
- Phase1/PhaseF/eval artifact key；
- 唯一输出目录；
- 依赖 cell；
- 预期输出和完成条件。

`cell_id` 由 canonical JSON 的 SHA-256 派生，不依赖字段输入顺序。

### 5.3 状态机

允许状态：

`PLANNED → BLOCKED | READY → RUNNING → COMPLETED | FAILED`

规则：

- `BLOCKED` 必须保存结构化原因，可在资产变化后重新预检；
- `RUNNING` 使用锁和原子写，防止两个作业写同一目录；
- `COMPLETED` 仅在所有 completion artifacts、hash 和解析校验通过后写入；
- 配置或输入指纹变化时旧状态不可复用。

## 6. 策略组件

### 6.1 表示位置

统一 `representation.position`：

- `last_prompt`
- `mean_prompt`
- `first_generated`
- `first_4_generated_mean`

Phase1 hidden extraction、目标构造和 PhaseF layer loss 必须使用同一位置语义。manifest 保存 token position 和实际生成长度。

### 6.2 层选择

统一 `layer_selection.mode`：

- `effect_probe_sum`
- `effect_only`
- `probe_only`
- `random_k`
- `evenly_spaced`
- `last_k`

随机策略必须显式 `draw_seed`；所有 K 层策略验证数量相同且层号唯一有效。

### 6.3 安全子空间

统一 `subspace.mode`：

- `learned`
- `none`
- `random_orthogonal`
- `bootstrap`

随机正交基与 learned U 同维度、同 rank；bootstrap 重新抽样、重算 layer score 和 U，输出 layer Jaccard、principal angles 和 projection overlap。

### 6.4 语义桥

统一 `bridge.mode`：

- `vocabulary`
- `token_string`
- `embedding_nearest`
- `ridge`
- `orthogonal_procrustes`

共享词表模型仍可使用 vocabulary index；跨 tokenizer 禁止直接复用 token ID。token-string 和 embedding 模式必须报告匹配覆盖率、冲突率和未匹配率。Ridge/Procrustes 只在 alignment split 拟合，不能读取 test。

### 6.5 层配对

统一 `pairing.mode`：

- `relative_depth`
- `cka_nearest`
- `random_permutation`
- `same_index_clamped`

CKA 只使用 alignment set。随机置乱保持 teacher layer 集和 student layer 数量不变。

### 6.6 目标控制

统一 `target.mode`：

- `semantic`
- `random_same_norm`
- `within_label_permutation`
- `cross_label_permutation`
- `raw_teacher`

置乱必须在稳定样本 ID 上完成并保存 permutation manifest；同一 seed 可复现，不同 seed 产生不同映射。

### 6.7 Loss 与监督

统一 `loss.kind`：

- `cosine`
- `normalized_mse`
- `raw_mse`
- `margin_contrastive`

监督策略：

- `harmful_only`
- `all`
- `label_weighted`
- `harmless_anchor`

所有 loss 在零向量、单类 batch、空 mask、混合精度和无 layer target 样本上有明确行为，不产生 NaN。

## 7. 35 项实验映射

### 7.1 P0

- `P0-01`：150 单元 provenance 与缺失单元审计；
- `P0-02`：matched SFT/random/Full，主模型 42/43/44；
- `P0-03`：WildGuard 重评、disagreement 包、人工回传验证；
- `P0-04`：训练 seed 汇总和 10k paired bootstrap；
- `P0-05`：validation-only ISO-HR checkpoint 匹配；
- `P0-06`：WJB global/override、curation、target 与 common-test 分析；
- `P0-07`：全局固定配置和搜索预算账本；
- `P0-08`：train-corpus × common-test 矩阵。

### 7.2 P1

- `P1-01`：类内和跨类 target permutation；
- `P1-02`：no projection/raw teacher；
- `P1-03`：同 rank 随机正交子空间 5 draws；
- `P1-04`：ridge/Procrustes；
- `P1-05`：selected/random/even/last-K；
- `P1-06`：effect/probe/sum/random 层评分；
- `P1-07`：四种层配对；
- `P1-08`：四种表示位置；
- `P1-09`：top-M 六点曲线及 retained energy；
- `P1-10`：abs/positive/negative 和 token filter；
- `P1-11`：六个 layer-loss 权重；
- `P1-12`：五种监督策略；
- `P1-13`：四种 loss；
- `P1-14`：LoRA rank、模块和参数匹配；
- `P1-15`：τ 与 rank cap；
- `P1-16`：Phase1/PhaseF 两条数据效率曲线；
- `P1-17`：balance 与 curation；
- `P1-18`：PAN jailbreak/benign 分组；
- `P1-19`：通用能力统一重评；
- `P1-20`：pre/post target similarity 与行为相关。

### 7.3 P2

- `P2-01`：20 次子空间 bootstrap；
- `P2-02`：key/random layer 上 ±U 强度干预；
- `P2-03`：teacher 大小/安全性控制；
- `P2-04`：token-string/embedding 跨 tokenizer 桥；
- `P2-05`：temperature/top-p/max-token 重评；
- `P2-06`：阶段时间、峰值内存、磁盘和 NPU-hours；
- `P2-07`：license、用途、去重、split overlap、target source、模板多样性审计。

## 8. Artifact 复用与污染防护

Phase1 key 至少包含：

- teacher/student checkpoint 和 tokenizer 指纹；
- dataset split hash 与 curation manifest；
- seed、representation position、layer selection；
- subspace、bridge、pairing、target 策略及参数；
- 代码 commit 和 schema version。

PhaseF key 额外包含 loss、监督策略、LoRA、优化器和训练预算。

不兼容 artifact 必须拒绝复用。软链接、同名目录和旧 manifest 不能绕过 key 校验。

## 9. 数据与评测门禁

### 9.1 Split 隔离

按规范化 prompt hash 检测 train/validation/test/common-test 交叉泄漏。PAN、STL、WGM 等已知问题必须由数据构建代码消除；预检对历史缓存继续 fail-fast。

数据内部重复与跨 split 泄漏分开报告：默认不静默删除训练内部重复，但正式运行不允许 train-test overlap。

### 9.2 选模与评测

- epoch 和超参只读 validation；
- ISO-HR operating point 只在 validation 匹配；
- test 只用于最终报告；
- WildGuard 与 keyword 指标分开；
- 逐样本 ID 必须对齐后才能做 paired statistics。

### 9.3 统计

- paired bootstrap 默认 10,000 次，显式 seed；
- seed 结果报告 mean ± std；
- 分类差异支持 McNemar；
- 多重检验使用 Holm；
- judge agreement 支持 Cohen's κ；
- 样本不对齐、单类标签或解析失败率过高时拒绝输出误导性统计量。

## 10. 外部和人工依赖

以下能力提供完整接口但不能伪造输入：

- 300 条双人人工审计：导出盲化 JSONL/CSV、校验双人回传、计算 agreement；
- MMLU/GSM8K/IFEval/HumanEval/MBPP：检查本地 benchmark 资产或显式 runner；
- 跨家族 checkpoint：要求 catalog 中的模型和 tokenizer 指纹；
- license 信息：要求数据 registry 中显式声明来源和许可。

缺失时 cell 为 `BLOCKED`，不会被汇总为负结果或零分。

## 11. 平台适配

- `SAFETY_MODEL_ROOT`、`SAFETY_DATA_ROOT`、`SAFETY_OUTPUT_ROOT` 覆盖配置路径；
- 不修改平台提供的 `ASCEND_RT_VISIBLE_DEVICES`，仅使用进程内逻辑设备；
- 输出必须在持久目录；
- 提交包检查模型、数据、输出、符号链接、单文件/总包大小；
- 每个 cell 一个输出目录、一个 lock、一个状态文件；
- 所有命令支持 `--dry-run`，但 dry-run 状态不升级为完成。

## 12. 错误处理

错误分为：

- `CatalogError`：定义不合法；
- `PlanError`：矩阵空、重复或输出冲突；
- `PreflightError`：资产、数据、平台或 provenance 不满足；
- `ExperimentBlocked`：合法但当前缺外部/人工依赖；
- `ExecutionError`：子阶段失败；
- `ArtifactError`：输出缺失、损坏或指纹不一致。

错误必须保存阶段、cell ID、可操作原因和安全重试建议，不包含访问令牌。

## 13. 测试策略

### 13.1 单元测试

- catalog 35/35 和 150 主表单元；
- canonical ID、重复输出、依赖 DAG；
- 各策略的小 tensor 数值性质；
- permutation 可复现性与标签约束；
- random U 正交性；
- Procrustes/ridge 恢复已知映射；
- CKA 配对和位置抽取；
- loss 的 mask、dtype、NaN 边界；
- statistical routines 对已知小样本结果；
- provenance 和数据 overlap 门禁。

### 13.2 集成测试

- 使用小型 fake runner 完成 `plan → preflight → run → status → summarize`；
- CPU 小 tensor 贯通策略到 PhaseF loss；
- 中断后恢复、锁冲突、失败重试和 artifact 失配；
- 外部/人工依赖正确进入 `BLOCKED`；
- 平台路径覆盖和提交包检查。

### 13.3 回归与 dry-run

- 原测试按文件分组执行，记录基线超时；
- 全部 35 项能生成合法计划；
- 五模型对、六数据集、五方法主表精确为 150；
- dry-run 不加载真实模型且不写 `COMPLETED`。

## 14. 审查门禁

实施期间进行三类重复审查：

1. 规格审查：35/35、150 单元、每项 completion contract；
2. 算法/边界审查：数值正确性、split 污染、test tuning、随机性和 artifact 复用；
3. 工程/平台审查：测试、断点、锁、错误、路径、包大小和敏感信息。

任何 Critical/Important 问题必须修复并新增回归测试后才能进入最终交付。

## 15. 非目标

- 不在本地 CPU 环境假装完成 NPU 训练；
- 不自动接受新的模型许可证或下载 gated 权重；
- 不生成虚假的人工标注、benchmark 分数或置信区间；
- 不删除用户现有模型、数据、输出或未跟踪文件；
- 不保证每个研究假设得到正结果，只保证实验设计可正确检验该假设。
