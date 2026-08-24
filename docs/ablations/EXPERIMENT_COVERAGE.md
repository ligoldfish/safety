# 实验覆盖清单

当前 catalog 覆盖 HTML 的 35/35 项，主表精确 150 单元。Full 正式计划为 360 单元，其中
140 train、31 evaluate、186 analyze、3 manual；Extended 显式扩展计划仍为 509 单元，其中
289 train、31 evaluate、186 analyze、3 manual。下表是代码 handler 与完成产物的审查索引；具体轴值以
`configs/ablations/catalog.yaml` 为唯一事实源。

Full 为 HTML 的 90–140 次训练档上界。`P1-06/10/12/14/15/17` 标记为 Extended；实现与
产物契约均保留，但只有显式 `--scope all` 才会调度，正式六波次使用 `--scope full`。

| ID | 类别 | Handler | 核心完成产物 |
|---|---|---|---|
| P0-01 | provenance | provenance_matrix | provenance_matrix.jsonl |
| P0-02 | matched controls | matched_controls | run_manifest.json |
| P0-03 | judge/manual | judge_agreement_audit | manual_audit_summary.json |
| P0-04 | statistics | seed_and_paired_bootstrap | paired_bootstrap.json |
| P0-05 | ISO-HR | validation_iso_hr | iso_hr_comparison.json |
| P0-06 | WJB boundary | wjb_failure_analysis | failure_analysis.json |
| P0-07 | fairness | global_default_budget | budget_summary.json |
| P0-08 | cross corpus | cross_corpus_matrix | cross_corpus_matrix.json |
| P1-01 | target | target_control | permutation_manifest.json |
| P1-02 | projection | subspace_control | subspace_manifest.json |
| P1-03 | random subspace | subspace_control | subspace_manifest.json |
| P1-04 | direct bridge | bridge_control | bridge_artifact.pt |
| P1-05 | layer placement | layer_selection_control | layer_selection.json |
| P1-06 | layer score | layer_score_control | layer_selection.json |
| P1-07 | pairing | pairing_control | pairing_manifest.json |
| P1-08 | position | representation_position_control | position_manifest.json |
| P1-09 | top-M | semantic_top_m_sweep | semantic_manifest.json |
| P1-10 | sign/filter | semantic_selection_control | semantic_manifest.json |
| P1-11 | loss weight | layer_loss_weight_sweep | training_manifest.json |
| P1-12 | supervision | supervision_policy_control | training_manifest.json |
| P1-13 | loss kind | layer_loss_kind_control | training_manifest.json |
| P1-14 | LoRA capacity | lora_capacity_control | parameter_budget.json |
| P1-15 | subspace hyperparameters | subspace_hyperparameter_sweep | subspace_manifest.json |
| P1-16 | data efficiency | data_efficiency_sweep | sampling_manifest.json |
| P1-17 | curation | curation_control | curation_manifest.json |
| P1-18 | PAN subgroup | pan_subgroup_analysis | pan_subgroups.json |
| P1-19 | utility | general_capability_eval | capability_summary.json |
| P1-20 | correlation | representation_behavior_correlation | correlation.json |
| P2-01 | stability | subspace_bootstrap | bootstrap_stability.json |
| P2-02 | causal | causal_layer_intervention | intervention_curve.json |
| P2-03 | teacher scale | teacher_scale_control | teacher_scale_summary.json |
| P2-04 | cross architecture | cross_architecture_transfer | tokenizer_bridge_audit.json |
| P2-05 | efficiency | efficiency_accounting | efficiency_summary.json |
| P2-06 | error taxonomy | error_taxonomy_audit | error_taxonomy.json |
| P2-07 | ethics/data | ethics_and_data_audit | ethics_data_audit.json |

操作顺序固定为 `catalog → plan → preflight → run → status → summarize`。任何依赖或契约缺失都应
保留为 `BLOCKED/FAILED`，不能从 dry-run 或空占位文件生成完成结论。
