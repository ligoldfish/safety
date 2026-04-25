# Qwen3.5 Safety Pipeline

This workspace now keeps the proposal-aligned `00 -> 11` phase-1 pipeline for the Qwen3.5-0.8B safety experiment, plus a smaller exploratory intervention path.

## Environment

- Python: `D:\\Anaconda3\\envs\\pytorch-cpu\\python.exe`
- Model: `D:\\safety\\models\\Qwen3.5-0.8B`

## Directory Layout

- `configs/`: CPU / NPU / TPU configs
- `scripts/`: runnable entrypoints and accelerator wrappers
- `src/`: implementation code
- `data/`: copied raw PAN files and processed splits
- `outputs/`: hidden states, layer analysis, safe subspaces, semantic-transfer artifacts, training outputs, sanity-eval outputs, and final tables
- `external/safety-residual-space/data/`: upstream source data snapshot used to reconstruct PAN-style splits
- proposal docs at workspace root: `方案详述.md` and `实验代码流程.md`

## Stage A

1. Prepare datasets:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\00_prepare_data.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

2. Extract hidden states:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\01_extract_hidden_states.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split alignment --model teacher
```

Supported splits now include `alignment`, `analysis_val`, `pan_test`, `sanity_test`, and `pan_train`.

## Stage B

Analyze teacher layers and select the proposal's top-3 key layers:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\02_analyze_teacher_layers.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

## Stage C

Build teacher safe subspaces for the selected key layers:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\03_build_teacher_safe_subspace.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

## Stage D

Pair teacher key layers to student layers by relative depth:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\04_pair_layers.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

## Stage E

Build semantic bases, project teacher safe components, decompose top-256 semantics, and recompose student targets:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\05_build_semantic_bases.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\06_project_teacher_safe_component.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split alignment
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\06_project_teacher_safe_component.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split analysis_val
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\06_project_teacher_safe_component.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split sanity_test
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\07_decompose_teacher_semantics.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split alignment
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\07_decompose_teacher_semantics.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split analysis_val
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\07_decompose_teacher_semantics.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split sanity_test
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\08_recompose_student_targets.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split alignment
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\08_recompose_student_targets.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split analysis_val
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\08_recompose_student_targets.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml --split sanity_test
```

## Stage F

Train the student paired-layer LoRA with language-model supervision plus layer alignment:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\09_train_student_semalign.py --config D:\safety\configs\qwen35_08b_phaseF_cpu.yaml
```

## Stage G

Run the baseline-vs-semalign sanity check on the held-out sanity split:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\10_sanity_eval.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

## Stage H

Assemble the final summary tables:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\11_make_tables.py --config D:\safety\configs\qwen35_08b_phase1_cpu.yaml
```

This stage writes:

- `tables/table_key_layers.csv`
- `tables/table_layer_pairs.csv`
- `tables/table_training_val.csv`
- `tables/table_sanity_comparison.csv`
- `tables/phase1_overview.json`

## Accelerator Entry Points

The main scripts are now backend-aware through `runtime_backend` and `runtime_device` in config.

For NPU:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\09_train_student_semalign_npu.py
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\10_sanity_eval_npu.py
```

For TPU:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\09_train_student_semalign_tpu.py
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\10_sanity_eval_tpu.py
```

For the earlier pipeline stages on NPU / TPU, run the same generic scripts with:

- `configs/qwen35_08b_phase1_npu.yaml`
- `configs/qwen35_08b_phase1_tpu.yaml`

Training configs are:

- `configs/qwen35_08b_phaseF_npu.yaml`
- `configs/qwen35_08b_phaseF_tpu.yaml`

## Exploratory Utilities

The earlier exploratory path remains available:

- `scripts/02_fit_safety_subspace.py`
- `scripts/03_run_safety_intervention.py`

## Baseline Suite

The workspace now also includes a reusable baseline stack for:

- `no-sft`: direct evaluation of the base model on PAN, MMLU, GSM8K, HumanEval, and MBPP
- `sft`: PAN-only supervised fine-tuning with LoRA, then the same evaluation suite
- `distill`: PAN-only teacher-student distillation from Qwen3.5-9B to Qwen3.5-0.8B, then the same evaluation suite

### New Entry Points

- `scripts/12_eval_baseline_suite.py`
- `scripts/13_train_pan_sft.py`
- `scripts/14_train_pan_distill.py`
- `scripts/15_run_oneclick.py`

### New Configs

- `configs/baseline_eval_qwen35_08b.yaml`
- `configs/baseline_eval_qwen35_9b.yaml`
- `configs/baseline_sft_qwen35_08b.yaml`
- `configs/baseline_sft_qwen35_9b.yaml`
- `configs/baseline_distill_qwen35_9b_to_08b.yaml`

### Dataset Notes

The reconstructed PAN training data is a safety-alignment dataset, not a general instruction corpus:

- `data/processed/pan_train_set.jsonl` contains 2600 examples
- the train reconstruction is balanced: 1300 harmful + 1300 harmless
- `alignment_set.jsonl` and `analysis_val_set.jsonl` split that PAN train reconstruction into train / validation
- harmful examples target refusal-style outputs, while harmless examples target normal helpful outputs

This means PAN is suitable for safety SFT and safety distillation, but it is not a task-specific training set for:

- MMLU
- GSM8K
- HumanEval
- MBPP

So those four benchmarks should be interpreted as transfer/generalization evaluation after safety tuning, not as in-domain training metrics.

### Benchmark Paths

The baseline eval configs expect the following benchmark roots:

- `data/benchmarks/mmlu`
- `data/benchmarks/gsm8k`
- `data/benchmarks/humaneval`
- `data/benchmarks/mbpp`

If one of those paths is missing, `scripts/12_eval_baseline_suite.py` writes a placeholder result for that task instead of crashing the entire run.

### Model Paths

The new configs are prepared for:

- `models/Qwen3.5-0.8B`
- `models/Qwen3.5-9B`

Update the baseline configs if your local `0.8B` / `9B` directories use different names.

### Example Commands

One-click launcher:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py nosft --device npu --device-id 0 --model 0.8b
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py sft --device tpu --device-id 1 --model 9b
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py distill --device npu --device-id 2
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py random --device tpu --device-id 0
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py full --device tpu --device-id 0
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\15_run_oneclick.py smoke --device npu --device-id 0
```

### Device Control Notes

- `--device npu --device-id 3` exposes physical NPU `3` via `ASCEND_RT_VISIBLE_DEVICES=3`, and the process then runs on logical `npu:0`
- `--device tpu --device-id 2` binds the run to `xla:2`
- `--num-devices` is exposed on the launcher, but the current code path is still single-process single-device, so only `--num-devices 1` is supported right now
- the launcher writes a temporary runtime-override config before execution, so you do not need to hand-edit the original NPU / TPU YAML files just to change card IDs

Direct evaluation without SFT:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\12_eval_baseline_suite.py --config D:\safety\configs\baseline_eval_qwen35_08b.yaml
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\12_eval_baseline_suite.py --config D:\safety\configs\baseline_eval_qwen35_9b.yaml
```

PAN SFT:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\13_train_pan_sft.py --config D:\safety\configs\baseline_sft_qwen35_08b.yaml
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\13_train_pan_sft.py --config D:\safety\configs\baseline_sft_qwen35_9b.yaml
```

Evaluate an SFT checkpoint:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\12_eval_baseline_suite.py `
  --config D:\safety\configs\baseline_eval_qwen35_08b.yaml `
  --adapter-manifest D:\safety\outputs\baselines\sft_qwen35_08b\manifest.json `
  --adapter-checkpoint D:\safety\outputs\baselines\sft_qwen35_08b\checkpoints\epoch_001.pt `
  --output-dir D:\safety\outputs\baselines\sft_qwen35_08b\eval_suite
```

PAN distillation (9B -> 0.8B):

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\14_train_pan_distill.py --config D:\safety\configs\baseline_distill_qwen35_9b_to_08b.yaml
```

Evaluate a distillation checkpoint:

```powershell
D:\Anaconda3\envs\pytorch-cpu\python.exe D:\safety\scripts\12_eval_baseline_suite.py `
  --config D:\safety\configs\baseline_eval_qwen35_08b.yaml `
  --adapter-manifest D:\safety\outputs\baselines\distill_qwen35_9b_to_08b\manifest.json `
  --adapter-checkpoint D:\safety\outputs\baselines\distill_qwen35_9b_to_08b\checkpoints\epoch_001.pt `
  --output-dir D:\safety\outputs\baselines\distill_qwen35_9b_to_08b\eval_suite
```
