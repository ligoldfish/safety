# LLM Judge Result Collector Design

## Goal

Add a read-only reporting script that collects completed WildGuard LLM-judge
metrics from the formal cross-scale safety-transfer experiment matrix. The
script writes one uniform CSV per teacher-to-student model pair in the project
root and an audit JSON describing missing or fallback inputs.

The collector must never treat hyperparameter-search runs as formal results.

## Formal Experiment Matrix

The expected matrix is explicit rather than discovered recursively:

- Model pairs: `qwen35_9b_to_08b`, `llama31_8b_to_1b`,
  `qwen3_8b_to_06b`, `qwen3_8b_to_4b`, and `qwen3_4b_to_06b`.
- Datasets: `pan`, `safety_tuned_llamas`, `coconot`, `wildguardmix`,
  `wildjailbreak`, and `c5`.
- Methods: `ours`, `ours_sft1`, `sft`, `distill`, and `nosft`.

The model registry in `src/pairs.py` supplies teacher, student, and model-tag
metadata. The path resolver constructs only canonical formal paths:

- PAN `ours` and `ours_sft1` come from
  `outputs/<pair>_phase1_npu/training[_sft1]/eval_suite`.
- The other five datasets' `ours` and `ours_sft1` come from
  `outputs/safety_full_<dataset>_npu[_<pair>]/phase1/training[_sft1]/eval_suite`.
  The canonical Qwen3.5 pair uses the legacy safety-full path without a pair
  suffix.
- `sft` comes from `outputs/baselines/sft_<student-tag>[_<dataset>]_npu`.
- `distill` comes from
  `outputs/baselines/distill_<pair>[_<dataset>]_npu`.
- `nosft` comes from the student model's single-run PAN directory or its
  per-dataset evaluation directory.

PAN is represented by the absence of a dataset suffix in baseline directory
names. Explicit path construction excludes `outputs/sweep`, `sweep_runs`, and
live search cells such as `LW*`, `TK*`, `DEF`, `EP5`, and `SFT1` parameter
search directories without relying on a blacklist.

## Epoch Selection

For `ours`, `ours_sft1`, `sft`, and `distill` on every non-STL dataset, emit
both `epoch_002` and `epoch_003`.

For `safety_tuned_llamas`:

- `ours` and `ours_sft1` emit both `epoch_002` and `epoch_003`, because these
  methods are configured for three epochs.
- `sft` and `distill` emit only `epoch_006`, from their eight-epoch runs.
- `nosft` emits its one `single` result for every dataset.

No best-epoch selection, averaging, or metric-based filtering is performed.

## Input Precedence

For each expected row, the collector first reads the sibling
`judge_results.json`. If that file is unavailable or lacks the three core
metrics, it falls back to `summary.json` at `results.pan`, where the judge
backfill script merges the same scalars.

The three required metrics are:

- `llm_judge_asr`
- `llm_judge_over_refusal`
- `llm_judge_refusal_rate`

Optional scalar fields are copied when present: keyword agreement, Cohen's
kappa, parse rate, and harmful/harmless judged sample counts. Values remain in
their stored zero-to-one representation; the collector does not silently
convert them to percentages.

## Outputs

By default, running from `/root/safety` writes these files directly there:

- `llm_judge_results_qwen35_9b_to_08b.csv`
- `llm_judge_results_llama31_8b_to_1b.csv`
- `llm_judge_results_qwen3_8b_to_06b.csv`
- `llm_judge_results_qwen3_8b_to_4b.csv`
- `llm_judge_results_qwen3_4b_to_06b.csv`
- `llm_judge_results_audit.json`

Every CSV has a stable schema:

`model_pair`, `teacher_model`, `student_model`, `dataset`, `method`, `epoch`,
`status`, `source_kind`, the judge metric columns, and `source_path`.

Expected rows are never silently dropped. A missing, malformed, or incomplete
input produces a row with a non-`ok` status and blank metric cells. The audit
JSON records counts by status, all summary-fallback rows, and details for every
non-`ok` row. The command prints a concise summary and exits nonzero when
required results are missing or invalid; `--allow-missing` permits a zero exit
after writing the same audit.

CLI options allow overriding `--outputs-root` and `--output-dir` for local
tests or alternate deployments. The default roots are `outputs` and the
project root respectively.

## Error Handling and Safety

The script is read-only over `outputs`. It creates or replaces only the six
report files named above. JSON decoding errors, absent core metrics, and
unexpected metric types are reported per row. Source paths are emitted relative
to the project root when possible so remote reports remain portable.

The resolver detects path ambiguity instead of selecting by modification time.
Because every expected path is deterministic, repeated runs are idempotent and
cannot ingest search results.

## Verification

Focused tests build a temporary synthetic output tree and verify:

- all five method path patterns for canonical and extension pairs;
- both non-STL epochs are emitted;
- the STL method-specific epoch rule;
- `nosft` single-run mapping, including students shared by two model pairs;
- `judge_results.json` precedence and `summary.json` fallback;
- stable missing/malformed rows and audit counts;
- no sweep or search path can be selected;
- deterministic CSV ordering and schema.

No model loading, judge inference, remote command, or training is required for
collection or testing.
