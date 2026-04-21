from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ("mmlu", "gsm8k", "humaneval", "mbpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenCompass (https://github.com/open-compass/opencompass) on a merged "
            "HuggingFace checkpoint to evaluate general capabilities (MMLU / GSM8K / HumanEval / MBPP). "
            "Safety-specific PAN evaluation remains in scripts/12_eval_baseline_suite.py."
        )
    )
    parser.add_argument(
        "--merged-model-dir",
        required=True,
        help="Path to the merged HuggingFace checkpoint produced by 16_merge_lora_for_opencompass.py.",
    )
    parser.add_argument(
        "--opencompass-dir",
        required=True,
        help="Path to a cloned OpenCompass repository (https://github.com/open-compass/opencompass).",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory to write OpenCompass outputs into.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to run (forwarded to opencompass --datasets).",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-out-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--hf-type",
        default="chat",
        choices=("chat", "base"),
        help="Whether to treat the model as a chat or base HF model inside OpenCompass.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded verbatim to opencompass run.py after -- .",
    )
    return parser.parse_args()


def _resolve_opencompass_entry(opencompass_dir: Path) -> Path:
    run_py = opencompass_dir / "run.py"
    if run_py.exists():
        return run_py
    tools_run = opencompass_dir / "tools" / "run.py"
    if tools_run.exists():
        return tools_run
    raise FileNotFoundError(
        f"Could not find OpenCompass entrypoint under {opencompass_dir}. "
        "Expected run.py or tools/run.py."
    )


def main() -> None:
    args = parse_args()
    merged_model_dir = Path(args.merged_model_dir).resolve()
    if not merged_model_dir.exists():
        raise FileNotFoundError(f"Merged model dir not found: {merged_model_dir}")
    opencompass_dir = Path(args.opencompass_dir).resolve()
    run_py = _resolve_opencompass_entry(opencompass_dir)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(run_py),
        "--hf-type",
        args.hf_type,
        "--hf-path",
        str(merged_model_dir),
        "--tokenizer-path",
        str(merged_model_dir),
        "--model-kwargs",
        "device_map='auto' trust_remote_code=True",
        "--tokenizer-kwargs",
        "trust_remote_code=True use_fast=True",
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-out-len",
        str(args.max_out_len),
        "--batch-size",
        str(args.batch_size),
        "--hf-num-gpus",
        str(args.num_gpus),
        "--work-dir",
        str(work_dir),
        "--datasets",
        *args.datasets,
    ]
    extra = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)

    invocation = {
        "merged_model_dir": str(merged_model_dir),
        "opencompass_dir": str(opencompass_dir),
        "entry": str(run_py),
        "work_dir": str(work_dir),
        "datasets": list(args.datasets),
        "command": cmd,
    }
    (work_dir / "opencompass_invocation.json").write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Running OpenCompass:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(opencompass_dir))


if __name__ == "__main__":
    main()
