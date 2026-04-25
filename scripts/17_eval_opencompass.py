from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ("mmlu_gen", "gsm8k_gen", "humaneval_gen", "mbpp_gen")
SUPPORTED_DATASETS = frozenset(DEFAULT_DATASETS)


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
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=-1,
        help=(
            "Number of accelerators (CUDA GPUs or Ascend NPUs) exposed per task. "
            "0 selects CPU-only. When left at -1, the launcher auto-selects 1 on "
            "hosts with a visible accelerator (CUDA or ASCEND) and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "npu", "cpu"),
        help=(
            "Target accelerator backend. 'auto' detects ASCEND_RT_VISIBLE_DEVICES / "
            "CUDA_VISIBLE_DEVICES from the environment."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        default="torch.float16",
        help=(
            "torch dtype string accepted by OpenCompass (_set_model_kwargs_torch_dtype): "
            "'torch.float16' (default, matches training dtype), 'torch.bfloat16', "
            "'torch.float', or 'auto'."
        ),
    )
    parser.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2", "auto"),
        help=(
            "Attention implementation forwarded to from_pretrained via model-kwargs. "
            "'eager' is the most portable and lossless on NPU; 'auto' lets transformers pick."
        ),
    )
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


def _detect_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    has_ascend = bool(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip())
    has_cuda = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    if has_ascend and not has_cuda:
        return "npu"
    if has_cuda:
        return "cuda"
    return "cpu"


def _default_num_gpus(requested: int, backend: str) -> int:
    if requested >= 0:
        return requested
    return 0 if backend == "cpu" else 1


def _build_model_kwargs_tokens(backend: str, attn_impl: str, torch_dtype: str) -> list[str]:
    # Each list entry becomes a separate argv token for mmengine DictAction (nargs='+').
    # Tokens are strings of form key=value; do NOT join with spaces (DictAction parses
    # each argv token individually via split('=', 1)).
    tokens = ["trust_remote_code=True"]
    if backend == "cpu":
        tokens.append("device_map=cpu")
    else:
        tokens.append("device_map=auto")
    if attn_impl != "auto":
        tokens.append(f"attn_implementation={attn_impl}")
    if torch_dtype:
        tokens.append(f"torch_dtype={torch_dtype}")
    return tokens


def _validate_requested_datasets(datasets: list[str]) -> None:
    unknown = sorted({dataset for dataset in datasets if dataset not in SUPPORTED_DATASETS})
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported OpenCompass dataset(s): {', '.join(unknown)}. "
            f"Supported datasets: {supported}."
        )


def _ensure_humaneval_dependency(datasets: list[str]) -> None:
    if "humaneval_gen" not in datasets:
        return
    if importlib.util.find_spec("human_eval") is not None:
        return
    raise ModuleNotFoundError(
        "Dataset 'humaneval_gen' requires the 'human_eval' package, but it is not installed. "
        "Install it in the current environment before running OpenCompass with HumanEval."
    )


def main() -> None:
    args = parse_args()
    requested_datasets = [str(dataset) for dataset in args.datasets]
    _validate_requested_datasets(requested_datasets)
    _ensure_humaneval_dependency(requested_datasets)
    merged_model_dir = Path(args.merged_model_dir).resolve()
    if not merged_model_dir.exists():
        raise FileNotFoundError(f"Merged model dir not found: {merged_model_dir}")
    opencompass_dir = Path(args.opencompass_dir).resolve()
    run_py = _resolve_opencompass_entry(opencompass_dir)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    backend = _detect_backend(args.device)
    num_gpus = _default_num_gpus(args.num_gpus, backend)
    model_kwargs_tokens = _build_model_kwargs_tokens(backend, args.attn_impl, args.torch_dtype)
    # Qwen3.5 fast tokenizers on our current transformers/OpenCompass stack can
    # expose a backend object without ``batch_encode_plus``, while the bundled
    # HuggingFace wrapper still calls that legacy API. Default to the slow
    # tokenizer for compatibility across 0.8B/9B local runs.
    tokenizer_kwargs_tokens = ["trust_remote_code=True", "use_fast=False"]

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
        *model_kwargs_tokens,
        "--tokenizer-kwargs",
        *tokenizer_kwargs_tokens,
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-out-len",
        str(args.max_out_len),
        "--batch-size",
        str(args.batch_size),
        "--hf-num-gpus",
        str(num_gpus),
        "--work-dir",
        str(work_dir),
        "--datasets",
        *requested_datasets,
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
        "datasets": requested_datasets,
        "hf_num_gpus": num_gpus,
        "backend": backend,
        "attn_impl": args.attn_impl,
        "torch_dtype": args.torch_dtype,
        "model_kwargs_tokens": model_kwargs_tokens,
        "command": cmd,
    }
    (work_dir / "opencompass_invocation.json").write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Running OpenCompass (backend={backend}, num_gpus={num_gpus}):")
    print("  " + " ".join(cmd))
    env = os.environ.copy()
    if backend == "npu":
        # Ensure torch_npu is auto-loaded so mmengine.device.is_npu_available() sees it.
        # Do NOT set TORCH_DEVICE_BACKEND_AUTOLOAD=0 on NPU — it disables torch_npu.
        env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
        # Propagate the user-selected devices; opencompass LocalRunner reads this.
        if "ASCEND_RT_VISIBLE_DEVICES" not in env:
            env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    elif backend == "cpu":
        # Force opencompass onto the CPU path; block torch_npu auto-loading on Ascend hosts.
        env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(cmd, check=True, cwd=str(opencompass_dir), env=env)


if __name__ == "__main__":
    main()
