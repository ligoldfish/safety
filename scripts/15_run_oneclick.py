from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import load_distill_config, load_sft_config
from src.utils.config import load_phase1_config, load_phasef_config


BASELINE_EVAL_CONFIGS = {
    ("npu", "1b"): "configs/baseline_eval_qwen35_1b_npu.yaml",
    ("tpu", "1b"): "configs/baseline_eval_qwen35_1b_tpu.yaml",
    ("npu", "9b"): "configs/baseline_eval_qwen35_9b_npu.yaml",
    ("tpu", "9b"): "configs/baseline_eval_qwen35_9b_tpu.yaml",
}

BASELINE_SFT_CONFIGS = {
    ("npu", "1b"): "configs/baseline_sft_qwen35_1b_npu.yaml",
    ("tpu", "1b"): "configs/baseline_sft_qwen35_1b_tpu.yaml",
    ("npu", "9b"): "configs/baseline_sft_qwen35_9b_npu.yaml",
    ("tpu", "9b"): "configs/baseline_sft_qwen35_9b_tpu.yaml",
}

BASELINE_DISTILL_CONFIGS = {
    "npu": "configs/baseline_distill_qwen35_9b_to_1b_npu.yaml",
    "tpu": "configs/baseline_distill_qwen35_9b_to_1b_tpu.yaml",
}

FULL_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_npu.yaml",
    },
    "tpu": {
        "phase1": "configs/qwen35_08b_phase1_tpu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_tpu.yaml",
    },
}

SMOKE_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_08b_phase1_smoke_npu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_smoke_npu.yaml",
    },
    "tpu": {
        "phase1": "configs/qwen35_08b_phase1_smoke_tpu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_smoke_tpu.yaml",
    },
}

RANDOM_PIPELINE_CONFIGS = {
    "npu": {
        "phase1": "configs/qwen35_08b_phase1_npu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_npu_random.yaml",
    },
    "tpu": {
        "phase1": "configs/qwen35_08b_phase1_tpu.yaml",
        "phasef": "configs/qwen35_08b_phaseF_tpu_random.yaml",
    },
}

PIPELINE_SPLITS = ["alignment", "analysis_val", "pan_test", "sanity_test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click launcher for baseline and full-stage experiments on NPU/TPU."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument(
            "--device",
            choices=["npu", "tpu"],
            required=True,
            help="Accelerator backend to use.",
        )
        target_parser.add_argument(
            "--device-id",
            type=int,
            default=0,
            help="Primary accelerator ordinal. NPU maps to npu:<id>; TPU maps to xla:<id>.",
        )
        target_parser.add_argument(
            "--num-devices",
            type=int,
            default=1,
            help="Requested accelerator count. The current code path is single-process single-device, so only 1 is supported.",
        )
        target_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the commands without executing them.",
        )

    nosft_parser = subparsers.add_parser("nosft", help="Run no-SFT benchmark evaluation.")
    nosft_parser.add_argument("--model", choices=["1b", "9b"], required=True)
    add_common_flags(nosft_parser)

    sft_parser = subparsers.add_parser("sft", help="Run PAN SFT and then benchmark evaluation.")
    sft_parser.add_argument("--model", choices=["1b", "9b"], required=True)
    add_common_flags(sft_parser)

    distill_parser = subparsers.add_parser("distill", help="Run PAN distillation and then benchmark evaluation.")
    add_common_flags(distill_parser)

    random_parser = subparsers.add_parser(
        "random",
        help="Run the random-vector baseline on the original 00->11 pipeline.",
    )
    add_common_flags(random_parser)

    full_parser = subparsers.add_parser("full", help="Run the original 00->11 full-stage pipeline.")
    add_common_flags(full_parser)

    smoke_parser = subparsers.add_parser("smoke", help="Run a smoke-sized 00->11 pipeline.")
    add_common_flags(smoke_parser)
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _run_script(
    script_name: str,
    args: Sequence[str],
    *,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    rendered = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(rendered)
    if env_overrides:
        print(
            "env:",
            " ".join(f"{key}={value}" for key, value in sorted(env_overrides.items())),
        )
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _latest_checkpoint_path(training_dir: Path) -> Path:
    checkpoint_dir = training_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_dir}")
    return candidates[-1]


def _validate_device_request(num_devices: int) -> None:
    if num_devices != 1:
        raise ValueError(
            "The current launcher only supports single-process single-device execution. "
            "Use --num-devices 1. Multi-device NPU/TPU parallelism would require a distributed training path."
        )


def _runtime_device_value(device: str, device_id: int) -> str:
    if device == "npu":
        return "npu:0"
    if device == "tpu":
        return f"xla:{device_id}"
    raise ValueError(f"Unsupported device: {device}")


def _build_env_overrides(device: str, device_id: int) -> dict[str, str]:
    if device == "npu":
        return {"ASCEND_RT_VISIBLE_DEVICES": str(device_id)}
    return {}


def _override_model_runtime(model_payload: dict[str, Any], device: str, device_id: int) -> None:
    model_payload["runtime_backend"] = device
    model_payload["runtime_device"] = _runtime_device_value(device, device_id)


def _make_runtime_override_config(config_path: Path, *, device: str, device_id: int) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    if isinstance(raw.get("model"), dict):
        _override_model_runtime(raw["model"], device, device_id)
    if isinstance(raw.get("teacher"), dict):
        _override_model_runtime(raw["teacher"], device, device_id)
    if isinstance(raw.get("student"), dict):
        _override_model_runtime(raw["student"], device, device_id)
    if isinstance(raw.get("models"), dict):
        for model_payload in raw["models"].values():
            if isinstance(model_payload, dict):
                _override_model_runtime(model_payload, device, device_id)

    override_dir = config_path.parent
    override_path = override_dir / f"{config_path.stem}_launcher_{device}_{device_id}_{uuid.uuid4().hex[:8]}.yaml"
    override_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return override_path


def _run_phase1_precompute(
    phase1_config: Path,
    *,
    smoke: bool,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    _run_script("00_prepare_data.py", ["--config", str(phase1_config)], dry_run=dry_run, env_overrides=env_overrides)

    for split in PIPELINE_SPLITS:
        split_args = [
            "--config",
            str(phase1_config),
            "--split",
            split,
            "--model",
            "teacher",
        ]
        _run_script("01_extract_hidden_states.py", split_args, dry_run=dry_run, env_overrides=env_overrides)

    analyze_args = ["--config", str(phase1_config)]
    subspace_args = ["--config", str(phase1_config)]
    semantic_args_suffix: list[str] = ["--config", str(phase1_config)]
    if smoke:
        analyze_args += [
            "--top-k",
            "2",
            "--probe-max-iter",
            "20",
            "--train-max-samples-per-label",
            "16",
            "--val-max-samples-per-label",
            "8",
        ]
        subspace_args += ["--rank", "4"]
        semantic_args_suffix += ["--top-k", "64", "--vocab-chunk-size", "2048"]

    _run_script("02_analyze_teacher_layers.py", analyze_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("03_build_teacher_safe_subspace.py", subspace_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("04_pair_layers.py", ["--config", str(phase1_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script("05_build_semantic_bases.py", ["--config", str(phase1_config)], dry_run=dry_run, env_overrides=env_overrides)

    for split in PIPELINE_SPLITS:
        _run_script(
            "06_project_teacher_safe_component.py",
            ["--config", str(phase1_config), "--split", split],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )
        _run_script(
            "07_decompose_teacher_semantics.py",
            [*semantic_args_suffix, "--split", split],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )
        _run_script(
            "08_recompose_student_targets.py",
            ["--config", str(phase1_config), "--split", split],
            dry_run=dry_run,
            env_overrides=env_overrides,
        )


def _run_baseline_nosft(device: str, model_size: str, *, device_id: int, num_devices: int, dry_run: bool) -> None:
    _validate_device_request(num_devices)
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    env_overrides = _build_env_overrides(device, device_id)
    _run_script(
        "12_eval_baseline_suite.py",
        ["--config", str(eval_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_baseline_sft(device: str, model_size: str, *, device_id: int, num_devices: int, dry_run: bool) -> None:
    _validate_device_request(num_devices)
    train_config = _make_runtime_override_config(
        _resolve(BASELINE_SFT_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    cfg = load_sft_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    _run_script(
        "13_train_pan_sft.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    if dry_run:
        checkpoint_path = Path(cfg.output.output_root) / "checkpoints" / f"epoch_{cfg.optim.epochs:03d}.pt"
    else:
        checkpoint_path = _latest_checkpoint_path(Path(cfg.output.output_root))
    _run_script(
        "12_eval_baseline_suite.py",
        [
            "--config",
            str(eval_config),
            "--adapter-manifest",
            str(Path(cfg.output.output_root) / "manifest.json"),
            "--adapter-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(Path(cfg.output.output_root) / "eval_suite"),
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_baseline_distill(device: str, *, device_id: int, num_devices: int, dry_run: bool) -> None:
    _validate_device_request(num_devices)
    train_config = _make_runtime_override_config(
        _resolve(BASELINE_DISTILL_CONFIGS[device]),
        device=device,
        device_id=device_id,
    )
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, "1b")]),
        device=device,
        device_id=device_id,
    )
    cfg = load_distill_config(train_config)
    env_overrides = _build_env_overrides(device, device_id)

    _run_script(
        "14_train_pan_distill.py",
        ["--config", str(train_config)],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    if dry_run:
        checkpoint_path = Path(cfg.output.output_root) / "checkpoints" / f"epoch_{cfg.optim.epochs:03d}.pt"
    else:
        checkpoint_path = _latest_checkpoint_path(Path(cfg.output.output_root))
    _run_script(
        "12_eval_baseline_suite.py",
        [
            "--config",
            str(eval_config),
            "--adapter-manifest",
            str(Path(cfg.output.output_root) / "manifest.json"),
            "--adapter-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(Path(cfg.output.output_root) / "eval_suite"),
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_adapter_eval(
    *,
    device: str,
    model_size: str,
    training_output_root: Path,
    device_id: int,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    eval_config = _make_runtime_override_config(
        _resolve(BASELINE_EVAL_CONFIGS[(device, model_size)]),
        device=device,
        device_id=device_id,
    )
    if dry_run:
        checkpoint_path = training_output_root / "checkpoints" / "epoch_999.pt"
    else:
        checkpoint_path = _latest_checkpoint_path(training_output_root)
    _run_script(
        "12_eval_baseline_suite.py",
        [
            "--config",
            str(eval_config),
            "--adapter-manifest",
            str(training_output_root / "manifest.json"),
            "--adapter-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(training_output_root / "eval_suite"),
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_random_baseline(device: str, *, device_id: int, num_devices: int, dry_run: bool) -> None:
    _validate_device_request(num_devices)
    phase1_config = _make_runtime_override_config(
        _resolve(RANDOM_PIPELINE_CONFIGS[device]["phase1"]),
        device=device,
        device_id=device_id,
    )
    phasef_config = _make_runtime_override_config(
        _resolve(RANDOM_PIPELINE_CONFIGS[device]["phasef"]),
        device=device,
        device_id=device_id,
    )
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(phase1_config, smoke=False, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script(
        "10_sanity_eval.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir",
            str(Path(phasef_cfg.output.output_root)),
            "--output-dir-name",
            "sanity_eval_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_script(
        "11_make_tables.py",
        [
            "--config",
            str(phase1_config),
            "--training-dir-name",
            Path(phasef_cfg.output.output_root).name,
            "--sanity-dir-name",
            "sanity_eval_random_same_norm",
            "--tables-dir-name",
            "tables_random_same_norm",
        ],
        dry_run=dry_run,
        env_overrides=env_overrides,
    )
    _run_adapter_eval(
        device=device,
        model_size="1b",
        training_output_root=Path(phasef_cfg.output.output_root),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )


def _run_full_pipeline(device: str, *, device_id: int, num_devices: int, smoke: bool, dry_run: bool) -> None:
    _validate_device_request(num_devices)
    config_map = SMOKE_PIPELINE_CONFIGS if smoke else FULL_PIPELINE_CONFIGS
    phase1_config = _make_runtime_override_config(
        _resolve(config_map[device]["phase1"]),
        device=device,
        device_id=device_id,
    )
    phasef_config = _make_runtime_override_config(
        _resolve(config_map[device]["phasef"]),
        device=device,
        device_id=device_id,
    )

    phase1_cfg = load_phase1_config(phase1_config)
    phasef_cfg = load_phasef_config(phasef_config)

    env_overrides = _build_env_overrides(device, device_id)
    _run_phase1_precompute(phase1_config, smoke=smoke, dry_run=dry_run, env_overrides=env_overrides)

    sanity_args = ["--config", str(phase1_config)]
    if smoke:
        sanity_args += ["--max-samples-per-label", "8", "--max-new-tokens", "32"]

    _run_script("09_train_student_semalign.py", ["--config", str(phasef_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_script("10_sanity_eval.py", sanity_args, dry_run=dry_run, env_overrides=env_overrides)
    _run_script("11_make_tables.py", ["--config", str(phase1_config)], dry_run=dry_run, env_overrides=env_overrides)
    _run_adapter_eval(
        device=device,
        model_size="1b",
        training_output_root=Path(phasef_cfg.output.output_root),
        device_id=device_id,
        dry_run=dry_run,
        env_overrides=env_overrides,
    )

    if not dry_run:
        summary = {
            "device": device,
            "device_id": device_id,
            "num_devices": num_devices,
            "smoke": smoke,
            "phase1_output_root": phase1_cfg.extraction.output_root,
            "phasef_output_root": phasef_cfg.output.output_root,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "nosft":
        _run_baseline_nosft(
            args.device,
            args.model,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
        )
        return
    if args.command == "sft":
        _run_baseline_sft(
            args.device,
            args.model,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
        )
        return
    if args.command == "distill":
        _run_baseline_distill(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
        )
        return
    if args.command == "random":
        _run_random_baseline(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            dry_run=args.dry_run,
        )
        return
    if args.command == "full":
        _run_full_pipeline(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            smoke=False,
            dry_run=args.dry_run,
        )
        return
    if args.command == "smoke":
        _run_full_pipeline(
            args.device,
            device_id=args.device_id,
            num_devices=args.num_devices,
            smoke=True,
            dry_run=args.dry_run,
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
