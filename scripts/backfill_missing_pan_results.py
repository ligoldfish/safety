#!/usr/bin/env python3
"""Regenerate the three known formal ``pan_results.json`` gaps without training.

The targets are intentionally explicit: Qwen3.5-0.8B C5 NoSFT plus the
Qwen3-4B WildJailbreak SFT epoch-2/3 checkpoints. Existing usable generation
files are skipped. The two large full-finetune checkpoints are evaluated
sequentially so each process releases CPU/NPU memory before the next load.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_KEYS = (
    "qwen35_c5_nosft",
    "qwen3_4b_wjb_sft_epoch_002",
    "qwen3_4b_wjb_sft_epoch_003",
)


@dataclass(frozen=True)
class PanBackfillTarget:
    key: str
    eval_config: Path
    output_dir: Path
    physical_device: int
    batch_size: int
    manifest_path: Path | None = None
    checkpoint_path: Path | None = None

    @property
    def pan_results_path(self) -> Path:
        return self.output_dir / "pan_results.json"

    @property
    def effective_config_path(self) -> Path:
        return self.output_dir / "backfill_eval_config.yaml"


def build_targets(
    project_root: Path,
    *,
    nosft_device: int,
    wjb_device: int,
    nosft_batch_size: int,
    wjb_batch_size: int,
) -> list[PanBackfillTarget]:
    root = project_root.resolve()
    wjb_training = (
        root / "outputs" / "baselines" / "sft_qwen3_4b_wildjailbreak_npu"
    )
    targets = [
        PanBackfillTarget(
            key="qwen35_c5_nosft",
            eval_config=root / "configs" / "baseline_eval_qwen35_08b_c5_npu.yaml",
            output_dir=root / "outputs" / "baselines" / "eval_c5_npu",
            physical_device=nosft_device,
            batch_size=nosft_batch_size,
        )
    ]
    for epoch in ("epoch_002", "epoch_003"):
        targets.append(
            PanBackfillTarget(
                key=f"qwen3_4b_wjb_sft_{epoch}",
                eval_config=(
                    root
                    / "configs"
                    / "baseline_eval_qwen3_4b_wildjailbreak_npu.yaml"
                ),
                output_dir=wjb_training / "eval_suite" / epoch,
                physical_device=wjb_device,
                batch_size=wjb_batch_size,
                manifest_path=wjb_training / "manifest.json",
                checkpoint_path=wjb_training / "checkpoints" / f"{epoch}.pt",
            )
        )
    return targets


def pan_results_is_usable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and bool(payload.get("generations"))


def _absolute_config_path(value: object, base_dir: Path) -> str:
    text = str(value or "")
    if not text or "://" in text:
        return text
    path = Path(text)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def effective_config_payload(target: PanBackfillTarget) -> dict[str, Any]:
    source = target.eval_config.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must contain a mapping: {source}")
    base_dir = source.parent

    model = raw.setdefault("model", {})
    if not isinstance(model, dict) or not model.get("path"):
        raise ValueError(f"Eval config has no model.path: {source}")
    model["path"] = _absolute_config_path(model["path"], base_dir)
    model["runtime_backend"] = "npu"
    # ASCEND_RT_VISIBLE_DEVICES selects the physical die; the process sees it as npu:0.
    model["runtime_device"] = "npu:0"

    datasets = raw.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Eval config datasets must be a mapping: {source}")
    for dataset in datasets.values():
        if isinstance(dataset, dict) and dataset.get("path"):
            dataset["path"] = _absolute_config_path(dataset["path"], base_dir)

    runtime = raw.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError(f"Eval config runtime must be a mapping: {source}")
    runtime["batch_size"] = target.batch_size

    adapter = raw.setdefault("adapter", {})
    if not isinstance(adapter, dict):
        raise ValueError(f"Eval config adapter must be a mapping: {source}")
    adapter["manifest_path"] = (
        str(target.manifest_path.resolve()) if target.manifest_path else ""
    )
    adapter["checkpoint_path"] = (
        str(target.checkpoint_path.resolve()) if target.checkpoint_path else ""
    )

    output = raw.setdefault("output", {})
    if not isinstance(output, dict):
        raise ValueError(f"Eval config output must be a mapping: {source}")
    output["output_root"] = str(target.output_dir.resolve())
    raw["backfill_provenance"] = {
        "source_config": str(source),
        "target": target.key,
        "physical_device": target.physical_device,
        "logical_device": "npu:0",
        "batch_size": target.batch_size,
    }
    return raw


def write_effective_config(target: PanBackfillTarget, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            effective_config_payload(target),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def build_eval_command(
    python_executable: str,
    project_root: Path,
    target: PanBackfillTarget,
    config_path: Path,
) -> list[str]:
    command = [
        python_executable,
        str(project_root / "scripts" / "12_eval_baseline_suite.py"),
        "--config",
        str(config_path),
    ]
    if target.manifest_path and target.checkpoint_path:
        command.extend([
            "--adapter-manifest",
            str(target.manifest_path),
            "--adapter-checkpoint",
            str(target.checkpoint_path),
        ])
    command.extend(["--output-dir", str(target.output_dir)])
    return command


def build_merge_command(
    python_executable: str,
    project_root: Path,
    target: PanBackfillTarget,
) -> list[str]:
    return [
        python_executable,
        str(project_root / "scripts" / "18_merge_opencompass_summary.py"),
        "--pan-summary",
        str(target.output_dir / "summary.json"),
        "--output",
        str(target.output_dir / "final_summary.json"),
    ]


def _required_inputs(target: PanBackfillTarget) -> list[Path]:
    required = [target.eval_config]
    if target.manifest_path:
        required.append(target.manifest_path)
    if target.checkpoint_path:
        required.append(target.checkpoint_path)
    return required


def _validate_effective_inputs(target: PanBackfillTarget) -> None:
    payload = effective_config_payload(target)
    paths = [Path(payload["model"]["path"]), Path(payload["datasets"]["pan"]["path"])]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{target.key} is missing model/test inputs: "
            + ", ".join(str(path) for path in missing)
        )


def _run(command: Sequence[str], *, project_root: Path, physical_device: int) -> None:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:512")
    print(f"[pan-backfill] run: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=project_root, env=env, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--targets", nargs="+", choices=TARGET_KEYS, default=list(TARGET_KEYS))
    parser.add_argument("--nosft-device", type=int, default=0)
    parser.add_argument("--wjb-device", type=int, default=1)
    parser.add_argument("--nosft-batch-size", type=int, default=16)
    parser.add_argument("--wjb-batch-size", type=int, default=4)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.nosft_device < 0 or args.wjb_device < 0:
        parser.error("device ids must be nonnegative")
    if args.nosft_batch_size <= 0 or args.wjb_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    all_targets = build_targets(
        project_root,
        nosft_device=args.nosft_device,
        wjb_device=args.wjb_device,
        nosft_batch_size=args.nosft_batch_size,
        wjb_batch_size=args.wjb_batch_size,
    )
    selected = [target for target in all_targets if target.key in set(args.targets)]
    pending = [
        target
        for target in selected
        if args.force or not pan_results_is_usable(target.pan_results_path)
    ]
    skipped = len(selected) - len(pending)

    missing = [
        path
        for target in pending
        for path in _required_inputs(target)
        if not path.is_file()
    ]
    if missing:
        for path in missing:
            print(f"[pan-backfill][ERR] required file missing: {path}", file=sys.stderr)
        return 2
    for target in pending:
        _validate_effective_inputs(target)

    print(
        f"[pan-backfill] selected={len(selected)} pending={len(pending)} "
        f"skipped_usable={skipped} dry_run={int(args.dry_run)}",
        flush=True,
    )
    for target in pending:
        config_path = target.effective_config_path
        eval_command = build_eval_command(
            args.python_executable, project_root, target, config_path
        )
        merge_command = build_merge_command(args.python_executable, project_root, target)
        print(
            f"[pan-backfill] target={target.key} physical_npu={target.physical_device} "
            f"batch_size={target.batch_size} output={target.output_dir}",
            flush=True,
        )
        if args.dry_run:
            print(f"[pan-backfill] would write config: {config_path}")
            print(f"[pan-backfill] would run: {shlex.join(eval_command)}")
            print(f"[pan-backfill] would run: {shlex.join(merge_command)}")
            continue

        write_effective_config(target, config_path)
        _run(eval_command, project_root=project_root, physical_device=target.physical_device)
        if not pan_results_is_usable(target.pan_results_path):
            raise RuntimeError(
                f"Eval completed without usable generations: {target.pan_results_path}"
            )
        _run(merge_command, project_root=project_root, physical_device=target.physical_device)
        payload = json.loads(target.pan_results_path.read_text(encoding="utf-8"))
        print(
            f"[pan-backfill] complete target={target.key} "
            f"generations={len(payload['generations'])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
