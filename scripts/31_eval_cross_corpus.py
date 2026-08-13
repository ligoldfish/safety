from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an immutable checkpoint registry on one shared safety suite."
    )
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("checkpoints"):
        raise ValueError("cross-corpus manifest must contain checkpoints")
    return payload


def _evaluation_config(entry: dict, manifest: dict, output_dir: Path) -> dict:
    template = PROJECT_ROOT / "configs" / "baseline_eval_qwen35_08b_npu.yaml"
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    for task in payload.get("datasets", {}).values():
        if isinstance(task, dict):
            task["enabled"] = False
    pan = payload["datasets"]["pan"]
    pan.update(
        path=str(manifest["test_jsonl"]),
        enabled=True,
        placeholder_ok=False,
        shuffle=False,
        max_samples=0,
    )
    payload["model"].update(
        path=str(entry.get("model_path") or entry.get("base_model_path")),
        runtime_backend=str(manifest["runtime_backend"]),
        runtime_device=str(manifest["runtime_device"]),
        local_files_only=True,
    )
    payload["output"]["output_root"] = str(output_dir)
    return payload


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    for entry in manifest["checkpoints"]:
        output_dir = Path(entry["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "eval_config.yaml"
        config_path.write_text(
            yaml.safe_dump(_evaluation_config(entry, manifest, output_dir), sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "12_eval_baseline_suite.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if entry["kind"] == "adapter":
            command.extend(
                [
                    "--adapter-manifest",
                    str(entry["manifest_path"]),
                    "--adapter-checkpoint",
                    str(entry["checkpoint_path"]),
                ]
            )
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "22_judge_generations.py"),
                "--pan-results",
                str(output_dir / "pan_results.json"),
                "--test-jsonl",
                str(manifest["test_jsonl"]),
                "--judge-model",
                str(manifest["wildguard_model"]),
                "--runtime-backend",
                str(manifest["runtime_backend"]),
                "--runtime-device",
                str(manifest["runtime_device"]),
                "--out",
                str(output_dir / "judge_results.json"),
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
