from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_mapping(value: str, label: str) -> dict:
    payload = json.loads(value or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _json_list(value: str, label: str) -> list[str]:
    payload = json.loads(value or "[]")
    if not isinstance(payload, list) or not all(isinstance(item, str) and item for item in payload):
        raise ValueError(f"{label} must be a JSON list of non-empty strings")
    return payload


def _set_dotted(payload: dict, dotted: str, value) -> None:
    node = payload
    parts = str(dotted).split(".")
    for part in parts[:-1]:
        current = node.get(part)
        if not isinstance(current, dict):
            raise ValueError(f"cannot apply unknown config path: {dotted}")
        node = current
    node[parts[-1]] = value


def _absolutize_paths(value, anchor: Path) -> None:
    if isinstance(value, dict):
        for key, child in tuple(value.items()):
            if isinstance(child, str) and child and "://" not in child and (child.startswith("../") or child.startswith("./")):
                value[key] = str((anchor / child).resolve())
            else:
                _absolutize_paths(child, anchor)
    elif isinstance(value, list):
        for child in value:
            _absolutize_paths(child, anchor)


def _stage_configs(args, phase1_updates: dict, phasef_updates: dict) -> tuple[Path, Path]:
    config_dir = Path(args.output_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    phase1_source = PROJECT_ROOT / "configs" / f"{args.pair}_phase1_{args.device}.yaml"
    phasef_source = PROJECT_ROOT / "configs" / f"{args.pair}_phaseF_{args.device}.yaml"
    if not phase1_source.is_file() or not phasef_source.is_file():
        raise FileNotFoundError(f"pair/device configs are missing: {phase1_source}, {phasef_source}")
    phase1 = yaml.safe_load(phase1_source.read_text(encoding="utf-8"))
    phasef = yaml.safe_load(phasef_source.read_text(encoding="utf-8"))
    _absolutize_paths(phase1, phase1_source.parent)
    _absolutize_paths(phasef, phasef_source.parent)
    phase1_root = (Path(args.output_dir) / "pipeline" / "phase1").resolve()
    phasef_root = phase1_root / "training"
    phase1["extraction"]["output_root"] = str(phase1_root)
    phasef["output"]["output_root"] = str(phasef_root)
    inputs = phasef["inputs"]
    inputs["train_targets_dir"] = str(phase1_root / "student_targets" / "student_safe_targets_alignment")
    inputs["val_targets_dir"] = str(phase1_root / "student_targets" / "student_safe_targets_val")
    inputs["pairing_path"] = str(phase1_root / "layer_pairing" / "teacher_student_layer_pairs.json")
    inputs["train_anchor_dir"] = str(phase1_root / "hidden_states" / "student_alignment")
    inputs["val_anchor_dir"] = str(phase1_root / "hidden_states" / "student_analysis_val")
    for key, value in phase1_updates.items():
        _set_dotted(phase1, key, value)
    for key, value in phasef_updates.items():
        _set_dotted(phasef, key, value)
    phase1_path = config_dir / "phase1.yaml"
    phasef_path = config_dir / "phaseF.yaml"
    phase1_path.write_text(yaml.safe_dump(phase1, sort_keys=False, allow_unicode=True), encoding="utf-8")
    phasef_path.write_text(yaml.safe_dump(phasef, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return phase1_path, phasef_path


def _collect_contract(output_dir: Path, required: list[str], source_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = source_root / "manifest.json"
    eval_candidates = sorted(source_root.glob("eval_suite/epoch_*/final_summary.json"))
    sources = {
        "run_manifest.json": manifest,
        "training_manifest.json": manifest,
        "parameter_budget.json": manifest,
        "eval_predictions.jsonl": next(iter(sorted(source_root.glob("eval_suite/epoch_*/*predictions.jsonl"))), None),
    }
    for name in required:
        destination = output_dir / name
        source = sources.get(name)
        if source is not None and Path(source).is_file():
            destination.write_bytes(Path(source).read_bytes())
        elif name.endswith(".json") and eval_candidates:
            destination.write_bytes(eval_candidates[-1].read_bytes())
        elif not destination.is_file():
            raise FileNotFoundError(
                f"real backend did not produce completion artifact {name}; "
                "the cell remains FAILED instead of fabricating a result"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal single-cell backend; use scripts/30_ablation.py.")
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-artifacts", default="[]")
    parser.add_argument("--analysis-handler", default="")
    parser.add_argument("--cell-spec", default="{}")
    parser.add_argument("--pair", default="qwen35_9b_to_08b")
    parser.add_argument("--dataset", default="pan")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--device", choices=["npu", "ppu", "cuda", "cpu"], default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--phase1-updates", default="{}")
    parser.add_argument("--phasef-updates", default="{}")
    parser.add_argument("--phase1-stage-extras", default="{}")
    parser.add_argument("--disable-dataset-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = _json_list(args.required_artifacts, "--required-artifacts")
    if args.analysis_handler:
        from src.ablations.handlers import execute_handler

        execute_handler(
            args.analysis_handler,
            _json_mapping(args.cell_spec, "--cell-spec"),
            output_dir=Path(args.output_dir),
            required_artifacts=required,
        )
        return 0
    phase1_path, phasef_path = _stage_configs(
        args,
        _json_mapping(args.phase1_updates, "--phase1-updates"),
        _json_mapping(args.phasef_updates, "--phasef-updates"),
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "15_run_oneclick.py"),
        "full",
        "--baseline",
        args.dataset,
        "--pair",
        args.pair,
        "--device",
        args.device,
        "--device-id",
        str(args.device_id),
        "--phase1-config",
        str(phase1_path),
        "--phasef-config",
        str(phasef_path),
        "--phase1-stage-extras=" + json.dumps(_json_mapping(args.phase1_stage_extras, "--phase1-stage-extras")),
        "--cell-id",
        args.cell_id,
    ]
    if args.disable_dataset_overrides:
        command.append("--disable-dataset-overrides")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode:
        return int(result.returncode)
    _collect_contract(Path(args.output_dir), required, Path(args.output_dir) / "pipeline" / "phase1" / "training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
