from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.layer_pairing import build_layer_pairs
from src.models.hf_loader import load_hf_model
from src.utils.config import load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair teacher key layers to student layers by relative depth."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--key-layers-path",
        type=str,
        default="",
        help="Optional explicit path to teacher_key_layers.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)
    logger, log_path = setup_stage_logger("04_pair_layers", Path(cfg.extraction.output_root) / "logs")

    key_layers_path = (
        Path(args.key_layers_path).resolve()
        if args.key_layers_path
        else Path(cfg.extraction.output_root) / "layer_analysis" / "teacher_key_layers.json"
    )
    key_layer_payload = json.loads(key_layers_path.read_text(encoding="utf-8"))
    teacher_key_layers = [int(layer_idx) for layer_idx in key_layer_payload["key_layers"]]
    if not teacher_key_layers:
        raise ValueError(f"No teacher key layers found in: {key_layers_path}")

    _, teacher_model, teacher_meta = load_hf_model(
        model_path=cfg.teacher.path,
        device_map=cfg.teacher.device_map,
        torch_dtype=cfg.teacher.torch_dtype,
        chat_template_enable_thinking=cfg.teacher.chat_template_enable_thinking,
        runtime_backend=cfg.teacher.runtime_backend,
        runtime_device=cfg.teacher.runtime_device,
        trust_remote_code=cfg.teacher.trust_remote_code,
        local_files_only=cfg.teacher.local_files_only,
        attn_implementation=cfg.teacher.attn_implementation,
    )
    del teacher_model

    _, student_model, student_meta = load_hf_model(
        model_path=cfg.student.path,
        device_map=cfg.student.device_map,
        torch_dtype=cfg.student.torch_dtype,
        chat_template_enable_thinking=cfg.student.chat_template_enable_thinking,
        runtime_backend=cfg.student.runtime_backend,
        runtime_device=cfg.student.runtime_device,
        trust_remote_code=cfg.student.trust_remote_code,
        local_files_only=cfg.student.local_files_only,
        attn_implementation=cfg.student.attn_implementation,
    )
    del student_model

    pairs = build_layer_pairs(
        teacher_key_layers,
        teacher_num_layers=teacher_meta["num_layers"],
        student_num_layers=student_meta["num_layers"],
    )
    output_root = ensure_dir(Path(cfg.extraction.output_root) / "layer_pairing")
    output_path = output_root / "teacher_student_layer_pairs.json"

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "teacher_model": cfg.teacher.name,
        "student_model": cfg.student.name,
        "teacher_num_layers": teacher_meta["num_layers"],
        "student_num_layers": student_meta["num_layers"],
        "pairs": [
            {
                "teacher_layer": pair.teacher_layer,
                "student_layer": pair.student_layer,
                "teacher_relative_depth": pair.teacher_relative_depth,
            }
            for pair in pairs
        ],
        "key_layers_path": str(key_layers_path),
    }
    write_json(output_path, payload)
    log_kv(
        logger,
        "layer_pairing_complete",
        config_path=str(Path(args.config).resolve()),
        teacher_num_layers=int(teacher_meta["num_layers"]),
        student_num_layers=int(student_meta["num_layers"]),
        key_layers=teacher_key_layers,
        pairs=payload["pairs"],
        output_path=str(output_path),
        log_path=str(log_path),
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
