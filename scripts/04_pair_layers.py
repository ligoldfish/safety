from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.layer_pairing import build_layer_pairs, map_teacher_to_student_layer
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
    parser.add_argument(
        "--no-student-layer-dedup",
        action="store_true",
        help=(
            "Disable greedy de-duplication of teacher layers that map to the "
            "same student layer. With dedup ON (default) we walk teacher layers "
            "in descending score order and skip any whose student-layer slot is "
            "already taken; this preserves K independent alignment positions "
            "across the K selected teacher layers. With dedup OFF we keep the "
            "raw top-K from 02_analyze_teacher_layers, which may collide and "
            "force 08_recompose_student_targets into mean reduction."
        ),
    )
    return parser.parse_args()


def _select_dedup_teacher_layers(
    *,
    ranked_layer_rows: list[dict],
    raw_top_k: list[int],
    top_k: int,
    teacher_num_layers: int,
    student_num_layers: int,
) -> tuple[list[int], list[int]]:
    """Greedy: pick the highest-scoring teacher layers whose student-layer
    mappings are pairwise distinct. Falls back to the raw top-K when ranked
    rows are missing (e.g. older 02 outputs without ``layer_rows``)."""

    if not ranked_layer_rows:
        return list(raw_top_k), []
    selected: list[int] = []
    occupied: set[int] = set()
    skipped: list[int] = []
    for row in ranked_layer_rows:
        if len(selected) >= top_k:
            break
        try:
            teacher_layer = int(row["layer_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        student_layer = map_teacher_to_student_layer(
            teacher_layer,
            teacher_num_layers=teacher_num_layers,
            student_num_layers=student_num_layers,
        )
        if student_layer in occupied:
            skipped.append(teacher_layer)
            continue
        selected.append(teacher_layer)
        occupied.add(student_layer)
    if len(selected) < top_k:
        raise ValueError(
            f"Dedup selection produced only {len(selected)} teacher layers but "
            f"top_k={top_k} requested. teacher_num_layers={teacher_num_layers}, "
            f"student_num_layers={student_num_layers}. Either lower top_k or "
            "re-run 02_analyze_teacher_layers with a larger candidate pool."
        )
    return selected, skipped


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
    raw_top_k_layers = [int(layer_idx) for layer_idx in key_layer_payload["key_layers"]]
    if not raw_top_k_layers:
        raise ValueError(f"No teacher key layers found in: {key_layers_path}")
    ranked_layer_rows = list(key_layer_payload.get("layer_rows") or [])
    top_k = int(key_layer_payload.get("top_k") or len(raw_top_k_layers))

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

    teacher_num_layers = int(teacher_meta["num_layers"])
    student_num_layers = int(student_meta["num_layers"])

    if args.no_student_layer_dedup:
        teacher_key_layers = list(raw_top_k_layers)
        dedup_skipped: list[int] = []
        dedup_applied = False
    else:
        teacher_key_layers, dedup_skipped = _select_dedup_teacher_layers(
            ranked_layer_rows=ranked_layer_rows,
            raw_top_k=raw_top_k_layers,
            top_k=top_k,
            teacher_num_layers=teacher_num_layers,
            student_num_layers=student_num_layers,
        )
        dedup_applied = True

    pairs = build_layer_pairs(
        teacher_key_layers,
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
    )
    output_root = ensure_dir(Path(cfg.extraction.output_root) / "layer_pairing")
    output_path = output_root / "teacher_student_layer_pairs.json"

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "teacher_model": cfg.teacher.name,
        "student_model": cfg.student.name,
        "teacher_num_layers": teacher_num_layers,
        "student_num_layers": student_num_layers,
        "pairs": [
            {
                "teacher_layer": pair.teacher_layer,
                "student_layer": pair.student_layer,
                "teacher_relative_depth": pair.teacher_relative_depth,
            }
            for pair in pairs
        ],
        "key_layers_path": str(key_layers_path),
        "student_layer_dedup_applied": dedup_applied,
        "raw_top_k_by_score": raw_top_k_layers,
        "dedup_skipped_teacher_layers": dedup_skipped,
    }
    write_json(output_path, payload)
    log_kv(
        logger,
        "layer_pairing_complete",
        config_path=str(Path(args.config).resolve()),
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        key_layers=teacher_key_layers,
        raw_top_k_by_score=raw_top_k_layers,
        student_layer_dedup_applied=dedup_applied,
        dedup_skipped_teacher_layers=dedup_skipped,
        pairs=payload["pairs"],
        output_path=str(output_path),
        log_path=str(log_path),
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
