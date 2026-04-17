from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.semantic_basis import build_semantic_basis_from_lm_head
from src.models.hf_loader import load_hf_model
from src.utils.config import Phase1ModelConfig, load_phase1_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log_kv, setup_stage_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build teacher/student semantic bases from the LM head pseudoinverse."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen35_08b_phase1_cpu.yaml",
        help="Path to the phase-A YAML config.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Row chunk size for gram accumulation and basis construction.",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        help="On-disk dtype for the semantic basis tensor.",
    )
    return parser.parse_args()


def _build_payload(
    *,
    model_cfg: Phase1ModelConfig,
    model_tag: str,
    basis_result,
    source_path: str,
) -> Dict[str, object]:
    return {
        "model_name": model_cfg.name,
        "model_path": model_cfg.path,
        "model_tag": model_tag,
        "source_lm_head_path": source_path,
        "vocab_size": basis_result.vocab_size,
        "hidden_size": basis_result.hidden_size,
        "basis": basis_result.basis,
        "token_ids": basis_result.token_ids,
        "normalized": basis_result.normalized,
        "gram_condition_number": basis_result.gram_condition_number,
    }


def _save_basis(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    cfg = load_phase1_config(args.config)
    set_global_seed(cfg.seed)

    output_root = ensure_dir(Path(cfg.extraction.output_root) / "semantic_bases")
    logger, log_path = setup_stage_logger("05_build_semantic_bases", Path(cfg.extraction.output_root) / "logs")
    teacher_path = output_root / "teacher_semantic_basis.pt"
    student_path = output_root / "student_semantic_basis.pt"
    log_kv(
        logger,
        "semantic_basis_setup",
        config_path=str(Path(args.config).resolve()),
        teacher_model=cfg.teacher.name,
        student_model=cfg.student.name,
        chunk_size=int(args.chunk_size),
        storage_dtype=args.storage_dtype,
        log_path=str(log_path),
    )

    tokenizer, model, meta = load_hf_model(
        model_path=cfg.teacher.path,
        device_map=cfg.teacher.device_map,
        torch_dtype=cfg.teacher.torch_dtype,
        runtime_backend=cfg.teacher.runtime_backend,
        runtime_device=cfg.teacher.runtime_device,
        trust_remote_code=cfg.teacher.trust_remote_code,
        local_files_only=cfg.teacher.local_files_only,
        attn_implementation=cfg.teacher.attn_implementation,
    )
    _ = tokenizer
    basis_result = build_semantic_basis_from_lm_head(
        model.lm_head.weight,
        chunk_size=args.chunk_size,
        storage_dtype=args.storage_dtype,
    )
    teacher_payload = _build_payload(
        model_cfg=cfg.teacher,
        model_tag="teacher",
        basis_result=basis_result,
        source_path=cfg.teacher.path,
    )
    _save_basis(teacher_path, teacher_payload)

    same_model = (
        cfg.teacher.path == cfg.student.path
        and cfg.teacher.name == cfg.student.name
        and meta["hidden_size"] == basis_result.hidden_size
    )
    if same_model:
        student_payload = dict(teacher_payload)
        student_payload["model_tag"] = "student"
        student_payload["model_name"] = cfg.student.name
        student_payload["model_path"] = cfg.student.path
        _save_basis(student_path, student_payload)
    else:
        del model
        _, student_model, _ = load_hf_model(
            model_path=cfg.student.path,
            device_map=cfg.student.device_map,
            torch_dtype=cfg.student.torch_dtype,
            runtime_backend=cfg.student.runtime_backend,
            runtime_device=cfg.student.runtime_device,
            trust_remote_code=cfg.student.trust_remote_code,
            local_files_only=cfg.student.local_files_only,
            attn_implementation=cfg.student.attn_implementation,
        )
        student_basis = build_semantic_basis_from_lm_head(
            student_model.lm_head.weight,
            chunk_size=args.chunk_size,
            storage_dtype=args.storage_dtype,
        )
        student_payload = _build_payload(
            model_cfg=cfg.student,
            model_tag="student",
            basis_result=student_basis,
            source_path=cfg.student.path,
        )
        _save_basis(student_path, student_payload)

    write_json(
        output_root / "vocab_index_map.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_model": cfg.teacher.name,
            "student_model": cfg.student.name,
            "teacher_basis_path": str(teacher_path),
            "student_basis_path": str(student_path),
            "vocab_size": basis_result.vocab_size,
            "tokenizer_shared": same_model,
            "note": "Token ids follow the tokenizer vocabulary indices directly.",
        },
    )
    write_json(
        output_root / "manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_basis_path": str(teacher_path),
            "student_basis_path": str(student_path),
            "same_model_reused": same_model,
            "chunk_size": args.chunk_size,
            "storage_dtype": args.storage_dtype,
            "vocab_size": basis_result.vocab_size,
            "hidden_size": basis_result.hidden_size,
        },
    )
    log_kv(
        logger,
        "semantic_basis_complete",
        teacher_basis_path=str(teacher_path),
        student_basis_path=str(student_path),
        same_model_reused=bool(same_model),
        vocab_size=int(basis_result.vocab_size),
        hidden_size=int(basis_result.hidden_size),
        teacher_gram_condition_number=float(teacher_payload["gram_condition_number"]),
    )

    print(
        json.dumps(
            {
                "teacher_basis_path": str(teacher_path),
                "student_basis_path": str(student_path),
                "same_model_reused": same_model,
                "vocab_size": basis_result.vocab_size,
                "hidden_size": basis_result.hidden_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
