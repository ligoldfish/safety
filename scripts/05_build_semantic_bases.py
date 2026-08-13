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
from src.ablations.strategies.bridge import (
    fit_orthogonal_procrustes,
    fit_ridge,
    match_embedding_nearest,
    match_token_strings,
    validate_bridge_mode,
    vocabularies_identical,
)
from src.phase_b.hidden_states import load_hidden_state_split
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
    parser.add_argument(
        "--bridge-mode",
        choices=["vocabulary", "token_string", "embedding_nearest", "ridge", "orthogonal_procrustes"],
        default="vocabulary",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--min-match-coverage", type=float, default=0.8)
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

    teacher_tokenizer, model, meta = load_hf_model(
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
    basis_result = build_semantic_basis_from_lm_head(
        model.lm_head.weight,
        chunk_size=args.chunk_size,
        storage_dtype=args.storage_dtype,
    )
    teacher_embeddings = (
        model.get_input_embeddings().weight.detach().cpu()
        if args.bridge_mode == "embedding_nearest"
        else None
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
        student_tokenizer, student_model, _ = load_hf_model(
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
        student_basis = build_semantic_basis_from_lm_head(
            student_model.lm_head.weight,
            chunk_size=args.chunk_size,
            storage_dtype=args.storage_dtype,
        )
        student_embeddings = (
            student_model.get_input_embeddings().weight.detach().cpu()
            if args.bridge_mode == "embedding_nearest"
            else None
        )
        student_payload = _build_payload(
            model_cfg=cfg.student,
            model_tag="student",
            basis_result=student_basis,
            source_path=cfg.student.path,
        )
        _save_basis(student_path, student_payload)

    if same_model:
        student_tokenizer = teacher_tokenizer
        student_model = model
        student_embeddings = teacher_embeddings

    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()
    tokenizer_shared = vocabularies_identical(teacher_vocab, student_vocab)
    bridge_mode = validate_bridge_mode(args.bridge_mode, tokenizer_shared=tokenizer_shared)
    bridge_payload: Dict[str, object] = {
        "bridge_mode": bridge_mode,
        "tokenizer_shared": tokenizer_shared,
        "teacher_model": cfg.teacher.name,
        "student_model": cfg.student.name,
    }
    if bridge_mode in {"vocabulary", "token_string", "embedding_nearest"}:
        if bridge_mode == "vocabulary":
            token_result = match_token_strings(teacher_vocab, student_vocab)
            if not tokenizer_shared:
                raise ValueError("vocabulary bridge requires exactly identical tokenizer vocabularies")
        elif bridge_mode == "token_string":
            token_result = match_token_strings(
                teacher_vocab,
                student_vocab,
                teacher_special_ids=set(getattr(teacher_tokenizer, "all_special_ids", [])),
                student_special_ids=set(getattr(student_tokenizer, "all_special_ids", [])),
            )
        else:
            if teacher_embeddings is None or student_embeddings is None:
                raise RuntimeError("embedding bridge weights were not retained during model loading")
            if teacher_embeddings.size(1) != student_embeddings.size(1):
                anchors = match_token_strings(teacher_vocab, student_vocab)
                if anchors.coverage < args.min_match_coverage:
                    raise ValueError("insufficient shared-token coverage to align cross-dimensional embeddings")
                teacher_ids = torch.tensor(sorted(anchors.teacher_to_student), dtype=torch.long)
                student_ids = torch.tensor([anchors.teacher_to_student[int(i)] for i in teacher_ids], dtype=torch.long)
                align = fit_ridge(
                    teacher_embeddings.index_select(0, teacher_ids),
                    student_embeddings.index_select(0, student_ids),
                    alpha=args.ridge_alpha,
                )
                teacher_embeddings = teacher_embeddings @ align
            token_result = match_embedding_nearest(
                teacher_embeddings,
                student_embeddings,
                min_cosine=0.0,
                teacher_special_ids=set(getattr(teacher_tokenizer, "all_special_ids", [])),
                student_special_ids=set(getattr(student_tokenizer, "all_special_ids", [])),
            )
        if token_result.coverage < args.min_match_coverage:
            raise ValueError(
                f"bridge match coverage {token_result.coverage:.4f} is below {args.min_match_coverage:.4f}"
            )
        bridge_payload.update(
            teacher_to_student={str(k): int(v) for k, v in token_result.teacher_to_student.items()},
            coverage=token_result.coverage,
            conflicts=token_result.conflicts,
            unmatched_teacher_ids=list(token_result.unmatched_teacher_ids),
        )
    else:
        pairing = json.loads(
            (Path(cfg.extraction.output_root) / "layer_pairing" / "teacher_student_layer_pairs.json").read_text(encoding="utf-8")
        )["pairs"]
        teacher_split = load_hidden_state_split(
            Path(cfg.extraction.output_root) / "hidden_states" / "teacher_alignment"
        )
        student_split = load_hidden_state_split(
            Path(cfg.extraction.output_root) / "hidden_states" / "student_alignment"
        )
        if teacher_split.sample_ids != student_split.sample_ids:
            raise ValueError("teacher/student alignment sample IDs must match exactly for hidden bridge fitting")
        mappings = {}
        for pair_idx, pair in enumerate(pairing):
            teacher_hidden = teacher_split.layer_tensors[int(pair["teacher_layer"])]
            student_hidden = student_split.layer_tensors[int(pair["student_layer"])]
            mapping = (
                fit_ridge(teacher_hidden, student_hidden, alpha=args.ridge_alpha)
                if bridge_mode == "ridge"
                else fit_orthogonal_procrustes(teacher_hidden, student_hidden)
            )
            mappings[str(pair_idx)] = mapping
        bridge_payload.update(pairing=pairing, mappings=mappings, alignment_sample_ids=teacher_split.sample_ids)

    bridge_path = output_root / "bridge_artifact.pt"
    torch.save(bridge_payload, bridge_path)

    write_json(
        output_root / "vocab_index_map.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "teacher_model": cfg.teacher.name,
            "student_model": cfg.student.name,
            "teacher_basis_path": str(teacher_path),
            "student_basis_path": str(student_path),
            "vocab_size": basis_result.vocab_size,
            "tokenizer_shared": tokenizer_shared,
            "bridge_mode": bridge_mode,
            "bridge_artifact_path": str(bridge_path),
            "note": "Token ids may be reused only when tokenizer vocabularies are exactly identical.",
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
            "bridge_mode": bridge_mode,
            "bridge_artifact_path": str(bridge_path),
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
