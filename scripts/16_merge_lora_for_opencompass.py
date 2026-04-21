from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import load_eval_config
from src.models import inject_lora_modules_by_names
from src.models.hf_loader import load_hf_model
from src.models.lora_utils import LoRALinear
from src.utils.io import ensure_dir
from src.utils.logging import log_kv, setup_stage_logger


def _merge_lora_into_base(model: torch.nn.Module) -> int:
    merged = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base_layer
        lora_A = module.lora_A.detach().to(device=base.weight.device, dtype=torch.float32)
        lora_B = module.lora_B.detach().to(device=base.weight.device, dtype=torch.float32)
        delta = lora_B @ lora_A
        if module.output_mask is not None:
            mask = module.output_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(1)
            delta = delta * mask
        delta = delta * float(module.scaling)
        with torch.no_grad():
            base.weight.add_(delta.to(dtype=base.weight.dtype))
        parent_path, _, attr_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr_name, base)
        merged += 1
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a Codex-style LoRA adapter into the base Qwen3.5 model and save a standalone "
            "HuggingFace checkpoint suitable for OpenCompass evaluation."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a baseline eval YAML config (supplies base model path and loading options).",
    )
    parser.add_argument("--adapter-manifest", required=True, help="Path to the LoRA manifest JSON.")
    parser.add_argument("--adapter-checkpoint", required=True, help="Path to the LoRA trainable checkpoint (.pt).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the merged HuggingFace checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)

    tokenizer, model, _ = load_hf_model(
        model_path=cfg.model.path,
        device_map=cfg.model.device_map,
        torch_dtype=cfg.model.torch_dtype,
        chat_template_enable_thinking=cfg.model.chat_template_enable_thinking,
        runtime_backend=cfg.model.runtime_backend,
        runtime_device=cfg.model.runtime_device,
        trust_remote_code=cfg.model.trust_remote_code,
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
    )

    manifest_path = Path(args.adapter_manifest)
    checkpoint_path = Path(args.adapter_checkpoint)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Adapter manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {checkpoint_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inject_lora_modules_by_names(
        model,
        module_names=manifest["lora_modules"],
        rank=int(manifest["lora_rank"]),
        alpha=float(manifest["lora_alpha"]),
        dropout=float(manifest["lora_dropout"]),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(payload["trainable_state_dict"], strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected adapter checkpoint keys: {load_result.unexpected_keys}")
    missing_lora = [key for key in load_result.missing_keys if ".lora_" in key]
    if missing_lora:
        raise ValueError(f"Missing LoRA weights while loading adapter: {missing_lora}")

    merged_count = _merge_lora_into_base(model)
    model.eval()

    output_dir = ensure_dir(args.output_dir)
    logger, log_path = setup_stage_logger("16_merge_lora_for_opencompass", output_dir / "logs")
    log_kv(
        logger,
        "merge_begin",
        config_path=str(Path(args.config).resolve()),
        adapter_manifest=str(manifest_path.resolve()),
        adapter_checkpoint=str(checkpoint_path.resolve()),
        output_dir=str(output_dir),
        merged_modules=merged_count,
    )

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "model_name": cfg.model.name,
        "base_model_path": cfg.model.path,
        "adapter_manifest": str(manifest_path.resolve()),
        "adapter_checkpoint": str(checkpoint_path.resolve()),
        "merged_modules": merged_count,
        "merged_checkpoint_dir": str(output_dir.resolve()),
        "log_path": str(log_path),
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_kv(logger, "merge_complete", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
