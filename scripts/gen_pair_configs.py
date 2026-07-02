"""Generate per-pair config yamls from the canonical Qwen3.5-9B->0.8B templates.

For each extension pair in src/pairs.py PAIRS (Llama-3.x / Qwen3), this loads the
canonical ``qwen35_9b_to_08b`` template configs, string-replaces the model
identifier tokens (pair_id / teacher+student name+tag) in BOTH the yaml text and
the filename, sets ``lora.target_modules`` to the pair's architecture-correct
list, and writes the result to configs/. Hyperparameters are inherited verbatim
from the templates (user decision: reuse 9B->0.8B overrides across pairs).

Backward compatibility: the default/template pair (qwen35_9b_to_08b) is NEVER
regenerated; its configs are the source of truth.

Usage:
    python scripts/gen_pair_configs.py --dry-run        # list what would be written
    python scripts/gen_pair_configs.py                  # generate ALL extension pairs
    python scripts/gen_pair_configs.py --pair qwen3_8b_to_06b
    python scripts/gen_pair_configs.py --force          # overwrite existing outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairs import PAIRS, apply_tokens, list_generated_pairs

CONFIGS = PROJECT_ROOT / "configs"

# The 6 fixed datasets (config tokens). "" = the base/pan variant.
DATASETS = ["", "safety_tuned_llamas", "coconot", "c5", "wildjailbreak", "wildguardmix"]


def _template_basenames() -> list[str]:
    """Curated list of canonical template configs the generator re-targets.

    Covers the 'core' variants per the user decision: ours + sft1 (phase1 +
    phaseF[+_sft1]) and the sft / distill / nosft-eval baselines for the 6
    datasets, plus the teacher nosft eval. Excludes random/bothpole and all
    runtime-temp (_launcher_/_safety_ uuid) configs.
    """
    names = [
        "qwen35_9b_to_08b_phase1_npu.yaml",        # ours/sft1 Phase 1 precompute
        "qwen35_9b_to_08b_phaseF_npu.yaml",        # ours Phase F
        "qwen35_9b_to_08b_phaseF_npu_sft1.yaml",   # sft1 ablation Phase F
        "baseline_eval_qwen35_9b_npu.yaml",        # teacher (nosft) eval
    ]
    for ds in DATASETS:
        suffix = f"_{ds}" if ds else ""
        names.append(f"baseline_sft_qwen35_08b{suffix}_npu.yaml")
        names.append(f"baseline_distill_qwen35_9b_to_08b{suffix}_npu.yaml")
        names.append(f"baseline_eval_qwen35_08b{suffix}_npu.yaml")
    return names


def _namespace_output_roots(node, pair_id: str, student_tag: str, teacher_tag: str) -> None:
    """Make every ``output_root`` unique per model so different students/pairs do
    not collide. Templates already embed pair_id / student / teacher tags in most
    output roots (phase1/phaseF/distill -> pair; sft/nosft -> model); the only
    model-agnostic ones are the per-dataset eval roots (``eval_<ds>_npu``), which
    we student-namespace so they stay SHARED across same-student pairs (computed
    once) but disjoint across different students."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "output_root" and isinstance(v, str) and "outputs/" in v:
                if not any(tok and tok in v for tok in (pair_id, student_tag, teacher_tag)):
                    head, _, base = v.rpartition("/")
                    node[k] = f"{head}/{student_tag}_{base}" if head else f"{student_tag}_{base}"
            else:
                _namespace_output_roots(v, pair_id, student_tag, teacher_tag)
    elif isinstance(node, list):
        for item in node:
            _namespace_output_roots(item, pair_id, student_tag, teacher_tag)


def _apply_device_map(data: dict, out_name: str, device_map, only_prefixes=None) -> None:
    """Set device_map on the full-finetune baseline configs (sft: model;
    distill: teacher+student) so a too-big model shards across cards. Only
    baseline_sft_/baseline_distill_ get it -- ours (LoRA phaseF) and eval/nosft
    fit one card. ``only_prefixes`` (pair spec "device_map_only") further
    restricts it, e.g. ["baseline_distill_"] for pairs whose sft fits 1 die but
    whose 8B-teacher distill is fragmentation-marginal. No-op without device_map."""
    if not device_map or not isinstance(data, dict):
        return
    if only_prefixes and not any(out_name.startswith(p) for p in only_prefixes):
        return
    if out_name.startswith("baseline_sft_") and isinstance(data.get("model"), dict):
        data["model"]["device_map"] = device_map
    elif out_name.startswith("baseline_distill_"):
        for key in ("teacher", "student"):
            if isinstance(data.get(key), dict):
                data[key]["device_map"] = device_map


def _render_one(template: Path, pair_id: str, *, dry_run: bool, force: bool) -> str:
    spec = PAIRS[pair_id]
    target_modules = spec["target_modules"]
    out_name = apply_tokens(template.name, pair_id)
    out_path = CONFIGS / out_name
    if out_name == template.name:
        return f"SKIP (no token change) {template.name}"
    if out_path.exists() and not force and not dry_run:
        return f"EXISTS (use --force) {out_name}"

    text = apply_tokens(template.read_text(encoding="utf-8"), pair_id)
    data = yaml.safe_load(text)
    if isinstance(data, dict) and isinstance(data.get("lora"), dict):
        data["lora"]["target_modules"] = list(target_modules)
    _apply_device_map(data, out_name, spec.get("device_map"), spec.get("device_map_only"))
    _namespace_output_roots(data, pair_id, spec["student"]["tag"], spec["teacher"]["tag"])
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if dry_run:
        return f"WOULD WRITE {out_name}"
    out_path.write_text(rendered, encoding="utf-8")
    return f"WROTE {out_name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="", help="One pair_id (default: all extension pairs).")
    ap.add_argument("--dry-run", action="store_true", help="List outputs without writing.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    pairs = [args.pair] if args.pair else list_generated_pairs()
    unknown = [p for p in pairs if p not in PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {unknown}. Known: {sorted(PAIRS)}")

    templates = _template_basenames()
    missing = [t for t in templates if not (CONFIGS / t).exists()]
    if missing:
        print(f"[gen] WARNING: {len(missing)} template(s) not found, skipped:")
        for m in missing:
            print(f"        - {m}")

    total = 0
    for pair_id in pairs:
        print(f"\n=== pair {pair_id} ({PAIRS[pair_id]['teacher']['name']} -> {PAIRS[pair_id]['student']['name']}) ===")
        if PAIRS[pair_id].get("tbd"):
            print(f"    NOTE (TBD): {PAIRS[pair_id]['tbd']}")
        for t in templates:
            tpath = CONFIGS / t
            if not tpath.exists():
                continue
            msg = _render_one(tpath, pair_id, dry_run=args.dry_run, force=args.force)
            print(f"    {msg}")
            if msg.startswith(("WROTE", "WOULD WRITE")):
                total += 1
    verb = "would generate" if args.dry_run else "generated"
    print(f"\n[gen] {verb} {total} config(s) across {len(pairs)} pair(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
