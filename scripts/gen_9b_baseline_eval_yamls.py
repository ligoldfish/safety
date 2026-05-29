"""Derive 9b safety-baseline eval YAMLs from existing 0.8b counterparts.

For each ``configs/baseline_eval_qwen35_08b_<baseline>_<device>.yaml`` write a
``configs/baseline_eval_qwen35_9b_<baseline>_<device>.yaml`` with the model
swapped to Qwen3.5-9B (bf16) and memory/length budgets bumped to match the
existing ``configs/baseline_eval_qwen35_9b_<device>.yaml`` 9b PAN config.

Output root is rewritten to ``../outputs/baselines/eval_<baseline>_9b_<device>``
so 0.8b and 9b nosft runs do not collide.

Idempotent: re-running overwrites existing 9b YAMLs.
"""
from __future__ import annotations

import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"

NINEB_OVERRIDES = {
    "model.name": "Qwen3.5-9B",
    "model.path": "../models/Qwen3.5-9B",
    "model.torch_dtype": "bfloat16",
    "runtime.max_length": 8192,
    "runtime.batch_size": 4,
}

# datasets.pan.max_new_tokens bumped to match 9b PAN yaml convention.
DATASET_PAN_MAX_NEW_TOKENS = 2048


def set_dotted(data: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def derive(src_path: Path, dst_path: Path, baseline: str, device: str) -> None:
    data = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{src_path} did not parse to a mapping")

    for dotted, value in NINEB_OVERRIDES.items():
        set_dotted(data, dotted, value)

    if isinstance(data.get("datasets"), dict) and isinstance(data["datasets"].get("pan"), dict):
        data["datasets"]["pan"]["max_new_tokens"] = DATASET_PAN_MAX_NEW_TOKENS

    set_dotted(
        data,
        "output.output_root",
        f"../outputs/baselines/eval_{baseline}_9b_{device}",
    )

    dst_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    written: list[str] = []
    for src in sorted(CONFIGS.glob("baseline_eval_qwen35_08b_*_npu.yaml")) + \
               sorted(CONFIGS.glob("baseline_eval_qwen35_08b_*_ppu.yaml")):
        stem = src.stem  # baseline_eval_qwen35_08b_<baseline>_<device>
        prefix = "baseline_eval_qwen35_08b_"
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        # suffix = "<baseline>_<device>"; device is last underscore-segment
        if suffix.endswith("_npu"):
            device = "npu"
            baseline = suffix[:-len("_npu")]
        elif suffix.endswith("_ppu"):
            device = "ppu"
            baseline = suffix[:-len("_ppu")]
        else:
            continue
        dst = CONFIGS / f"baseline_eval_qwen35_9b_{baseline}_{device}.yaml"
        derive(src, dst, baseline, device)
        written.append(str(dst.relative_to(PROJECT_ROOT)))

    for path in written:
        print(f"[gen-9b-eval] wrote {path}")
    print(f"[gen-9b-eval] total: {len(written)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
