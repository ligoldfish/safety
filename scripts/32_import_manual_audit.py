from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.manual_audit import import_double_annotations
from src.ablations.statistics import cohen_kappa
from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and import two blind manual-annotation files.")
    parser.add_argument("--key", required=True)
    parser.add_argument("--rater-a", required=True)
    parser.add_argument("--rater-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="safe,unsafe,uncertain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_payload = json.loads(Path(args.key).read_text(encoding="utf-8"))
    allowed = {label.strip() for label in args.labels.split(",") if label.strip()}
    imported = import_double_annotations(
        key_payload["mapping"],
        read_jsonl(args.rater_a),
        read_jsonl(args.rater_b),
        allowed_labels=allowed,
    )
    left = [row["rater_a"] for row in imported]
    right = [row["rater_b"] for row in imported]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "validated_annotations.jsonl", imported)
    write_json(
        output / "manual_audit_summary.json",
        {
            "n": len(imported),
            "agreement": sum(a == b for a, b in zip(left, right)) / len(imported),
            "cohens_kappa": cohen_kappa(left, right),
            "allowed_labels": sorted(allowed),
        },
    )


if __name__ == "__main__":
    main()
