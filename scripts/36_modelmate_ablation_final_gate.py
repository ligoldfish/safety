from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.catalog import load_catalog
from src.ablations.modelmate_pool import (
    FINAL_ROUND_ORDER,
    ROUND_SPECS,
    select_round_cells,
)
from src.ablations.planner import build_catalog_plan


DEFAULT_OUTPUT_ROOT = Path(
    "/opt/dpcvol/datasets/safetytransfer/ablation-outputs/iclr-886760f"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_completion(output_root: Path) -> dict[str, object]:
    output_root = Path(output_root).expanduser().resolve()
    catalog = load_catalog(PROJECT_ROOT / "configs" / "ablations" / "catalog.yaml")
    complete_plan = build_catalog_plan(
        catalog,
        output_root=output_root / "cell-outputs",
        scope="all",
    )
    expected_all = {cell.cell_id for cell in complete_plan.cells}
    seen: set[str] = set()
    completed_seen: set[str] = set()
    blockers: list[str] = []
    rounds: list[dict[str, object]] = []

    for round_name in FINAL_ROUND_ORDER:
        spec = ROUND_SPECS[round_name]
        expected_ids = {
            cell.cell_id
            for cell in select_round_cells(catalog, complete_plan, spec)
        }
        round_root = output_root / "jobs" / round_name
        round_blockers: list[str] = []
        try:
            summary = _read_json(round_root / "pool-summary.json")
            status = _read_json(round_root / "status.json")
        except RuntimeError as exc:
            round_blockers.append(str(exc))
            summary = {}
            status = {}

        if summary:
            if summary.get("status") != "READY":
                round_blockers.append(
                    f"pool status is {summary.get('status')!r}, expected 'READY'"
                )
            if summary.get("dry_run") is not False:
                round_blockers.append("pool summary is not a real execution")
            if summary.get("expected_cells") != spec.expected_cells:
                round_blockers.append(
                    f"expected_cells is {summary.get('expected_cells')!r}, "
                    f"expected {spec.expected_cells}"
                )
            if summary.get("failed_shards") or summary.get("pending_shards"):
                round_blockers.append("pool summary contains failed or pending shards")

        actual_ids: set[str] = set()
        completed_ids: set[str] = set()
        non_completed: list[str] = []
        if status:
            rows = status.get("cells")
            if not isinstance(rows, list):
                round_blockers.append("status.json does not contain a cells list")
            else:
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
                        round_blockers.append("status.json contains a malformed cell row")
                        continue
                    cell_id = row["cell_id"]
                    actual_ids.add(cell_id)
                    if row.get("state") == "COMPLETED":
                        completed_ids.add(cell_id)
                    else:
                        non_completed.append(cell_id)
                if len(rows) != len(expected_ids):
                    round_blockers.append(
                        f"status row count is {len(rows)}, expected {len(expected_ids)}"
                    )
                if actual_ids != expected_ids:
                    round_blockers.append(
                        f"status cell set mismatch: missing={len(expected_ids - actual_ids)}, "
                        f"unexpected={len(actual_ids - expected_ids)}"
                    )
                if non_completed:
                    round_blockers.append(
                        f"{len(non_completed)} cells are not COMPLETED"
                    )

        overlap = seen.intersection(expected_ids)
        if overlap:
            round_blockers.append(f"{len(overlap)} cells overlap an earlier final round")
        seen.update(expected_ids)
        completed_seen.update(completed_ids.intersection(expected_ids))
        blockers.extend(f"{round_name}: {item}" for item in round_blockers)
        rounds.append(
            {
                "round": round_name,
                "expected_cells": spec.expected_cells,
                "completed_cells": len(completed_ids.intersection(expected_ids)),
                "status": "PASS" if not round_blockers else "BLOCKED",
                "blockers": round_blockers,
            }
        )

    if seen != expected_all:
        blockers.append(
            f"final round coverage mismatch: missing={len(expected_all - seen)}, "
            f"unexpected={len(seen - expected_all)}"
        )
    if len(expected_all) != 509:
        blockers.append(f"catalog drift: expected 509 cells, got {len(expected_all)}")

    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": "READY" if not blockers else "BLOCKED",
        "output_root": str(output_root),
        "expected_cells": len(expected_all),
        "covered_cells": len(completed_seen),
        "rounds": rounds,
        "blockers": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed final completion gate for all 509 ICLR ablation cells."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_completion(Path(args.output_root))
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(args.output_root).expanduser().resolve() / "final-completion.json"
    )
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "READY" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
