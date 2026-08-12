from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.catalog import load_catalog
from src.ablations.planner import build_catalog_plan, validate_plan
from src.ablations.preflight import (
    AssetRequirement,
    PreflightIssue,
    PreflightReport,
    requirements_from_manifest,
    run_preflight,
)
from src.ablations.runner import AblationRunner, RunnerContext, RunnerError
from src.ablations.schema import ExperimentCell, ExperimentPlan


DEFAULT_CATALOG = PROJECT_ROOT / "configs" / "ablations" / "catalog.yaml"


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_plan(path: str | Path) -> ExperimentPlan:
    source = Path(path)
    cells = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cells.append(
                ExperimentCell(
                    cell_id=str(row["cell_id"]),
                    experiment_id=str(row["experiment_id"]),
                    axes=dict(row.get("axes", {})),
                    overrides=dict(row.get("overrides", {})),
                    output_dir=str(row["output_dir"]),
                    depends_on=tuple(row.get("depends_on", ())),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"invalid plan row {line_number}: {source}") from exc
    return validate_plan(ExperimentPlan(schema_version=1, cells=tuple(cells)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, preflight, run, resume, and summarize the complete ICLR ablation suite."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--scope", choices=["main-table", "all", "p0", "p1", "p2"], default="all")
    plan.add_argument("--output-root", default="../outputs/ablations")
    plan.add_argument("--output", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--plan", required=True)
    asset_source = preflight.add_mutually_exclusive_group(required=True)
    asset_source.add_argument("--asset-manifest")
    asset_source.add_argument(
        "--asset-root",
        help="Legacy layout: every asset is a directory named <asset-root>/<asset-id>.",
    )
    preflight.add_argument("--output", default="")

    for name in ("status", "summarize"):
        command = subparsers.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--state-root", required=True)
        command.add_argument("--output", default="")

    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--cell-id", default="")
    run.add_argument("--state-root", default="../outputs/ablation-state")
    run.add_argument("--device", choices=["npu", "ppu", "cuda", "cpu"], default="npu")
    run.add_argument("--device-id", type=int, default=0)
    run.add_argument("--num-devices", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--asset-manifest", default="")
    return parser


def _context(args) -> RunnerContext:
    return RunnerContext(
        project_root=PROJECT_ROOT,
        state_root=Path(args.state_root).expanduser().resolve(),
        python_executable=sys.executable,
        device=args.device,
        device_id=args.device_id,
        num_devices=args.num_devices,
        asset_manifest=(Path(args.asset_manifest).expanduser().resolve() if args.asset_manifest else None),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = load_catalog(args.catalog)
    if args.command == "catalog":
        payload = {
            "schema_version": catalog.schema_version,
            "experiment_count": len(catalog.experiments),
            "experiment_ids": sorted(catalog.experiments),
            "handlers": sorted({item.handler for item in catalog.experiments.values()}),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else "\n".join(payload["experiment_ids"]))
        return 0

    if args.command == "plan":
        plan = build_catalog_plan(catalog, output_root=args.output_root, scope=args.scope)
        _write_jsonl(Path(args.output), (asdict(cell) for cell in plan.cells))
        print(json.dumps({"status": "PLANNED", "cells": len(plan.cells), "output": args.output}))
        return 0

    plan = _read_plan(args.plan)
    if args.command == "preflight":
        requirements: list[AssetRequirement] = []
        missing_issues: list[PreflightIssue] = []
        if args.asset_manifest:
            manifest_path = Path(args.asset_manifest).expanduser().resolve()
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerError(f"invalid asset manifest: {manifest_path}") from exc
            if not isinstance(manifest, dict):
                raise RunnerError("asset manifest must be a JSON object")
            for cell in plan.cells:
                definition = catalog.experiments[cell.experiment_id]
                cell_requirements, missing = requirements_from_manifest(
                    definition.requires,
                    manifest,
                    cell_id=cell.cell_id,
                    base_dir=manifest_path.parent,
                )
                requirements.extend(cell_requirements)
                missing_issues.extend(
                    PreflightIssue(
                        cell_id=cell.cell_id,
                        asset_id=asset_id,
                        code="MANIFEST_KEY_MISSING",
                        category="manifest",
                        message="required asset is not declared in the manifest",
                        suggestion="add a typed path entry to the asset manifest",
                    )
                    for asset_id in missing
                )
        else:
            root = Path(args.asset_root).expanduser().resolve()
            for cell in plan.cells:
                definition = catalog.experiments[cell.experiment_id]
                for asset_id in definition.requires:
                    requirements.append(
                        AssetRequirement(asset_id, root / asset_id, "directory", cell.cell_id)
                    )
        checked = run_preflight(requirements)
        report = PreflightReport(
            "READY" if checked.status == "READY" and not missing_issues else "BLOCKED",
            tuple(missing_issues) + checked.issues,
            checked.checked,
        )
        payload = report.to_dict()
        if args.output:
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if report.status == "READY" else 3

    runner = AblationRunner(catalog, _context(args))
    if args.command == "run":
        if not args.cell_id:
            raise RunnerError("run requires --cell-id; unbounded plan execution is forbidden")
        cell = runner.select_cell(plan.cells, args.cell_id)
        print(json.dumps(runner.run_cell(cell, dry_run=args.dry_run), ensure_ascii=False))
        return 0

    rows = []
    for cell in plan.cells:
        status_path = runner.context.state_root / cell.cell_id / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            status = {"cell_id": cell.cell_id, "state": "PLANNED"}
        if args.command == "summarize" and status["state"] != "COMPLETED":
            continue
        rows.append(status)
    payload = {"schema_version": 1, "command": args.command, "cells": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
