"""Run the 12 validation-only P0-07 fairness-search candidates.

The command is intentionally bounded: ``run`` accepts exactly one trial ID.
Search candidates skip all test evaluation, then WildGuard scores only the
saved final validation generations. ``collect`` refuses incomplete/tampered
evidence and atomically emits the formal search ledger used by P0-07.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.fairness_search import (
    FairnessSearchError,
    build_fairness_search_trials,
    collect_fairness_search_ledger,
    compile_fairness_judge_command,
    compile_fairness_search_command,
)
from src.ablations.preflight import inspect_model_directory
from src.ablations.preflight import run_preflight, training_data_requirements, training_model_requirements
from src.ablations.schema import ExperimentCell


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Write the immutable 12-trial JSONL plan.")
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--out", type=Path, default=None)

    run = sub.add_parser("run", help="Run exactly one candidate and its validation judge.")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--trial-id", required=True)
    run.add_argument("--judge-model", type=Path, required=True)
    run.add_argument("--pair", default="qwen35_9b_to_08b")
    run.add_argument("--device", choices=["npu", "ppu"], required=True)
    run.add_argument("--device-id", type=int, default=0)
    run.add_argument("--dry-run", action="store_true")

    collect = sub.add_parser("collect", help="Verify all trials and write search-ledger.jsonl.")
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    return parser


def _absolute_root(value: Path) -> Path:
    if not value.is_absolute():
        raise FairnessSearchError("--output-root must be an absolute persistent path")
    root = value.resolve()
    normalized = root.as_posix().lower().rstrip("/") + "/"
    forbidden = ("/tmp/", "/cache/", "/home/work/user-job-dir/app/")
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents or any(
        normalized.startswith(prefix) for prefix in forbidden
    ):
        raise FairnessSearchError(
            "--output-root must be persistent and outside the uploaded source tree"
        )
    return root


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _absolute_root(args.output_root)
    trials = build_fairness_search_trials(root)
    if args.command == "plan":
        destination = _absolute_root(args.out or root / "fairness-search-plan.jsonl")
        _atomic_jsonl(destination, [trial.to_dict() for trial in trials])
        print(json.dumps({"trials": len(trials), "plan": str(destination)}, ensure_ascii=False))
        return 0
    if args.command == "collect":
        destination = _absolute_root(args.out)
        rows = collect_fairness_search_ledger(trials, destination)
        print(json.dumps({"rows": len(rows), "ledger": str(destination)}, ensure_ascii=False))
        return 0

    matches = [trial for trial in trials if trial.trial_id == args.trial_id]
    if len(matches) != 1:
        known = ", ".join(trial.trial_id for trial in trials)
        raise FairnessSearchError(f"unknown --trial-id {args.trial_id!r}; choose one of: {known}")
    trial = matches[0]
    worker = compile_fairness_search_command(
        trial,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        device=args.device,
        device_id=args.device_id,
        pair=args.pair,
    )
    if args.dry_run:
        print(json.dumps({"trial": trial.to_dict(), "worker": worker}, ensure_ascii=False))
        return 0
    judge_report = inspect_model_directory(args.judge_model, cell_id=trial.trial_id)
    if judge_report.status != "READY":
        codes = ", ".join(issue.code for issue in judge_report.issues)
        raise FairnessSearchError(f"--judge-model is incomplete: {codes}")
    environment = dict(os.environ)
    cell = ExperimentCell(
        cell_id=trial.trial_id,
        experiment_id="P0-07",
        axes={"dataset": trial.dataset, "method": trial.method, "pair": args.pair},
        overrides={},
        output_dir=str(trial.output_dir),
    )
    source_data = tuple(
        requirement
        for requirement in training_data_requirements(
            cell, project_root=PROJECT_ROOT, environment=environment, device=args.device
        )
        if requirement.asset_id not in {"training_safety_eval", "training_pan_test"}
    )
    requirements = (
        *training_model_requirements(cell, project_root=PROJECT_ROOT, environment=environment, device=args.device),
        *source_data,
    )
    asset_report = run_preflight(requirements, environment=environment)
    if asset_report.status != "READY":
        codes = ", ".join(issue.code for issue in asset_report.issues)
        raise FairnessSearchError(f"training assets are not ready: {codes}")
    completed = subprocess.run(worker, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode:
        return int(completed.returncode)
    judge = compile_fairness_judge_command(
        trial,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        judge_model=args.judge_model.resolve(),
        device=args.device,
        device_id=args.device_id,
    )
    completed = subprocess.run(judge, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FairnessSearchError, OSError, ValueError) as exc:
        raise SystemExit(f"fairness search failed: {exc}") from exc
