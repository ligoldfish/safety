#!/usr/bin/env python3
"""Emit the exact formal LLM-judge target matrix used by reports and backfill."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_llm_judge_results import (  # noqa: E402
    CORE_METRICS,
    FORMAL_PAIRS,
    ResultSpec,
    iter_result_specs,
)


@dataclass(frozen=True)
class FormalJudgeTarget:
    pair_id: str
    dataset: str
    method: str
    epoch: str
    kind: str
    owner_path: Path
    pan_results_path: Path
    run_root: Path

    def identity(self) -> dict[str, str]:
        return {
            "model_pair": self.pair_id,
            "dataset": self.dataset,
            "method": self.method,
            "epoch": self.epoch,
        }


def target_from_spec(spec: ResultSpec) -> FormalJudgeTarget:
    pan_results_path = spec.result_dir / "pan_results.json"
    if spec.epoch == "single":
        return FormalJudgeTarget(
            pair_id=spec.pair_id,
            dataset=spec.dataset,
            method=spec.method,
            epoch=spec.epoch,
            kind="single",
            owner_path=pan_results_path,
            pan_results_path=pan_results_path,
            run_root=spec.result_dir,
        )
    eval_suite = spec.result_dir.parent
    if eval_suite.name != "eval_suite":
        raise ValueError(f"trained result is not under eval_suite: {spec.result_dir}")
    return FormalJudgeTarget(
        pair_id=spec.pair_id,
        dataset=spec.dataset,
        method=spec.method,
        epoch=spec.epoch,
        kind="suite",
        owner_path=eval_suite,
        pan_results_path=pan_results_path,
        run_root=eval_suite.parent,
    )


def iter_formal_targets(
    outputs_root: Path,
    pair_ids: Sequence[str] = FORMAL_PAIRS,
) -> list[FormalJudgeTarget]:
    return [target_from_spec(spec) for spec in iter_result_specs(outputs_root, pair_ids)]


def judge_payload_is_complete(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for key in CORE_METRICS:
        value = payload.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return False

    num_items = payload.get("judge_num_items")
    if num_items is not None and (
        not isinstance(num_items, int) or isinstance(num_items, bool) or num_items <= 0
    ):
        return False
    parse_rate = payload.get("judge_parse_rate")
    if parse_rate is not None and (
        not isinstance(parse_rate, (int, float))
        or isinstance(parse_rate, bool)
        or not math.isfinite(parse_rate)
        or parse_rate <= 0
        or parse_rate > 1
    ):
        return False
    scored = (
        payload.get("judge_num_harmful_scored"),
        payload.get("judge_num_harmless_scored"),
    )
    if all(value is not None for value in scored):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in scored
        ):
            return False
        if sum(scored) <= 0:
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args(argv)

    fields = (
        "pair_id",
        "dataset",
        "method",
        "epoch",
        "kind",
        "owner_path",
        "pan_results_path",
        "run_root",
    )
    if not args.no_header:
        print("\t".join(fields))
    for target in iter_formal_targets(args.outputs_root):
        print("\t".join((
            target.pair_id,
            target.dataset,
            target.method,
            target.epoch,
            target.kind,
            target.owner_path.as_posix(),
            target.pan_results_path.as_posix(),
            target.run_root.as_posix(),
        )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
