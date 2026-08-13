from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ablations.fairness import GLOBAL_HYPERPARAMETERS, HISTORICAL_HYPERPARAMETERS


EVIDENCE_NAMES = ("judge", "generations", "train_split", "validation_split", "training", "layer", "subspace")


def attach_validation_evidence(rows: list[dict], root: Path) -> list[dict]:
    """Materialize minimal immutable files for tests that exercise ledger loading."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("config") == "validation_selected":
            groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    for (dataset, method), group in groups.items():
        if len(group) == 1:
            selected = group[0]
            global_row = {
                **selected,
                "trial_id": str(selected["trial_id"]) + "-global-control",
                "selected": False,
                "validation_metric": float(selected["validation_metric"]) - 0.1,
            }
            rows.append(global_row)
            group.append(global_row)
        if len(group) != 2:
            raise ValueError("test evidence helper expects one or two candidates per method")
        if sum(bool(row["selected"]) for row in group) != 1:
            raise ValueError("test evidence helper expects exactly one winner per method")
        for row in group:
            candidate = "historical_override" if row["selected"] else "global"
            source = (
                HISTORICAL_HYPERPARAMETERS[dataset]
                if candidate == "historical_override"
                else GLOBAL_HYPERPARAMETERS
            )
            row["candidate"] = candidate
            row["hyperparameters"] = dict(source)
            if method == "sft1":
                row["hyperparameters"]["layer_loss_weight"] = 0.0
    for row in rows:
        if row.get("config") != "validation_selected":
            continue
        metric = float(row["validation_metric"])
        if not -1.0 <= metric <= 1.0:
            raise ValueError("test validation metric must be a rate difference")
        over_refusal = max(0.0, -metric)
        harmful_refusal = metric + over_refusal
        evidence = {}
        validation_path = root / "evidence" / str(row["trial_id"]) / "validation_split.jsonl"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_payload = (
            b'{"id":"x","label":"harmful","messages":[{"role":"user","content":"validation harmful"}]}\n'
            b'{"id":"y","label":"harmless","messages":[{"role":"user","content":"validation harmless"}]}\n'
        )
        validation_path.write_bytes(validation_payload)
        train_path = root / "evidence" / str(row["trial_id"]) / "train_split.jsonl"
        train_payload = (
            b'{"id":"t","label":"harmful","messages":[{"role":"user","content":"training harmful"}]}\n'
            b'{"id":"u","label":"harmless","messages":[{"role":"user","content":"training harmless"}]}\n'
        )
        train_path.write_bytes(train_payload)
        contents = {
            "layer": {"top_k": row["hyperparameters"]["top_k"]},
            "subspace": {
                "rank_cap": row["hyperparameters"]["rank_cap"],
                "energy_threshold": row["hyperparameters"]["energy_threshold"],
            },
            "training": {
                "epochs": row["hyperparameters"]["epochs"],
                "epochs_completed": row["hyperparameters"]["epochs"],
                "layer_loss_weight": row["hyperparameters"]["layer_loss_weight"],
                "target_mode": "random_same_norm" if row["method"] == "random" else "semantic",
                "train_split": str(train_path.resolve()),
                "val_split": str(validation_path.resolve()),
            },
            "judge": {
                "judge": "wildguard", "num_unmatched_ids": 0, "judge_parse_rate": 1.0,
                "num_generations": 2, "llm_judge_refusal_rate": harmful_refusal,
                "llm_judge_over_refusal": over_refusal,
                "judge_num_harmful_scored": 1, "judge_num_harmless_scored": 1,
                "generations": [
                    {"id": "x", "label": "harmful"}, {"id": "y", "label": "harmless"}
                ],
                "pan_results": str((root / "evidence" / str(row["trial_id"]) / "generations.json").resolve()),
            },
            "generations": {"generations": [
                {"id": "x", "label": "harmful"}, {"id": "y", "label": "harmless"}
            ]},
        }
        for name in EVIDENCE_NAMES:
            path = validation_path if name == "validation_split" else train_path if name == "train_split" else (
                root / "evidence" / str(row["trial_id"]) / f"{name}.json"
            )
            payload = validation_payload if name == "validation_split" else train_payload if name == "train_split" else (
                json.dumps(contents[name], sort_keys=True) + "\n"
            ).encode("utf-8")
            path.write_bytes(payload)
            evidence[name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        row.update(
            {
                "selection_metric": "wildguard_refusal_minus_over_refusal",
                "validation_epoch": int(row["hyperparameters"]["epochs"]),
                "validation_harmful_refusal": harmful_refusal,
                "validation_over_refusal": over_refusal,
                "evidence": evidence,
            }
        )
    return rows
