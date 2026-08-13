from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


FAIRNESS_METHODS = ("sft1", "random", "ours")
VALIDATION_SELECTED_DATASETS = frozenset({"wildjailbreak", "wildguardmix"})
GLOBAL_HYPERPARAMETERS = {
    "top_k": 5,
    "energy_threshold": 0.8,
    "rank_cap": 32,
    "layer_loss_weight": 0.25,
    "epochs": 3,
}
HISTORICAL_HYPERPARAMETERS = {
    "wildjailbreak": {
        "top_k": 3,
        "energy_threshold": 0.7,
        "rank_cap": 8,
        "layer_loss_weight": 0.25,
        "epochs": 5,
    },
    "wildguardmix": {
        "top_k": 7,
        "energy_threshold": 0.9,
        "rank_cap": 32,
        "layer_loss_weight": 0.5,
        "epochs": 3,
    },
}
_HYPERPARAMETER_KEYS = frozenset(GLOBAL_HYPERPARAMETERS)
_REQUIRED_FIELDS = {
    "trial_id",
    "dataset",
    "config",
    "method",
    "selection_split",
    "selected",
    "validation_metric",
}
_VALIDATION_EVIDENCE = frozenset(
    {"judge", "generations", "train_split", "validation_split", "training", "layer", "subspace"}
)
_VALIDATION_SELECTION_METRIC = "wildguard_refusal_minus_over_refusal"


class FairnessLedgerError(ValueError):
    pass


@dataclass(frozen=True)
class FairnessConfiguration:
    hyperparameters: dict[str, int | float]
    phase1_stage_extras: dict[str, list[str]]
    phasef_updates: dict[str, int | float]
    selected_trial_id: str | None
    search_ledger_sha256: str


def _finite_number(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise FairnessLedgerError(f"{label} must be a finite number")
    return float(value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_hyperparameters(value: object, method: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != _HYPERPARAMETER_KEYS:
        raise FairnessLedgerError(
            "selected trials need exactly top_k, energy_threshold, rank_cap, "
            "layer_loss_weight, and epochs"
        )
    top_k = value["top_k"]
    rank_cap = value["rank_cap"]
    epochs = value["epochs"]
    if type(top_k) is not int or not 1 <= top_k <= 128:
        raise FairnessLedgerError("top_k must be an integer in [1, 128]")
    if type(rank_cap) is not int or not 1 <= rank_cap <= 4096:
        raise FairnessLedgerError("rank_cap must be an integer in [1, 4096]")
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise FairnessLedgerError("epochs must be an integer in [1, 100]")
    energy_threshold = _finite_number(value["energy_threshold"], "energy_threshold")
    if not 0.0 < energy_threshold <= 1.0:
        raise FairnessLedgerError("energy_threshold must be in (0, 1]")
    layer_loss_weight = _finite_number(value["layer_loss_weight"], "layer_loss_weight")
    if not 0.0 <= layer_loss_weight <= 10.0:
        raise FairnessLedgerError("layer_loss_weight must be in [0, 10]")
    if method == "sft1" and layer_loss_weight != 0.0:
        raise FairnessLedgerError("sft1 selected trials must keep layer_loss_weight=0")
    if method in {"random", "ours"} and layer_loss_weight <= 0.0:
        raise FairnessLedgerError(f"{method} selected trials require positive layer_loss_weight")
    return {
        "top_k": top_k,
        "energy_threshold": energy_threshold,
        "rank_cap": rank_cap,
        "layer_loss_weight": layer_loss_weight,
        "epochs": epochs,
    }


def validate_search_ledger_rows(rows: Iterable[Mapping]) -> tuple[dict, ...]:
    normalized: list[dict] = []
    trial_ids: set[str] = set()
    groups: dict[tuple[str, str], list[dict]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or not _REQUIRED_FIELDS <= set(raw):
            raise FairnessLedgerError("search ledger lacks required trial provenance fields")
        row = dict(raw)
        trial_id = str(row["trial_id"]).strip()
        dataset = str(row["dataset"]).strip()
        config = str(row["config"]).strip()
        method = str(row["method"]).strip()
        if not trial_id or trial_id in trial_ids or not dataset:
            raise FairnessLedgerError("trial identities must be non-empty and unique")
        if config not in {"global", "validation_selected"}:
            raise FairnessLedgerError(f"unsupported fairness config: {config}")
        if config == "validation_selected" and dataset not in VALIDATION_SELECTED_DATASETS:
            raise FairnessLedgerError(
                "validation-selected tuning is limited to the two historical override corpora"
            )
        if method not in FAIRNESS_METHODS:
            raise FairnessLedgerError(f"unsupported fairness method: {method}")
        if str(row["selection_split"]) != "validation":
            raise FairnessLedgerError("search trials must use validation-only selection")
        _finite_number(row["validation_metric"], "validation_metric")
        if type(row["selected"]) is not bool:
            raise FairnessLedgerError("selected must be boolean")
        trial_ids.add(trial_id)
        row.update({"trial_id": trial_id, "dataset": dataset, "config": config, "method": method})
        normalized.append(row)
        groups.setdefault((dataset, config), []).append(row)
    if not normalized:
        raise FairnessLedgerError("search ledger is empty")
    for (dataset, config), group in groups.items():
        counts = {method: 0 for method in FAIRNESS_METHODS}
        selected = {method: [] for method in FAIRNESS_METHODS}
        for row in group:
            method = row["method"]
            counts[method] += 1
            if row["selected"]:
                selected[method].append(row)
        if any(count == 0 for count in counts.values()) or len(set(counts.values())) != 1:
            raise FairnessLedgerError(
                f"search budgets must be equal across {FAIRNESS_METHODS} for {dataset}/{config}: {counts}"
            )
        if config == "global":
            if any(count != 1 for count in counts.values()):
                raise FairnessLedgerError(
                    f"global requires exactly one fixed record per method for {dataset}"
                )
            if any(row["selected"] for row in group):
                raise FairnessLedgerError("global fixed records cannot be selected search trials")
            for row in group:
                method = row["method"]
                expected = dict(GLOBAL_HYPERPARAMETERS)
                if method == "sft1":
                    expected["layer_loss_weight"] = 0.0
                applied = _validated_hyperparameters(row.get("hyperparameters"), method)
                if applied != expected:
                    raise FairnessLedgerError(
                        f"global record for {dataset}/{method} differs from preregistered defaults"
                    )
                row["hyperparameters"] = applied
        if config == "validation_selected":
            for row in group:
                row["hyperparameters"] = _validated_hyperparameters(
                    row.get("hyperparameters"), row["method"]
                )
            if any(len(selected[method]) != 1 for method in FAIRNESS_METHODS):
                raise FairnessLedgerError(
                    f"validation_selected requires exactly one winner per method for {dataset}"
                )
            for method in FAIRNESS_METHODS:
                winner_metric = float(selected[method][0]["validation_metric"])
                best_metric = max(
                    float(row["validation_metric"])
                    for row in group
                    if row["method"] == method
                )
                if winner_metric != best_metric:
                    raise FairnessLedgerError(
                        f"selected trial is not the best validation result for {dataset}/{method}"
                    )
            search_spaces: dict[str, Counter] = {}
            for method in FAIRNESS_METHODS:
                signatures = []
                for row in group:
                    if row["method"] != method:
                        continue
                    values = dict(row["hyperparameters"])
                    values.pop("layer_loss_weight")
                    signatures.append(json.dumps(values, sort_keys=True, separators=(",", ":")))
                search_spaces[method] = Counter(signatures)
            if len({tuple(sorted(space.items())) for space in search_spaces.values()}) != 1:
                raise FairnessLedgerError(
                    f"validation search space differs across methods for {dataset}"
                )
            random_weights = sorted(
                row["hyperparameters"]["layer_loss_weight"]
                for row in group
                if row["method"] == "random"
            )
            ours_weights = sorted(
                row["hyperparameters"]["layer_loss_weight"]
                for row in group
                if row["method"] == "ours"
            )
            if random_weights != ours_weights:
                raise FairnessLedgerError(
                    f"validation layer-loss search space differs for random/ours on {dataset}"
                )
            full_spaces = {}
            for method in ("random", "ours"):
                full_spaces[method] = Counter(
                    json.dumps(row["hyperparameters"], sort_keys=True, separators=(",", ":"))
                    for row in group
                    if row["method"] == method
                )
            if full_spaces["random"] != full_spaces["ours"]:
                raise FairnessLedgerError(
                    f"validation joint search space differs for random/ours on {dataset}"
                )
    return tuple(normalized)


def verify_search_ledger_evidence(rows: Iterable[Mapping]) -> None:
    """Verify validation selection provenance against the exact current files.

    Global rows are preregistered constants and have no search evidence. Every
    validation-selected candidate must retain all artifacts needed to reproduce
    its WildGuard score; this check deliberately hashes files at preflight time.
    """
    validation_rows = [row for row in rows if row.get("config") == "validation_selected"]
    present_groups = {
        (str(row.get("dataset")), str(row.get("method"))) for row in validation_rows
    }
    for dataset, method in sorted(present_groups):
            group = [
                row for row in validation_rows
                if row.get("dataset") == dataset and row.get("method") == method
            ]
            if {row.get("candidate") for row in group} != {"global", "historical_override"} or len(group) != 2:
                raise FairnessLedgerError(
                    f"validation search must contain exact global/historical candidates for {dataset}/{method}"
                )
            for row in group:
                source = (
                    GLOBAL_HYPERPARAMETERS
                    if row["candidate"] == "global"
                    else HISTORICAL_HYPERPARAMETERS[dataset]
                )
                expected = dict(source)
                if method == "sft1":
                    expected["layer_loss_weight"] = 0.0
                if dict(row["hyperparameters"]) != expected:
                    raise FairnessLedgerError(
                        f"validation candidate differs from preregistered search for {dataset}/{method}"
                    )
    for row in validation_rows:
        trial_id = str(row.get("trial_id", ""))
        if row.get("selection_metric") != _VALIDATION_SELECTION_METRIC:
            raise FairnessLedgerError(
                f"validation trial {trial_id} must use {_VALIDATION_SELECTION_METRIC}"
            )
        epoch = row.get("validation_epoch")
        if type(epoch) is not int or epoch < 1:
            raise FairnessLedgerError(f"validation trial {trial_id} lacks a final epoch")
        harmful = _finite_number(
            row.get("validation_harmful_refusal"), "validation_harmful_refusal"
        )
        over_refusal = _finite_number(
            row.get("validation_over_refusal"), "validation_over_refusal"
        )
        metric = _finite_number(row.get("validation_metric"), "validation_metric")
        if not (0.0 <= harmful <= 1.0 and 0.0 <= over_refusal <= 1.0):
            raise FairnessLedgerError(f"validation rates are outside [0,1] for {trial_id}")
        if not math.isclose(metric, harmful - over_refusal, rel_tol=0.0, abs_tol=1e-12):
            raise FairnessLedgerError(f"validation metric is inconsistent for {trial_id}")
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != _VALIDATION_EVIDENCE:
            raise FairnessLedgerError(f"validation evidence is incomplete for {trial_id}")
        for label, descriptor in evidence.items():
            if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
                raise FairnessLedgerError(f"invalid {label} evidence for {trial_id}")
            path = Path(str(descriptor["path"]))
            expected = str(descriptor["sha256"]).lower()
            if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
                raise FairnessLedgerError(f"invalid {label} hash for {trial_id}")
            try:
                actual = _sha256_path(path)
            except OSError as exc:
                raise FairnessLedgerError(
                    f"validation evidence is missing for {trial_id}: {path}"
                ) from exc
            if actual != expected:
                raise FairnessLedgerError(
                    f"validation evidence changed for {trial_id}: {path}"
                )
        try:
            layer = json.loads(Path(evidence["layer"]["path"]).read_text(encoding="utf-8"))
            subspace = json.loads(Path(evidence["subspace"]["path"]).read_text(encoding="utf-8"))
            training = json.loads(Path(evidence["training"]["path"]).read_text(encoding="utf-8"))
            judge = json.loads(Path(evidence["judge"]["path"]).read_text(encoding="utf-8"))
            generations = json.loads(
                Path(evidence["generations"]["path"]).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FairnessLedgerError(f"invalid backend evidence for {trial_id}") from exc
        if not all(isinstance(item, Mapping) for item in (layer, subspace, training, judge, generations)):
            raise FairnessLedgerError(f"invalid backend evidence for {trial_id}")
        values = row["hyperparameters"]
        expected_mode = "random_same_norm" if row["method"] == "random" else "semantic"
        try:
            energy = _finite_number(subspace.get("energy_threshold"), "evidence energy_threshold")
            layer_weight = _finite_number(training.get("layer_loss_weight"), "evidence layer_loss_weight")
        except FairnessLedgerError as exc:
            raise FairnessLedgerError(f"invalid backend evidence for {trial_id}") from exc
        if (
            layer.get("top_k") != values["top_k"]
            or subspace.get("rank_cap") != values["rank_cap"]
            or training.get("epochs") != values["epochs"]
            or training.get("epochs_completed") != values["epochs"]
            or training.get("target_mode") != expected_mode
            or Path(str(training.get("train_split", ""))).resolve()
            != Path(str(evidence["train_split"]["path"])).resolve()
            or Path(str(training.get("val_split", ""))).resolve()
            != Path(str(evidence["validation_split"]["path"])).resolve()
            or not math.isclose(energy, float(values["energy_threshold"]), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(layer_weight, float(values["layer_loss_weight"]), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise FairnessLedgerError(f"backend evidence differs from ledger for {trial_id}")
        generated = generations.get("generations")
        if not isinstance(generated, list) or not generated:
            raise FairnessLedgerError(f"backend evidence has no validation generations for {trial_id}")
        generation_ids = [str(item.get("id", "")) for item in generated if isinstance(item, Mapping)]
        if len(generation_ids) != len(generated) or any(not value for value in generation_ids) or len(set(generation_ids)) != len(generation_ids):
            raise FairnessLedgerError(f"backend evidence has invalid generation IDs for {trial_id}")
        try:
            validation_records = [
                json.loads(line)
                for line in Path(evidence["validation_split"]["path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            train_records = [
                json.loads(line)
                for line in Path(evidence["train_split"]["path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            raise FairnessLedgerError(f"invalid validation split evidence for {trial_id}") from exc
        if not validation_records or not train_records or not all(
            isinstance(item, Mapping) for item in [*validation_records, *train_records]
        ):
            raise FairnessLedgerError(f"invalid train/validation split evidence for {trial_id}")
        for label, records in (("train", train_records), ("validation", validation_records)):
            identifiers = [str(item.get("id", "")).strip() for item in records]
            labels = {str(item.get("label", "")).strip().lower() for item in records}
            if (
                any(not identifier for identifier in identifiers)
                or len(identifiers) != len(set(identifiers))
                or labels != {"harmful", "harmless"}
            ):
                raise FairnessLedgerError(f"invalid {label} split schema for {trial_id}")
        split_labels = {
            str(item.get("id", "")): str(item.get("label", "")).strip().lower()
            for item in validation_records
        }
        generation_labels = {
            str(item.get("id", "")): str(item.get("label", "")).strip().lower()
            for item in generated
        }
        split_ids = set(split_labels)
        if "" in split_ids or set(generation_ids) != split_ids or generation_labels != split_labels:
            raise FairnessLedgerError(f"validation generation IDs differ from split for {trial_id}")
        from .data_audit import audit_train_eval_splits

        split_audit = audit_train_eval_splits(train_records, validation_records)
        if any(
            split_audit[key]
            for key in ("overlap_count", "train_duplicate_count", "eval_duplicate_count")
        ):
            raise FairnessLedgerError(f"train/validation split leakage for {trial_id}")
        judged = judge.get("generations")
        judged_ids = [str(item.get("id", "")) for item in judged] if isinstance(judged, list) and all(isinstance(item, Mapping) for item in judged) else []
        judged_labels = {
            str(item.get("id", "")): str(item.get("label", "")).strip().lower()
            for item in judged
        } if judged_ids else {}
        try:
            parse_rate = _finite_number(judge.get("judge_parse_rate"), "judge_parse_rate")
            judge_harmful = _finite_number(judge.get("llm_judge_refusal_rate"), "judge harmful refusal")
            judge_over = _finite_number(judge.get("llm_judge_over_refusal"), "judge over-refusal")
        except FairnessLedgerError as exc:
            raise FairnessLedgerError(f"invalid WildGuard evidence for {trial_id}") from exc
        if (
            judge.get("judge") != "wildguard"
            or type(judge.get("num_unmatched_ids")) is not int
            or judge["num_unmatched_ids"] != 0
            or not math.isclose(parse_rate, 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(judge_harmful, harmful, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(judge_over, over_refusal, rel_tol=0.0, abs_tol=1e-12)
            or type(judge.get("num_generations")) is not int
            or judge["num_generations"] != len(generated)
            or Path(str(judge.get("pan_results", ""))).resolve()
            != Path(str(evidence["generations"]["path"])).resolve()
            or set(judged_ids) != set(generation_ids)
            or len(judged_ids) != len(generation_ids)
            or judged_labels != generation_labels
            or type(judge.get("judge_num_harmful_scored")) is not int
            or judge["judge_num_harmful_scored"] <= 0
            or type(judge.get("judge_num_harmless_scored")) is not int
            or judge["judge_num_harmless_scored"] <= 0
        ):
            raise FairnessLedgerError(f"WildGuard evidence differs from ledger for {trial_id}")


def load_search_ledger_snapshot(path: Path) -> tuple[tuple[dict, ...], str]:
    try:
        payload = Path(path).read_bytes()
        rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FairnessLedgerError(f"missing or invalid search ledger: {path}") from exc
    normalized = validate_search_ledger_rows(rows)
    verify_search_ledger_evidence(normalized)
    return normalized, hashlib.sha256(payload).hexdigest()


def load_search_ledger(path: Path) -> tuple[dict, ...]:
    rows, _ = load_search_ledger_snapshot(path)
    return rows


def _configuration(
    hyperparameters: Mapping[str, int | float],
    *,
    selected_trial_id: str | None,
    search_ledger_sha256: str,
) -> FairnessConfiguration:
    values = dict(hyperparameters)
    return FairnessConfiguration(
        hyperparameters=values,
        phase1_stage_extras={
            "analyze": ["--top-k", str(values["top_k"])],
            "subspace": [
                "--energy-threshold",
                str(values["energy_threshold"]),
                "--rank-cap",
                str(values["rank_cap"]),
            ],
        },
        phasef_updates={
            "optim.layer_loss_weight": values["layer_loss_weight"],
            "optim.epochs": values["epochs"],
        },
        selected_trial_id=selected_trial_id,
        search_ledger_sha256=search_ledger_sha256,
    )


def resolve_fairness_configuration(cell_spec: Mapping) -> FairnessConfiguration:
    if str(cell_spec.get("experiment_id", "")) != "P0-07":
        raise FairnessLedgerError("fairness configuration is only valid for P0-07")
    axes = dict(cell_spec.get("axes") or {})
    dataset = str(axes.get("dataset", "")).strip()
    config = str(axes.get("config", "")).strip()
    method = str(axes.get("method", "")).strip()
    if not dataset or method not in FAIRNESS_METHODS:
        raise FairnessLedgerError("P0-07 requires dataset and method axes")
    source = str((cell_spec.get("inputs") or {}).get("search_ledger", "")).strip()
    if not source:
        raise FairnessLedgerError("P0-07 requires search_ledger input")
    source_path = Path(source)
    all_rows, ledger_hash = load_search_ledger_snapshot(source_path)
    if config == "global":
        matching = [
            row
            for row in all_rows
            if row["dataset"] == dataset
            and row["config"] == config
            and row["method"] == method
        ]
        if len(matching) != 1:
            raise FairnessLedgerError(
                f"search ledger lacks one global fixed record for {dataset}/{method}"
            )
        values = dict(GLOBAL_HYPERPARAMETERS)
        if method == "sft1":
            values["layer_loss_weight"] = 0.0
        return _configuration(
            values,
            selected_trial_id=None,
            search_ledger_sha256=ledger_hash,
        )
    if config != "validation_selected":
        raise FairnessLedgerError(f"unsupported fairness config: {config}")
    rows = [
        row
        for row in all_rows
        if row["dataset"] == dataset
        and row["config"] == config
        and row["method"] == method
        and row["selected"]
    ]
    if len(rows) != 1:
        raise FairnessLedgerError(
            f"search ledger lacks one selected trial for {dataset}/{config}/{method}"
        )
    row = rows[0]
    return _configuration(
        row["hyperparameters"],
        selected_trial_id=str(row["trial_id"]),
        search_ledger_sha256=ledger_hash,
    )
