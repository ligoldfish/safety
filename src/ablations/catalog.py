from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .schema import (
    ExecutionKind,
    ExperimentCatalog,
    ExperimentDefinition,
    Priority,
)


class CatalogError(ValueError):
    """Raised when a declarative experiment catalog is invalid."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "formal_pairs",
    "formal_datasets",
    "main_methods",
    "strategies",
    "experiments",
}
_EXPERIMENT_KEYS = {
    "id",
    "priority",
    "family",
    "question",
    "execution_kind",
    "handler",
    "axes",
    "overrides",
    "requires",
    "metrics",
    "completion_artifacts",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogError(f"{label} must be a mapping")
    return value


def _tuple_of_strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CatalogError(f"{label} must be a list")
    result = tuple(str(item).strip() for item in value)
    if not allow_empty and (not result or any(not item for item in result)):
        raise CatalogError(f"{label} must contain non-empty values")
    if len(set(result)) != len(result):
        raise CatalogError(f"{label} contains duplicate values")
    return result


def _validate_strategy_overrides(
    overrides: Mapping[str, Any], strategies: Mapping[str, tuple[str, ...]]
) -> None:
    for key, value in overrides.items():
        if key.endswith(".mode") or key == "loss.kind":
            if key not in strategies:
                raise CatalogError(f"unknown strategy key: {key}")
            if str(value) not in strategies[key]:
                raise CatalogError(f"unknown {key}: {value}")


def _parse_experiment(
    raw: Any, strategies: Mapping[str, tuple[str, ...]]
) -> ExperimentDefinition:
    item = _mapping(raw, "experiment")
    unknown = set(item) - _EXPERIMENT_KEYS
    missing = {"id", "priority", "family", "question", "execution_kind", "handler"} - set(item)
    if unknown:
        raise CatalogError(f"unknown experiment fields: {sorted(unknown)}")
    if missing:
        raise CatalogError(f"missing experiment fields: {sorted(missing)}")

    axes_raw = _mapping(item.get("axes", {}), f"{item['id']}.axes")
    axes: dict[str, tuple[Any, ...]] = {}
    for key, value in axes_raw.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise CatalogError(f"{item['id']}.axes.{key} must be a non-empty list")
        values = tuple(value)
        if len({repr(entry) for entry in values}) != len(values):
            raise CatalogError(f"{item['id']}.axes.{key} contains duplicate values")
        axes[str(key)] = values

    overrides = dict(_mapping(item.get("overrides", {}), f"{item['id']}.overrides"))
    _validate_strategy_overrides(overrides, strategies)
    try:
        priority = Priority(str(item["priority"]))
        execution_kind = ExecutionKind(str(item["execution_kind"]))
    except ValueError as exc:
        raise CatalogError(str(exc)) from exc

    return ExperimentDefinition(
        id=str(item["id"]).strip(),
        priority=priority,
        family=str(item["family"]).strip(),
        question=str(item["question"]).strip(),
        execution_kind=execution_kind,
        handler=str(item["handler"]).strip(),
        axes=axes,
        overrides=overrides,
        requires=_tuple_of_strings(item.get("requires", []), f"{item['id']}.requires", allow_empty=True),
        metrics=_tuple_of_strings(item.get("metrics", []), f"{item['id']}.metrics"),
        completion_artifacts=_tuple_of_strings(
            item.get("completion_artifacts", []), f"{item['id']}.completion_artifacts"
        ),
    )


def load_catalog(path: str | Path) -> ExperimentCatalog:
    source = Path(path)
    raw = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "catalog")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise CatalogError(f"unknown catalog fields: {sorted(unknown)}")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] <= 0:
        raise CatalogError("schema_version must be a positive integer")

    strategies_raw = _mapping(raw.get("strategies", {}), "strategies")
    strategies = {
        str(key): _tuple_of_strings(value, f"strategies.{key}")
        for key, value in strategies_raw.items()
    }
    experiments: dict[str, ExperimentDefinition] = {}
    raw_experiments = raw.get("experiments")
    if not isinstance(raw_experiments, Sequence) or isinstance(raw_experiments, (str, bytes)):
        raise CatalogError("experiments must be a list")
    for raw_item in raw_experiments:
        item = _parse_experiment(raw_item, strategies)
        if item.id in experiments:
            raise CatalogError(f"duplicate experiment id: {item.id}")
        experiments[item.id] = item

    return ExperimentCatalog(
        schema_version=raw["schema_version"],
        formal_pairs=_tuple_of_strings(raw.get("formal_pairs"), "formal_pairs"),
        formal_datasets=_tuple_of_strings(raw.get("formal_datasets"), "formal_datasets"),
        main_methods=_tuple_of_strings(raw.get("main_methods"), "main_methods"),
        strategies=strategies,
        experiments=experiments,
    )
