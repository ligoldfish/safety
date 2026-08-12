from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from .schema import ExperimentCatalog, ExperimentCell, ExperimentPlan


class PlanError(ValueError):
    """Raised when an experiment plan is ambiguous or internally inconsistent."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_cell_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def build_main_table_plan(catalog: ExperimentCatalog, *, output_root: str) -> ExperimentPlan:
    cells: list[ExperimentCell] = []
    for pair in catalog.formal_pairs:
        for dataset in catalog.formal_datasets:
            for method in catalog.main_methods:
                axes = {"pair": pair, "dataset": dataset, "method": method}
                payload = {"experiment_id": "P0-01", "axes": axes, "overrides": {}}
                cell_id = canonical_cell_id(payload)
                output_dir = str(PurePosixPath(output_root) / "main_table" / pair / dataset / method)
                cells.append(
                    ExperimentCell(
                        cell_id=cell_id,
                        experiment_id="P0-01",
                        axes=axes,
                        overrides={},
                        output_dir=output_dir,
                    )
                )
    plan = ExperimentPlan(schema_version=catalog.schema_version, cells=tuple(cells))
    validate_plan(plan)
    return plan


def _find_cycle(cells: tuple[ExperimentCell, ...]) -> bool:
    edges = {cell.cell_id: tuple(cell.depends_on) for cell in cells}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in edges.get(node, ()):
            if visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in edges)


def validate_plan(plan: ExperimentPlan | Iterable[ExperimentCell]) -> ExperimentPlan:
    if isinstance(plan, ExperimentPlan):
        normalized = plan
    else:
        normalized = ExperimentPlan(schema_version=1, cells=tuple(plan))
    if not normalized.cells:
        raise PlanError("plan must contain at least one cell")
    ids = [cell.cell_id for cell in normalized.cells]
    if len(ids) != len(set(ids)):
        raise PlanError("duplicate cell id")
    outputs = [cell.output_dir for cell in normalized.cells]
    if len(outputs) != len(set(outputs)):
        raise PlanError("duplicate output directory")
    known = set(ids)
    for cell in normalized.cells:
        unknown = set(cell.depends_on) - known
        if unknown:
            raise PlanError(f"unknown dependencies for {cell.cell_id}: {sorted(unknown)}")
    if _find_cycle(normalized.cells):
        raise PlanError("dependency cycle detected")
    return normalized
