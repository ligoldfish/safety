from __future__ import annotations

import hashlib
import itertools
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


def _expand_axes(axes: Mapping[str, tuple[Any, ...]]) -> Iterable[dict[str, Any]]:
    if not axes:
        yield {}
        return
    keys = tuple(axes)
    for values in itertools.product(*(axes[key] for key in keys)):
        yield dict(zip(keys, values))


def _valid_declared_cell(experiment_id: str, axes: Mapping[str, Any]) -> bool:
    """Apply the correlated-axis constraints stated in the experiment design."""

    if experiment_id == "P0-07":
        # The preregistered fixed policy runs on every formal corpus. Only the
        # two corpora with historical per-dataset overrides receive the
        # validation-selected comparison; expanding this to the other four
        # would add unplanned tuning runs and change the fairness budget.
        return (
            str(axes.get("config")) == "global"
            or str(axes.get("dataset")) in {"wildjailbreak", "wildguardmix"}
        )
    if experiment_id == "P1-05":
        # selected/evenly/last are deterministic single cells; only random-K
        # has the five independent draws required by the design.
        return str(axes.get("mode")) == "random_k" or int(axes.get("draw", 0)) == 0
    if experiment_id == "P1-15":
        # Two one-dimensional sensitivity curves: tau @ cap=32 and cap @ tau=.8.
        return int(axes.get("rank_cap")) == 32 or float(axes.get("energy_threshold")) == 0.8
    return True


def build_catalog_plan(
    catalog: ExperimentCatalog,
    *,
    output_root: str,
    scope: str = "all",
) -> ExperimentPlan:
    """Expand a stable, immutable plan without loading data or models.

    ``P0-01`` is the exact 5 x 6 x 5 main-table provenance matrix. Every
    other experiment expands its declared axes as a Cartesian product. This
    intentionally makes expensive work visible before submission rather than
    hiding implicit loops in a launcher.
    """

    normalized_scope = str(scope).strip().lower().replace("_", "-")
    if normalized_scope in {"main", "main-table"}:
        return build_main_table_plan(catalog, output_root=output_root)
    if normalized_scope not in {"all", "p0", "p1", "p2"}:
        raise PlanError(f"unknown plan scope: {scope}")

    cells: list[ExperimentCell] = []
    main_cells = build_main_table_plan(catalog, output_root=output_root).cells
    selected_ids = {
        experiment_id
        for experiment_id in catalog.experiments
        if normalized_scope == "all" or experiment_id.startswith(normalized_scope.upper() + "-")
    }
    if "P0-01" in selected_ids:
        cells.extend(main_cells)
    for experiment_id in sorted(selected_ids):
        if experiment_id == "P0-01":
            continue
        definition = catalog.experiments[experiment_id]
        for axes in _expand_axes(definition.axes):
            if not _valid_declared_cell(experiment_id, axes):
                continue
            overrides = dict(definition.overrides)
            payload = {
                "experiment_id": experiment_id,
                "axes": axes,
                "overrides": overrides,
            }
            cell_id = canonical_cell_id(payload)
            output_dir = str(
                PurePosixPath(output_root)
                / "ablations"
                / experiment_id.lower()
                / cell_id
            )
            cells.append(
                ExperimentCell(
                    cell_id=cell_id,
                    experiment_id=experiment_id,
                    axes=axes,
                    overrides=overrides,
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
