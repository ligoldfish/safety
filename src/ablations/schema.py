from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple


def _experiment_ids(prefix: str, stop: int) -> set[str]:
    return {f"{prefix}-{index:02d}" for index in range(1, stop + 1)}


EXPECTED_EXPERIMENT_IDS = frozenset(
    _experiment_ids("P0", 8) | _experiment_ids("P1", 20) | _experiment_ids("P2", 7)
)


class ExecutionKind(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    MANUAL = "manual"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class CampaignTier(str, Enum):
    FULL = "full"
    EXTENDED = "extended"


@dataclass(frozen=True)
class ExperimentDefinition:
    id: str
    priority: Priority
    family: str
    question: str
    execution_kind: ExecutionKind
    handler: str
    campaign_tier: CampaignTier = CampaignTier.FULL
    axes: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)
    overrides: Mapping[str, Any] = field(default_factory=dict)
    requires: Tuple[str, ...] = ()
    metrics: Tuple[str, ...] = ()
    completion_artifacts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentCatalog:
    schema_version: int
    formal_pairs: Tuple[str, ...]
    formal_datasets: Tuple[str, ...]
    main_methods: Tuple[str, ...]
    strategies: Mapping[str, Tuple[str, ...]]
    experiments: Mapping[str, ExperimentDefinition]


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    experiment_id: str
    axes: Mapping[str, Any]
    overrides: Mapping[str, Any]
    output_dir: str
    depends_on: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentPlan:
    schema_version: int
    cells: Tuple[ExperimentCell, ...]
