from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Phase1ModelConfig:
    name: str
    path: str
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    chat_template_enable_thinking: bool = True
    runtime_backend: str = ""
    runtime_device: str = ""
    trust_remote_code: bool = True
    local_files_only: bool = True
    attn_implementation: str = ""


@dataclass
class Phase1DataConfig:
    pan_repo_dir: str
    raw_dir: str
    processed_dir: str
    metadata_dir: str
    exposure_size: int = 60
    pan_test_size_per_type: int = 60
    pan_train_size: int = 2600
    alignment_size: int = 2100
    analysis_val_size: int = 500
    sanity_test_size_per_label: int = 200
    max_prompt_chars: int = 2048
    system_prompt: str = "You are a helpful assistant."
    include_system_prompt: bool = True


@dataclass
class Phase1ExtractionConfig:
    output_root: str
    max_length: int = 4096
    batch_size: int = 4
    shard_size: int = 128
    storage_dtype: str = "float16"
    skip_existing: bool = True


@dataclass
class Phase1Config:
    seed: int
    dataset: Phase1DataConfig
    teacher: Phase1ModelConfig
    student: Phase1ModelConfig
    extraction: Phase1ExtractionConfig


@dataclass
class PhaseBInputConfig:
    hidden_root: str
    train_dir: str
    val_dir: str
    test_dir: str = ""


@dataclass
class PhaseBMethodConfig:
    rank: int = 1
    target_label: str = "harmless"
    reference_label: str = "harmful"
    selection_metric: str = "balanced_accuracy"
    selected_layers: List[int] = field(default_factory=list)


@dataclass
class PhaseBDataLimitConfig:
    train_max_samples_per_label: int = 0
    val_max_samples_per_label: int = 0
    test_max_samples_per_label: int = 0


@dataclass
class PhaseBOutputConfig:
    output_root: str


@dataclass
class PhaseBConfig:
    seed: int
    inputs: PhaseBInputConfig
    method: PhaseBMethodConfig
    limits: PhaseBDataLimitConfig
    output: PhaseBOutputConfig


@dataclass
class PhaseCInputConfig:
    artifact_path: str
    val_split: str
    test_split: str


@dataclass
class PhaseCMethodConfig:
    alphas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    selection_metric: str = "balanced_accuracy"
    max_length: int = 4096
    batch_size: int = 1
    intervention_position: str = "last_non_padding"


@dataclass
class PhaseCDataLimitConfig:
    val_max_samples_per_label: int = 0
    test_max_samples_per_label: int = 0


@dataclass
class PhaseCOutputConfig:
    output_root: str


@dataclass
class PhaseCConfig:
    seed: int
    model: Phase1ModelConfig
    inputs: PhaseCInputConfig
    method: PhaseCMethodConfig
    limits: PhaseCDataLimitConfig
    output: PhaseCOutputConfig


@dataclass
class PhaseFInputConfig:
    train_split: str
    val_split: str
    train_targets_dir: str
    val_targets_dir: str
    pairing_path: str


@dataclass
class PhaseFLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: ["self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]
    )


@dataclass
class PhaseFOptimConfig:
    epochs: int = 3
    batch_size: int = 16
    micro_batch_size: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    layer_loss_weight: float = 0.5
    max_length: int = 1024
    max_new_tokens: int = 64
    log_every_steps: int = 1


@dataclass
class PhaseFTargetConfig:
    mode: str = "semantic"
    random_seed: int = 2042
    match_l2_norm: bool = True


@dataclass
class PhaseFOutputConfig:
    output_root: str


@dataclass
class PhaseFConfig:
    seed: int
    model: Phase1ModelConfig
    inputs: PhaseFInputConfig
    lora: PhaseFLoRAConfig
    optim: PhaseFOptimConfig
    target: PhaseFTargetConfig
    output: PhaseFOutputConfig


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return payload


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    if "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _to_model_config(raw: Dict[str, Any], base_dir: Path) -> Phase1ModelConfig:
    data = dict(raw)
    data["path"] = _resolve_path(str(data["path"]), base_dir)
    return Phase1ModelConfig(**data)


def _resolve_path_in_mapping(raw: Dict[str, Any], base_dir: Path, keys: List[str]) -> Dict[str, Any]:
    data = dict(raw)
    for key in keys:
        if key in data:
            data[key] = _resolve_path(str(data[key]), base_dir)
    return data


def load_phase1_config(path: str | Path) -> Phase1Config:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    dataset = dict(raw["dataset"])
    extraction = dict(raw["extraction"])

    for key in ["pan_repo_dir", "raw_dir", "processed_dir", "metadata_dir"]:
        dataset[key] = _resolve_path(str(dataset[key]), base_dir)
    extraction["output_root"] = _resolve_path(str(extraction["output_root"]), base_dir)

    return Phase1Config(
        seed=int(raw.get("seed", 42)),
        dataset=Phase1DataConfig(**dataset),
        teacher=_to_model_config(dict(raw["models"]["teacher"]), base_dir),
        student=_to_model_config(dict(raw["models"]["student"]), base_dir),
        extraction=Phase1ExtractionConfig(**extraction),
    )


def load_phaseb_config(path: str | Path) -> PhaseBConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    inputs = _resolve_path_in_mapping(
        dict(raw["inputs"]),
        base_dir,
        ["hidden_root"],
    )
    output = _resolve_path_in_mapping(
        dict(raw["output"]),
        base_dir,
        ["output_root"],
    )
    method = dict(raw.get("method", {}))
    limits = dict(raw.get("limits", {}))

    return PhaseBConfig(
        seed=int(raw.get("seed", 42)),
        inputs=PhaseBInputConfig(**inputs),
        method=PhaseBMethodConfig(**method),
        limits=PhaseBDataLimitConfig(**limits),
        output=PhaseBOutputConfig(**output),
    )


def load_phasec_config(path: str | Path) -> PhaseCConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    model = _to_model_config(dict(raw["model"]), base_dir)
    inputs = _resolve_path_in_mapping(
        dict(raw["inputs"]),
        base_dir,
        ["artifact_path", "val_split", "test_split"],
    )
    output = _resolve_path_in_mapping(
        dict(raw["output"]),
        base_dir,
        ["output_root"],
    )
    method = dict(raw.get("method", {}))
    limits = dict(raw.get("limits", {}))

    return PhaseCConfig(
        seed=int(raw.get("seed", 42)),
        model=model,
        inputs=PhaseCInputConfig(**inputs),
        method=PhaseCMethodConfig(**method),
        limits=PhaseCDataLimitConfig(**limits),
        output=PhaseCOutputConfig(**output),
    )


def load_phasef_config(path: str | Path) -> PhaseFConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    model = _to_model_config(dict(raw["model"]), base_dir)
    inputs = _resolve_path_in_mapping(
        dict(raw["inputs"]),
        base_dir,
        ["train_split", "val_split", "train_targets_dir", "val_targets_dir", "pairing_path"],
    )
    output = _resolve_path_in_mapping(
        dict(raw["output"]),
        base_dir,
        ["output_root"],
    )
    lora = dict(raw.get("lora", {}))
    optim = dict(raw.get("optim", {}))
    target = dict(raw.get("target", {}))

    return PhaseFConfig(
        seed=int(raw.get("seed", 42)),
        model=model,
        inputs=PhaseFInputConfig(**inputs),
        lora=PhaseFLoRAConfig(**lora),
        optim=PhaseFOptimConfig(**optim),
        target=PhaseFTargetConfig(**target),
        output=PhaseFOutputConfig(**output),
    )
