from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class BaselineModelConfig:
    name: str
    path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    chat_template_enable_thinking: bool = False
    runtime_backend: str = ""
    runtime_device: str = ""
    trust_remote_code: bool = True
    local_files_only: bool = True
    attn_implementation: str = ""


@dataclass
class AdapterConfig:
    manifest_path: str = ""
    checkpoint_path: str = ""


@dataclass
class EvalTaskConfig:
    path: str = ""
    split: str = "test"
    max_samples: int = 0
    shuffle: bool = False
    enabled: bool = True
    placeholder_ok: bool = True
    max_new_tokens: int = 128
    initial_max_new_tokens: int = 0
    exec_timeout_seconds: int = 3


@dataclass
class EvalDatasetsConfig:
    pan: EvalTaskConfig = field(default_factory=EvalTaskConfig)
    mmlu: EvalTaskConfig = field(default_factory=EvalTaskConfig)
    gsm8k: EvalTaskConfig = field(default_factory=EvalTaskConfig)
    humaneval: EvalTaskConfig = field(default_factory=EvalTaskConfig)
    mbpp: EvalTaskConfig = field(default_factory=EvalTaskConfig)


@dataclass
class EvalRuntimeConfig:
    max_length: int = 1024
    batch_size: int = 1


@dataclass
class EvalOutputConfig:
    output_root: str


@dataclass
class BaselineEvalConfig:
    seed: int
    model: BaselineModelConfig
    adapter: AdapterConfig
    datasets: EvalDatasetsConfig
    runtime: EvalRuntimeConfig
    output: EvalOutputConfig


@dataclass
class SupervisedDataConfig:
    train_split: str
    val_split: str
    test_split: str = ""


@dataclass
class BaselineLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    )


@dataclass
class BaselineOptimConfig:
    epochs: int = 3
    batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_length: int = 1024
    max_new_tokens: int = 64
    log_every_steps: int = 1


@dataclass
class BaselineOutputConfig:
    output_root: str


@dataclass
class BaselineSFTConfig:
    seed: int
    model: BaselineModelConfig
    data: SupervisedDataConfig
    lora: BaselineLoRAConfig
    optim: BaselineOptimConfig
    output: BaselineOutputConfig


@dataclass
class DistillLossConfig:
    temperature: float = 2.0
    hard_loss_weight: float = 0.5
    soft_loss_weight: float = 0.5


@dataclass
class BaselineDistillConfig:
    seed: int
    teacher: BaselineModelConfig
    student: BaselineModelConfig
    data: SupervisedDataConfig
    lora: BaselineLoRAConfig
    optim: BaselineOptimConfig
    distill: DistillLossConfig
    output: BaselineOutputConfig


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


def _resolve_path_in_mapping(raw: Dict[str, Any], base_dir: Path, keys: list[str]) -> Dict[str, Any]:
    data = dict(raw)
    for key in keys:
        if key in data:
            data[key] = _resolve_path(str(data[key]), base_dir)
    return data


def _to_model_config(raw: Dict[str, Any], base_dir: Path) -> BaselineModelConfig:
    data = dict(raw)
    data["path"] = _resolve_path(str(data["path"]), base_dir)
    return BaselineModelConfig(**data)


def _to_eval_task_config(raw: Dict[str, Any], base_dir: Path) -> EvalTaskConfig:
    data = dict(raw)
    if "path" in data:
        data["path"] = _resolve_path(str(data["path"]), base_dir)
    return EvalTaskConfig(**data)


def _to_output_config(raw: Dict[str, Any], base_dir: Path) -> BaselineOutputConfig:
    data = dict(raw)
    data["output_root"] = _resolve_path(str(data["output_root"]), base_dir)
    return BaselineOutputConfig(**data)


def load_eval_config(path: str | Path) -> BaselineEvalConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent

    datasets_raw = dict(raw.get("datasets", {}))
    datasets = EvalDatasetsConfig(
        pan=_to_eval_task_config(dict(datasets_raw.get("pan", {})), base_dir),
        mmlu=_to_eval_task_config(dict(datasets_raw.get("mmlu", {})), base_dir),
        gsm8k=_to_eval_task_config(dict(datasets_raw.get("gsm8k", {})), base_dir),
        humaneval=_to_eval_task_config(dict(datasets_raw.get("humaneval", {})), base_dir),
        mbpp=_to_eval_task_config(dict(datasets_raw.get("mbpp", {})), base_dir),
    )
    adapter = _resolve_path_in_mapping(
        dict(raw.get("adapter", {})),
        base_dir,
        ["manifest_path", "checkpoint_path"],
    )

    return BaselineEvalConfig(
        seed=int(raw.get("seed", 42)),
        model=_to_model_config(dict(raw["model"]), base_dir),
        adapter=AdapterConfig(**adapter),
        datasets=datasets,
        runtime=EvalRuntimeConfig(**dict(raw.get("runtime", {}))),
        output=EvalOutputConfig(
            output_root=_resolve_path(str(dict(raw["output"])["output_root"]), base_dir)
        ),
    )


def _to_supervised_data_config(raw: Dict[str, Any], base_dir: Path) -> SupervisedDataConfig:
    data = _resolve_path_in_mapping(
        raw,
        base_dir,
        ["train_split", "val_split", "test_split"],
    )
    return SupervisedDataConfig(**data)


def load_sft_config(path: str | Path) -> BaselineSFTConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    return BaselineSFTConfig(
        seed=int(raw.get("seed", 42)),
        model=_to_model_config(dict(raw["model"]), base_dir),
        data=_to_supervised_data_config(dict(raw["data"]), base_dir),
        lora=BaselineLoRAConfig(**dict(raw.get("lora", {}))),
        optim=BaselineOptimConfig(**dict(raw.get("optim", {}))),
        output=_to_output_config(dict(raw["output"]), base_dir),
    )


def load_distill_config(path: str | Path) -> BaselineDistillConfig:
    raw = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    return BaselineDistillConfig(
        seed=int(raw.get("seed", 42)),
        teacher=_to_model_config(dict(raw["teacher"]), base_dir),
        student=_to_model_config(dict(raw["student"]), base_dir),
        data=_to_supervised_data_config(dict(raw["data"]), base_dir),
        lora=BaselineLoRAConfig(**dict(raw.get("lora", {}))),
        optim=BaselineOptimConfig(**dict(raw.get("optim", {}))),
        distill=DistillLossConfig(**dict(raw.get("distill", {}))),
        output=_to_output_config(dict(raw["output"]), base_dir),
    )
