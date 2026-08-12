from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from src.ablations.platform import resolve_portable_path


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
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if float(self.temperature) < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be in (0,1]")


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
class SafetyDatasetConfig:
    """Optional safety SFT dataset spec consumed at config-load time."""

    name: str = ""
    force_rebuild: bool = False
    cache_dir: str = ""
    source_name: str = ""
    split: str = ""
    sources: list[str] = field(default_factory=list)
    repo_or_data_path: str = ""
    file_name: str = ""
    refusal_template: str = ""
    system_prompt: str = ""
    # STL: pull alpaca_small.json as harmless contrast.
    include_harmless_contrast: bool = False
    harmless_file_name: str = "alpaca_small.json"
    harmless_max_samples: int = 0  # 0 = match harmful count automatically
    # BeaverTails: drop duplicate prompts (multi-response per prompt).
    dedup_prompts: bool = True
    # Round 2: BeaverTails uses category-dict prompt-level labels by
    # default; legacy ``"is_safe"`` retained for reproducing older runs.
    label_strategy: str = "category_any"
    # Round 2: Tülu3 v2 helpful slices (in-domain harmless contrast).
    helpful_sources: list[str] = field(default_factory=list)
    helpful_max_samples: int = 0  # 0 = match harmful count automatically
    # New dataset materialization knobs. Train/eval sampling are independent:
    # max_* == 0 or subset_mode == False means "use the full available split".
    train_subset_mode: bool = True
    max_train_samples: int = 0
    max_train_samples_per_label: int = 0
    eval_subset_mode: bool = False
    max_eval_samples: int = 0
    max_eval_samples_per_label: int = 0
    eval_output_path: str = ""
    eval_holdout_fraction: float = 0.1


@dataclass
class SupervisedDataConfig:
    train_split: str
    val_split: str
    test_split: str = ""
    safety_dataset: SafetyDatasetConfig = field(default_factory=SafetyDatasetConfig)


@dataclass
class BaselineLoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
    )


@dataclass
class BaselineOptimConfig:
    epochs: int = 1
    batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    # Qwen3 official full-FT recipe (Qwen3.5-4B reference): warmup_ratio=0.03,
    # cosine LR, AdamW betas=(0.9, 0.95), max_grad_norm=1.0. Defaults preserve
    # legacy behavior (no scheduler); YAML overrides activate stable training.
    warmup_ratio: float = 0.0
    max_grad_norm: float = 0.0
    lr_scheduler: str = "constant"
    early_stopping_patience: int = 0
    max_length: int = 1024
    max_new_tokens: int = 256
    log_every_steps: int = 1
    gradient_checkpointing: bool = False
    save_optimizer_state: bool = True
    # iso-HR matching: enable sub-epoch val-HR eval every N optimizer steps. A
    # step checkpoint (step_NNNNNN.pt) is persisted only when the val HR is within
    # match_band of match_target_hr (or match_target_hr < 0 = save every probe),
    # which bounds disk for full-finetune checkpoints.
    checkpoint_every_steps: int = 0
    match_target_hr: float = -1.0
    match_band: float = 0.02


@dataclass
class BaselineTrainingModeConfig:
    """Selects between LoRA adapter training and full fine-tuning.

    ``mode == "lora"`` keeps the historical behaviour: inject LoRA modules,
    freeze everything else, save only LoRA delta. ``mode == "full_finetune"``
    drops LoRA, leaves all parameters trainable, and saves the full model
    state dict. ``ours`` (SemAlign script 09) is unaffected -- it has no
    BaselineTrainingModeConfig and continues to use LoRA unconditionally.
    """

    mode: str = "lora"


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
    training: BaselineTrainingModeConfig = field(default_factory=BaselineTrainingModeConfig)


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
    training: BaselineTrainingModeConfig = field(default_factory=BaselineTrainingModeConfig)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return payload


def _resolve_path(value: str, base_dir: Path, *, category: str) -> str:
    return resolve_portable_path(value, base_dir, category=category)


def _resolve_path_in_mapping(
    raw: Dict[str, Any], base_dir: Path, keys: list[str], *, category: str
) -> Dict[str, Any]:
    data = dict(raw)
    for key in keys:
        if key in data:
            data[key] = _resolve_path(str(data[key]), base_dir, category=category)
    return data


def _to_model_config(raw: Dict[str, Any], base_dir: Path) -> BaselineModelConfig:
    data = dict(raw)
    data["path"] = _resolve_path(str(data["path"]), base_dir, category="model")
    return BaselineModelConfig(**data)


def _to_eval_task_config(raw: Dict[str, Any], base_dir: Path) -> EvalTaskConfig:
    data = dict(raw)
    if "path" in data:
        data["path"] = _resolve_path(str(data["path"]), base_dir, category="data")
    return EvalTaskConfig(**data)


def _to_output_config(raw: Dict[str, Any], base_dir: Path) -> BaselineOutputConfig:
    data = dict(raw)
    data["output_root"] = _resolve_path(str(data["output_root"]), base_dir, category="output")
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
        category="output",
    )

    return BaselineEvalConfig(
        seed=int(raw.get("seed", 42)),
        model=_to_model_config(dict(raw["model"]), base_dir),
        adapter=AdapterConfig(**adapter),
        datasets=datasets,
        runtime=EvalRuntimeConfig(**dict(raw.get("runtime", {}))),
        output=EvalOutputConfig(
            output_root=_resolve_path(str(dict(raw["output"])["output_root"]), base_dir, category="output")
        ),
    )


def _to_safety_dataset_config(raw: Dict[str, Any], base_dir: Path) -> SafetyDatasetConfig:
    data = dict(raw)
    if "repo_or_data_path" in data and data["repo_or_data_path"]:
        data["repo_or_data_path"] = _resolve_path(str(data["repo_or_data_path"]), base_dir, category="data")
    if "cache_dir" in data and data["cache_dir"]:
        data["cache_dir"] = _resolve_path(str(data["cache_dir"]), base_dir, category="data")
    if "eval_output_path" in data and data["eval_output_path"]:
        data["eval_output_path"] = _resolve_path(str(data["eval_output_path"]), base_dir, category="data")
    if "sources" in data and data["sources"] is not None:
        data["sources"] = [str(item) for item in data["sources"]]
    if "helpful_sources" in data and data["helpful_sources"] is not None:
        data["helpful_sources"] = [str(item) for item in data["helpful_sources"]]
    return SafetyDatasetConfig(**data)


def _to_supervised_data_config(raw: Dict[str, Any], base_dir: Path) -> SupervisedDataConfig:
    safety_dataset_raw = dict(raw.pop("safety_dataset", {}) or {})
    data = _resolve_path_in_mapping(
        raw,
        base_dir,
        ["train_split", "val_split", "test_split"],
        category="data",
    )
    return SupervisedDataConfig(
        train_split=str(data.get("train_split", "")),
        val_split=str(data.get("val_split", "")),
        test_split=str(data.get("test_split", "")),
        safety_dataset=_to_safety_dataset_config(safety_dataset_raw, base_dir),
    )


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
        training=BaselineTrainingModeConfig(**dict(raw.get("training", {}))),
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
        training=BaselineTrainingModeConfig(**dict(raw.get("training", {}))),
    )
