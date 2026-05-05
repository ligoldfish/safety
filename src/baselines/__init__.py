from .config import (
    AdapterConfig,
    BaselineDistillConfig,
    BaselineEvalConfig,
    BaselineLoRAConfig,
    BaselineModelConfig,
    BaselineSFTConfig,
    EvalTaskConfig,
    load_distill_config,
    load_eval_config,
    load_sft_config,
)
from .datasets import (
    CodeExample,
    GSM8KExample,
    extract_prediction_number,
    extract_reference_number,
    load_code_examples,
    load_gsm8k_examples,
    normalize_numeric_answer,
    sanitize_code_generation,
)
from .debug import collect_error_predictions, export_error_predictions

_EVAL_EXPORTS = {
    "evaluate_code_generation",
    "evaluate_gsm8k",
    "evaluate_mcq",
    "evaluate_pan",
    "load_mcq_examples",
    "load_model_for_evaluation",
}

_TRAIN_EXPORTS = {
    "PanSupervisedDataset",
    "SupervisedBatch",
    "SupervisedCollator",
    "build_supervised_dataloader",
    "forward_distill_batch",
    "forward_supervised_batch",
}


def __getattr__(name: str):
    if name in _EVAL_EXPORTS:
        from . import eval as eval_module

        value = getattr(eval_module, name)
        globals()[name] = value
        return value
    if name in _TRAIN_EXPORTS:
        from . import train as train_module

        value = getattr(train_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdapterConfig",
    "BaselineDistillConfig",
    "BaselineEvalConfig",
    "BaselineLoRAConfig",
    "BaselineModelConfig",
    "BaselineSFTConfig",
    "CodeExample",
    "EvalTaskConfig",
    "GSM8KExample",
    "PanSupervisedDataset",
    "SupervisedBatch",
    "SupervisedCollator",
    "build_supervised_dataloader",
    "collect_error_predictions",
    "evaluate_code_generation",
    "evaluate_gsm8k",
    "evaluate_mcq",
    "evaluate_pan",
    "export_error_predictions",
    "extract_prediction_number",
    "extract_reference_number",
    "forward_distill_batch",
    "forward_supervised_batch",
    "load_code_examples",
    "load_distill_config",
    "load_eval_config",
    "load_gsm8k_examples",
    "load_mcq_examples",
    "load_model_for_evaluation",
    "load_sft_config",
    "normalize_numeric_answer",
    "sanitize_code_generation",
]
