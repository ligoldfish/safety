"""CPU regression test for the PhaseF LoRA-placement fix.

Bug (scripts/09_train_student_semalign.py): LoRA was injected on PAIR INDICES
[0,1,2] instead of the paired physical student layers (``unique_student_layers``,
e.g. [16,18,19]) whose activations ``L_layer`` supervises. This test pins:

  1. ``inject_lora_modules`` places LoRA on exactly the requested physical layers
     (and never on the bottom blocks 0/1/2) for a Qwen3.5-like linear_attn model.
  2. The 09 guard invariant -- injected physical layers must be a superset of the
     supervised student layers -- ACCEPTS the fix and REJECTS the old pair-index
     placement, so the index-space confusion cannot silently recur.
  3. The BT collision case (two pairs -> one student layer) yields the correct
     reduced unique-layer set with no length coupling.

Runs on CPU with a tiny mock model; no Qwen weights are loaded.
"""
import unittest

import torch.nn as nn

from src.models import (
    count_trainable_parameters,
    freeze_non_lora_parameters,
    inject_lora_modules,
)

# Mirrors configs/qwen35_9b_to_08b_phaseF_npu.yaml -> lora.target_modules
LORA_TARGETS = [
    "self_attn.v_proj",   # -> fallback linear_attn.in_proj_qkv (masked to value_dim)
    "self_attn.o_proj",   # -> fallback linear_attn.out_proj
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


class _LinearAttn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.value_dim = d
        self.in_proj_qkv = nn.Linear(d, 3 * d, bias=False)  # [q|k|v] concat
        self.out_proj = nn.Linear(d, d, bias=False)


class _MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)


class _Layer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear_attn = _LinearAttn(d)
        self.mlp = _MLP(d)


class _Inner(nn.Module):
    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(d) for _ in range(n))


class _MockQwen(nn.Module):
    """Minimal stand-in exposing ``model.model.layers[i].{linear_attn, mlp}``."""

    def __init__(self, num_layers: int = 24, d: int = 8) -> None:
        super().__init__()
        self.model = _Inner(num_layers, d)


def _touched_layers(replaced_module_names) -> set:
    # names look like "model.layers.<idx>.<suffix>"
    return {int(name.split(".")[2]) for name in replaced_module_names}


def _injection_covers_supervised(injection, pair_to_student_layer) -> bool:
    """Replicates the guard condition added in 09_train_student_semalign.py."""
    supervised = {int(v) for v in pair_to_student_layer.values()}
    injected = {int(x) for x in injection.layer_indices}
    return injected >= supervised


class LoraPlacementSmokeTest(unittest.TestCase):
    def test_inject_lands_on_paired_layers_not_bottom(self):
        # WGM-style pairing: pair idx -> physical student layer
        pair_to_student_layer = {0: 19, 1: 16, 2: 18}
        unique_student_layers = sorted(set(pair_to_student_layer.values()))  # [16,18,19]
        model = _MockQwen(num_layers=24, d=8)
        injection = inject_lora_modules(
            model,
            layer_indices=unique_student_layers,
            target_suffixes=LORA_TARGETS,
            rank=16,
            alpha=32.0,
            dropout=0.05,
        )
        touched = _touched_layers(injection.replaced_module_names)
        self.assertEqual(touched, {16, 18, 19})
        for bottom in (0, 1, 2):
            self.assertNotIn(bottom, touched)
        self.assertEqual(len(injection.replaced_module_names), 3 * len(LORA_TARGETS))

    def test_guard_accepts_fix_rejects_pair_indices(self):
        pair_to_student_layer = {0: 19, 1: 16, 2: 18}
        unique_student_layers = sorted(set(pair_to_student_layer.values()))

        fixed = inject_lora_modules(
            _MockQwen(), layer_indices=unique_student_layers,
            target_suffixes=LORA_TARGETS, rank=8, alpha=16.0, dropout=0.0,
        )
        self.assertTrue(_injection_covers_supervised(fixed, pair_to_student_layer))

        buggy = inject_lora_modules(
            _MockQwen(), layer_indices=sorted(int(k) for k in pair_to_student_layer),  # [0,1,2]
            target_suffixes=LORA_TARGETS, rank=8, alpha=16.0, dropout=0.0,
        )
        self.assertFalse(_injection_covers_supervised(buggy, pair_to_student_layer))

    def test_bt_collision_yields_two_unique_layers(self):
        pair_to_student_layer = {0: 15, 1: 14, 2: 14}  # collision
        unique_student_layers = sorted(set(pair_to_student_layer.values()))  # [14,15]
        injection = inject_lora_modules(
            _MockQwen(), layer_indices=unique_student_layers,
            target_suffixes=LORA_TARGETS, rank=8, alpha=16.0, dropout=0.0,
        )
        self.assertEqual(_touched_layers(injection.replaced_module_names), {14, 15})
        self.assertTrue(_injection_covers_supervised(injection, pair_to_student_layer))

    def test_only_lora_params_trainable_after_freeze(self):
        model = _MockQwen()
        inject_lora_modules(
            model, layer_indices=[16, 18, 19],
            target_suffixes=LORA_TARGETS, rank=8, alpha=16.0, dropout=0.0,
        )
        freeze_non_lora_parameters(model)
        trainable, total = count_trainable_parameters(model)
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertTrue(("lora_A" in name) or ("lora_B" in name), name)


if __name__ == "__main__":
    unittest.main()
