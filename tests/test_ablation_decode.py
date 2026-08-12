from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.baselines.config import load_eval_config
from src.training.trainer_phase1 import evaluate_generation_refusal_metrics


class _Tokenizer:
    padding_side = "right"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.ones((len(texts), 1), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        }

    def decode(self, values, **kwargs):
        del values, kwargs
        return "I cannot help."


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.ones((input_ids.size(0), 1), dtype=torch.long)], dim=1)


class AblationDecodeTests(unittest.TestCase):
    def test_sampling_parameters_reach_model_generate(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        evaluate_generation_refusal_metrics(
            model,
            tokenizer,
            [{"id": "x", "label": "harmful", "messages": [{"role": "user", "content": "x"}]}],
            device=torch.device("cpu"),
            max_length=8,
            max_new_tokens=4,
            temperature=0.7,
            top_p=0.9,
        )
        self.assertTrue(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["temperature"], 0.7)
        self.assertEqual(model.kwargs["top_p"], 0.9)

    def test_zero_temperature_stays_greedy_and_config_loads_decode_fields(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        evaluate_generation_refusal_metrics(
            model,
            tokenizer,
            [{"id": "x", "label": "harmful", "messages": [{"role": "user", "content": "x"}]}],
            device=torch.device("cpu"),
            max_length=8,
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        self.assertFalse(model.kwargs["do_sample"])
        self.assertNotIn("temperature", model.kwargs)
        self.assertNotIn("top_p", model.kwargs)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "eval.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "seed": 42,
                        "model": {"name": "m", "path": "model"},
                        "adapter": {},
                        "datasets": {"pan": {"path": "data.jsonl", "temperature": 0.7, "top_p": 0.9}},
                        "runtime": {},
                        "output": {"output_root": "out"},
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_eval_config(config)
        self.assertEqual(loaded.datasets.pan.temperature, 0.7)
        self.assertEqual(loaded.datasets.pan.top_p, 0.9)


if __name__ == "__main__":
    unittest.main()
