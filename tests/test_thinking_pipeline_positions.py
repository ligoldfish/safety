import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.train import SupervisedCollator
from src.data.template_qwen import (
    render_qwen_generation_prompt,
    render_qwen_supervised_text,
)
from src.phase_b.hidden_states import load_hidden_state_split
from src.training.trainer_phase1 import SemAlignCollator


class FakeQwenTokenizer:
    """Minimal Qwen-like tokenizer.

    Chat template mirrors ``add_generation_prompt=True`` -> ends with an
    ``<A>`` marker. ``add_generation_prompt=False`` keeps the assistant
    body plus ``<E>``. One character -> one token so text length equals
    token-count, letting us verify boundary positions directly.
    """

    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del enable_thinking
        if tokenize:
            raise NotImplementedError("This fake tokenizer only supports tokenize=False.")
        non_assistant = [
            f"{msg['role']}:{msg['content']}"
            for msg in messages
            if msg["role"] != "assistant"
        ]
        prefix = "<|>".join(non_assistant)
        if add_generation_prompt:
            return prefix + "<A>"
        assistant_text = ""
        if messages and messages[-1]["role"] == "assistant":
            assistant_text = str(messages[-1]["content"])
        return prefix + "<A>" + assistant_text + "<E>"

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ):
        del return_tensors
        if isinstance(texts, str):
            texts = [texts]
        token_rows = []
        for text in texts:
            encoded = [(ord(ch) % 251) + 1 for ch in str(text)]
            if truncation:
                encoded = encoded[:max_length]
            token_rows.append(encoded)

        max_tokens = max((len(row) for row in token_rows), default=0)
        padded_rows = []
        attention_rows = []
        for row in token_rows:
            pad_width = max_tokens - len(row)
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_width + row
                attention = [0] * pad_width + [1] * len(row)
            else:
                padded = row + [self.pad_token_id] * pad_width
                attention = [1] * len(row) + [0] * pad_width
            padded_rows.append(padded)
            attention_rows.append(attention)

        return {
            "input_ids": torch.tensor(padded_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }


def _build_record() -> dict:
    return {
        "id": "sample-1",
        "label": "harmful",
        "messages": [{"role": "user", "content": "Explain safe behavior."}],
        "target_response": "Refuse and redirect safely.",
    }


class FirstGeneratedTokenPositionTests(unittest.TestCase):
    def test_supervised_collator_masks_generation_prompt_tokens(self):
        tokenizer = FakeQwenTokenizer()
        record = _build_record()
        prompt_text = render_qwen_generation_prompt(tokenizer, record["messages"])
        full_text = render_qwen_supervised_text(
            tokenizer, record["messages"], record["target_response"]
        )
        self.assertGreater(len(full_text), len(prompt_text))

        collator = SupervisedCollator(tokenizer, max_length=1024)
        batch = collator([record])

        prompt_len = len(prompt_text)
        labels = batch.labels[0]
        self.assertTrue(torch.all(labels[:prompt_len] == -100))
        self.assertNotEqual(int(labels[prompt_len].item()), -100)

    def test_semalign_collator_aligns_on_generation_prompt_last_token(self):
        tokenizer = FakeQwenTokenizer()
        record = _build_record()
        prompt_text = render_qwen_generation_prompt(tokenizer, record["messages"])

        collator = SemAlignCollator(tokenizer, max_length=1024, layer_ids=[0])
        batch = collator(
            [
                {
                    "record": record,
                    "targets": {0: torch.tensor([1.0, 2.0], dtype=torch.float32)},
                }
            ]
        )

        # prompt_last_positions is the index of the last token of
        # apply_chat_template(..., add_generation_prompt=True), i.e. the
        # first-generated-token capture position matching 01_extract_hidden_states.
        self.assertEqual(int(batch.prompt_last_positions[0].item()), len(prompt_text) - 1)
        prompt_len = len(prompt_text)
        self.assertTrue(torch.all(batch.labels[0, :prompt_len] == -100))
        self.assertNotEqual(int(batch.labels[0, prompt_len].item()), -100)


class HiddenStateLoaderStrictnessTests(unittest.TestCase):
    def _write_shard(self, split_dir: Path, feature_type: str) -> None:
        torch.save(
            {
                "feature_type": feature_type,
                "sample_ids": ["sample-1"],
                "labels": ["harmful"],
                "hidden_by_layer": {"0": torch.zeros(1, 2)},
            },
            split_dir / "part_000.pt",
        )

    def test_loader_accepts_canonical_feature_type_by_default(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            split_dir = Path(tmpdir)
            self._write_shard(split_dir, "first_generated_token_hidden_state")
            split = load_hidden_state_split(split_dir)
            self.assertFalse(split.legacy_final_response_prefix)
            self.assertEqual(split.labels, ["harmful"])

    def test_loader_rejects_legacy_final_response_prefix_by_default(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            split_dir = Path(tmpdir)
            self._write_shard(split_dir, "final_response_prefix_hidden_state")
            with self.assertRaisesRegex(ValueError, "Legacy hidden-state shard"):
                load_hidden_state_split(split_dir)

    def test_loader_accepts_legacy_shard_when_explicitly_opted_in(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            split_dir = Path(tmpdir)
            self._write_shard(split_dir, "final_response_prefix_hidden_state")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                split = load_hidden_state_split(
                    split_dir,
                    allow_legacy_final_response_prefix=True,
                )
            self.assertTrue(split.legacy_final_response_prefix)
            self.assertTrue(any("LEGACY final_response_prefix" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
