import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.train import SupervisedCollator
from src.data.template_qwen import render_qwen_final_response_prefix, render_qwen_supervised_text
from src.phase_b.hidden_states import load_hidden_state_split
from src.training.trainer_phase1 import SemAlignCollator


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        if tokenize:
            raise NotImplementedError("This fake tokenizer only supports tokenize=False.")
        if enable_thinking is not True:
            raise ValueError("FakeQwenTokenizer only supports thinking mode.")

        non_assistant = [f"{msg['role']}:{msg['content']}" for msg in messages if msg["role"] != "assistant"]
        prefix = "<|>".join(non_assistant)
        if add_generation_prompt:
            return prefix + "<A><think>\n"

        assistant_text = ""
        if messages and messages[-1]["role"] == "assistant":
            assistant_text = str(messages[-1]["content"])
        return prefix + "<A><think>\n\n</think>\n\n" + assistant_text + "<E>"

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ):
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


class ThinkingPipelinePositionTests(unittest.TestCase):
    def test_supervised_collator_masks_thinking_prefix_tokens(self):
        tokenizer = FakeQwenTokenizer()
        record = _build_record()
        prefix_text = render_qwen_final_response_prefix(tokenizer, record["messages"])
        full_text = render_qwen_supervised_text(tokenizer, record["messages"], record["target_response"])
        self.assertGreater(len(full_text), len(prefix_text))

        collator = SupervisedCollator(tokenizer, max_length=1024)
        batch = collator([record])

        prefix_len = len(prefix_text)
        labels = batch.labels[0]
        self.assertTrue(torch.all(labels[:prefix_len] == -100))
        self.assertNotEqual(int(labels[prefix_len].item()), -100)

    def test_semalign_collator_aligns_on_final_response_prefix(self):
        tokenizer = FakeQwenTokenizer()
        record = _build_record()
        prefix_text = render_qwen_final_response_prefix(tokenizer, record["messages"])

        collator = SemAlignCollator(tokenizer, max_length=1024, layer_ids=[0])
        batch = collator(
            [
                {
                    "record": record,
                    "targets": {0: torch.tensor([1.0, 2.0], dtype=torch.float32)},
                }
            ]
        )

        self.assertEqual(int(batch.response_prefix_last_positions[0].item()), len(prefix_text) - 1)
        prefix_len = len(prefix_text)
        self.assertTrue(torch.all(batch.labels[0, :prefix_len] == -100))
        self.assertNotEqual(int(batch.labels[0, prefix_len].item()), -100)

    def test_hidden_state_loader_rejects_legacy_feature_type(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            split_dir = Path(tmpdir)
            torch.save(
                {
                    "feature_type": "first_generated_token_hidden_state",
                    "sample_ids": ["sample-1"],
                    "labels": ["harmful"],
                    "hidden_by_layer": {"0": torch.zeros(1, 2)},
                },
                split_dir / "part_000.pt",
            )

            with self.assertRaisesRegex(ValueError, "thinking-on final-response-prefix pipeline"):
                load_hidden_state_split(split_dir)


if __name__ == "__main__":
    unittest.main()
