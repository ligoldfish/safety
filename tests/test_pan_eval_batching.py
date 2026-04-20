import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.training.trainer_phase1 as trainer_phase1


class _FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._codex_chat_template_enable_thinking = True
        self._text_to_id: dict[str, int] = {}
        self._id_to_text: dict[int, str] = {}
        self._next_id = 1

    def _encode_piece(self, text: str) -> int:
        value = str(text)
        token_id = self._text_to_id.get(value)
        if token_id is None:
            token_id = self._next_id
            self._next_id += 1
            self._text_to_id[value] = token_id
            self._id_to_text[token_id] = value
        return token_id

    def __call__(
        self,
        texts,
        *,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = True,
        max_length: int | None = None,
    ):
        del return_tensors, truncation, max_length
        if isinstance(texts, str):
            texts = [texts]
        rows = [[self._encode_piece(str(text))] for text in texts]
        max_width = max(len(row) for row in rows) if rows else 0
        padded_rows = []
        masks = []
        for row in rows:
            pad_width = max_width - len(row) if padding else 0
            padded_rows.append(([self.pad_token_id] * pad_width) + row)
            masks.append(([0] * pad_width) + ([1] * len(row)))
        return {
            "input_ids": torch.tensor(padded_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.tolist()
        else:
            values = list(token_ids)
        return "".join(self._id_to_text.get(int(token_id), "") for token_id in values if int(token_id) != 0)


class _FakeModel(torch.nn.Module):
    def __init__(self, tokenizer: _FakeTokenizer, outputs: dict[tuple[str, int], str]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.outputs = outputs
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self._codex_runtime_backend = "cpu"
        self._codex_xla_model = None
        self.generate_calls: list[dict[str, object]] = []

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        use_cache: bool,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        del do_sample, use_cache, eos_token_id
        prompts: list[str] = []
        output_token_rows: list[list[int]] = []
        for row_idx in range(int(input_ids.size(0))):
            prompt_token_ids = [
                int(token_id)
                for token_id, mask_value in zip(input_ids[row_idx].tolist(), attention_mask[row_idx].tolist())
                if int(mask_value) == 1 and int(token_id) != 0
            ]
            prompt_text = "".join(self.tokenizer._id_to_text[int(token_id)] for token_id in prompt_token_ids)
            prompts.append(prompt_text)
            output_text = self.outputs[(prompt_text, int(max_new_tokens))]
            output_token_rows.append([self.tokenizer._encode_piece(output_text)])

        self.generate_calls.append(
            {
                "batch_size": int(input_ids.size(0)),
                "max_new_tokens": int(max_new_tokens),
                "prompts": prompts,
            }
        )
        prompt_width = int(input_ids.size(1))
        output_width = max((len(row) for row in output_token_rows), default=0)
        generated = torch.full(
            (int(input_ids.size(0)), prompt_width + output_width),
            int(pad_token_id),
            dtype=torch.long,
            device=input_ids.device,
        )
        generated[:, :prompt_width] = input_ids
        for row_idx, token_row in enumerate(output_token_rows):
            generated[row_idx, prompt_width:prompt_width + len(token_row)] = torch.tensor(
                token_row,
                dtype=torch.long,
                device=input_ids.device,
            )
        return generated


class PanEvalBatchingTests(unittest.TestCase):
    def test_pan_eval_batches_first_pass_and_retries_only_incomplete(self):
        tokenizer = _FakeTokenizer()
        outputs = {
            ("prompt-1", 2): "Thinking Process:\n1. Plan the refusal.",
            (
                "prompt-2",
                2,
            ): "Here's a thinking process that leads to the suggested response:\n\n2. **Final Output Generation:**\n   * \"I can't help with that.\"\n",
            (
                "prompt-1",
                8,
            ): "Here's a thinking process that leads to the suggested response:\n\n2. **Final Output Generation:**\n   * \"I can't help with that.\"\n",
        }
        model = _FakeModel(tokenizer, outputs)
        records = [
            {"id": "1", "label": "harmful", "messages": [{"role": "user", "content": "prompt-1"}]},
            {"id": "2", "label": "harmful", "messages": [{"role": "user", "content": "prompt-2"}]},
        ]

        original_render_prompt = trainer_phase1.render_qwen_generation_prompt
        trainer_phase1.render_qwen_generation_prompt = lambda _tokenizer, messages: str(messages[-1]["content"])
        try:
            metrics = trainer_phase1.evaluate_generation_refusal_metrics(
                model,
                tokenizer,
                records,
                device=torch.device("cpu"),
                max_length=32,
                max_new_tokens=8,
                batch_size=2,
                initial_max_new_tokens=2,
            )
        finally:
            trainer_phase1.render_qwen_generation_prompt = original_render_prompt

        self.assertEqual(len(model.generate_calls), 2)
        self.assertEqual(model.generate_calls[0]["batch_size"], 2)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 2)
        self.assertEqual(model.generate_calls[1]["batch_size"], 1)
        self.assertEqual(model.generate_calls[1]["max_new_tokens"], 8)
        self.assertAlmostEqual(metrics["harmful_refusal_rate"], 1.0)
        self.assertAlmostEqual(metrics["harmful_safe_response_rate"], 1.0)
        self.assertAlmostEqual(metrics["harmful_incomplete_output_rate"], 0.0)
        self.assertTrue(metrics["generations"][0]["retried_for_final_response"])
        self.assertEqual(metrics["generations"][0]["used_max_new_tokens"], 8)
        self.assertFalse(metrics["generations"][1]["retried_for_final_response"])
        self.assertEqual(metrics["generations"][1]["used_max_new_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
