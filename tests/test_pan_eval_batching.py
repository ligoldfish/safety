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
        return "".join(
            self._id_to_text.get(int(token_id), "")
            for token_id in values
            if int(token_id) != 0
        )


class _FakeModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: _FakeTokenizer,
        outputs: dict[str, str],
    ) -> None:
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
                for token_id, mask_value in zip(
                    input_ids[row_idx].tolist(),
                    attention_mask[row_idx].tolist(),
                )
                if int(mask_value) == 1 and int(token_id) != 0
            ]
            prompt_text = "".join(
                self.tokenizer._id_to_text[int(token_id)]
                for token_id in prompt_token_ids
            )
            prompts.append(prompt_text)
            output_text = self.outputs[prompt_text]
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


class RefusalOnlyEvaluatorTests(unittest.TestCase):
    """Phase-G sanity evaluator: refusal-only metrics.

    Contract enforced:
    - Returns exactly the refusal-only keys used by scripts 09 / 10 / 13 / 14:
      ``harmful_refusal_rate``, ``harmful_unsafe_output_rate``,
      ``harmless_over_refusal_rate``, ``num_harmful``, ``num_harmless``,
      ``generations``. No ``harmful_safe_response_rate`` or thinking-era
      retry/incomplete fields.
    """

    def _patch_prompt_renderer(self):
        original = trainer_phase1.render_qwen_generation_prompt
        trainer_phase1.render_qwen_generation_prompt = (
            lambda _tokenizer, messages: str(messages[-1]["content"])
        )
        return original

    def test_returns_refusal_only_fields(self):
        tokenizer = _FakeTokenizer()
        outputs = {
            "prompt-harmful-refused": "I cannot help with that request.",
            "prompt-harmful-unsafe": "Sure, step 1: bypass the lock, step 2: steal the target.",
            "prompt-harmless-answer": "Here is a friendly recipe for cookies.",
            "prompt-harmless-overrefused": "I'm sorry, but I cannot help with that.",
        }
        model = _FakeModel(tokenizer, outputs)
        records = [
            {"id": "h1", "label": "harmful", "messages": [{"role": "user", "content": "prompt-harmful-refused"}]},
            {"id": "h2", "label": "harmful", "messages": [{"role": "user", "content": "prompt-harmful-unsafe"}]},
            {"id": "b1", "label": "harmless", "messages": [{"role": "user", "content": "prompt-harmless-answer"}]},
            {"id": "b2", "label": "harmless", "messages": [{"role": "user", "content": "prompt-harmless-overrefused"}]},
        ]

        original_render_prompt = self._patch_prompt_renderer()
        try:
            metrics = trainer_phase1.evaluate_generation_refusal_metrics(
                model,
                tokenizer,
                records,
                device=torch.device("cpu"),
                max_length=32,
                max_new_tokens=4,
                batch_size=2,
            )
        finally:
            trainer_phase1.render_qwen_generation_prompt = original_render_prompt

        expected_keys = {
            "harmful_refusal_rate",
            "harmful_unsafe_output_rate",
            "harmless_over_refusal_rate",
            "num_harmful",
            "num_harmless",
            "generations",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)
        # No thinking-era / safe-response fields must be present.
        for dead_key in (
            "harmful_safe_response_rate",
            "harmful_incomplete_output_rate",
            "retried_for_final_response",
            "used_max_new_tokens",
        ):
            self.assertNotIn(dead_key, metrics)
        for gen in metrics["generations"]:
            for dead_key in ("retried_for_final_response", "used_max_new_tokens"):
                self.assertNotIn(dead_key, gen)
            # final_text is the text actually fed to looks_like_refusal
            # (post-preamble-strip). For preamble-free outputs it equals the
            # raw decoded text; scripts/aggregators read it alongside
            # generated_text.
            self.assertIn("final_text", gen)
            self.assertIn("generated_text", gen)

        self.assertEqual(metrics["num_harmful"], 2)
        self.assertEqual(metrics["num_harmless"], 2)
        self.assertAlmostEqual(metrics["harmful_refusal_rate"], 0.5)
        self.assertAlmostEqual(metrics["harmful_unsafe_output_rate"], 0.5)
        self.assertAlmostEqual(metrics["harmless_over_refusal_rate"], 0.5)

    def test_single_pass_generation_no_retry(self):
        """Refusal-only evaluator must do exactly one generate() call per batch
        — no thinking-era retry pass with a different ``max_new_tokens``."""

        tokenizer = _FakeTokenizer()
        outputs = {"p1": "I cannot help.", "p2": "Sure, here you go."}
        model = _FakeModel(tokenizer, outputs)
        records = [
            {"id": "1", "label": "harmful", "messages": [{"role": "user", "content": "p1"}]},
            {"id": "2", "label": "harmful", "messages": [{"role": "user", "content": "p2"}]},
        ]
        original_render_prompt = self._patch_prompt_renderer()
        try:
            trainer_phase1.evaluate_generation_refusal_metrics(
                model,
                tokenizer,
                records,
                device=torch.device("cpu"),
                max_length=32,
                max_new_tokens=4,
                batch_size=2,
            )
        finally:
            trainer_phase1.render_qwen_generation_prompt = original_render_prompt
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["batch_size"], 2)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 4)

    def test_9b_reasoning_preamble_refusal_is_detected(self):
        """9B output pattern: long plain-text reasoning preamble + final
        refusal. Raw text would fail ``looks_like_refusal`` because the
        refusal sits past ``REFUSAL_PREFIX_WINDOW``. ``final_text`` (post-
        ``strip_qwen_thinking_content``) must extract the "Final Output
        Generation" section so refusal is correctly counted.

        Preamble fixture mirrors the shape validated in
        ``tests/test_qwen_eval_postprocess.py``.
        """

        tokenizer = _FakeTokenizer()
        preamble_refusal = (
            "Here's a thinking process that leads to the suggested response:\n\n"
            "1. **Analyze the Request**\n"
            "2. **Final Output Generation:**\n"
            "   * \"I cannot fulfill this request.\"\n"
            "   * \"Creating false stories about real companies is harmful.\"\n"
            "   * \"Please consult reputable sources instead.\"\n"
        )
        outputs = {"p-harmful-9b": preamble_refusal}
        model = _FakeModel(tokenizer, outputs)
        records = [
            {"id": "h1", "label": "harmful", "messages": [{"role": "user", "content": "p-harmful-9b"}]},
        ]
        original_render_prompt = self._patch_prompt_renderer()
        try:
            metrics = trainer_phase1.evaluate_generation_refusal_metrics(
                model,
                tokenizer,
                records,
                device=torch.device("cpu"),
                max_length=32,
                max_new_tokens=4,
                batch_size=1,
            )
        finally:
            trainer_phase1.render_qwen_generation_prompt = original_render_prompt

        self.assertAlmostEqual(metrics["harmful_refusal_rate"], 1.0)
        self.assertAlmostEqual(metrics["harmful_unsafe_output_rate"], 0.0)
        gen = metrics["generations"][0]
        self.assertTrue(gen["is_refusal"])
        # generated_text keeps the raw preamble for audit.
        self.assertIn("Here's a thinking process", gen["generated_text"])
        # final_text is the stripped classification input; preamble removed,
        # refusal line preserved.
        self.assertNotIn("thinking process", gen["final_text"].lower())
        self.assertIn("cannot fulfill", gen["final_text"].lower())

    def test_preamble_without_final_response_falls_back_to_raw(self):
        """If the stripper returns empty (preamble gate matched but no
        ``Final Output Generation`` section found), evaluator must fall
        back to the raw decoded text rather than silently passing an empty
        string to ``looks_like_refusal`` (which would flip the sample to
        "unsafe" for harmful records and skew metrics)."""

        tokenizer = _FakeTokenizer()
        # Preamble gate matches (starts with "Thinking Process:") but
        # ``_QWEN_REASONING_SECTION_RE`` does not find a "Final Output
        # Generation" / "Final Response" header -> stripper returns "".
        preamble_no_final = (
            "Thinking Process:\n\n"
            "1. **Analyze the Request**\n"
            "2. **Final Plan:**\n"
            "   1. State inability to comply.\n"
            "   2. Explain the risks.\n"
            "   3. Offer legitimate resources.\n"
        )
        outputs = {"p-no-final": preamble_no_final}
        model = _FakeModel(tokenizer, outputs)
        records = [
            {"id": "h1", "label": "harmful", "messages": [{"role": "user", "content": "p-no-final"}]},
        ]
        original_render_prompt = self._patch_prompt_renderer()
        try:
            metrics = trainer_phase1.evaluate_generation_refusal_metrics(
                model,
                tokenizer,
                records,
                device=torch.device("cpu"),
                max_length=32,
                max_new_tokens=4,
                batch_size=1,
            )
        finally:
            trainer_phase1.render_qwen_generation_prompt = original_render_prompt

        gen = metrics["generations"][0]
        # Fallback-to-raw: final_text equals the raw decoded text (non-empty).
        self.assertEqual(gen["final_text"], gen["generated_text"])
        self.assertIn("Thinking Process", gen["final_text"])


if __name__ == "__main__":
    unittest.main()
