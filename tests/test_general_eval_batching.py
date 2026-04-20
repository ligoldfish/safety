import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.baselines.eval as baseline_eval
from src.baselines.datasets import CodeExample, GSM8KExample
from src.data.task_datasets import MCQExample


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
        repetition_penalty: float,
        use_cache: bool,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        del do_sample, repetition_penalty, use_cache, eos_token_id
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


class GeneralEvalBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_build_messages = baseline_eval.build_qwen_messages
        self.original_render_prompt = baseline_eval.render_qwen_generation_prompt
        baseline_eval.build_qwen_messages = lambda user_text: [{"role": "user", "content": str(user_text)}]
        baseline_eval.render_qwen_generation_prompt = lambda _tokenizer, messages: str(messages[-1]["content"])

    def tearDown(self) -> None:
        baseline_eval.build_qwen_messages = self.original_build_messages
        baseline_eval.render_qwen_generation_prompt = self.original_render_prompt

    def test_evaluate_mcq_batches_thinking_generation(self):
        tokenizer = _FakeTokenizer()
        examples = [
            MCQExample(
                sample_id="m1",
                question="What is the capital of France?",
                choices=["London", "Paris", "Rome", "Berlin"],
                answer_index=1,
                subject="geography",
            ),
            MCQExample(
                sample_id="m2",
                question="Which planet is known as the Red Planet?",
                choices=["Mercury", "Venus", "Mars", "Jupiter"],
                answer_index=2,
                subject="astronomy",
            ),
        ]
        prompt_1 = baseline_eval._render_official_mmlu_chat_prompt(examples[0])
        prompt_2 = baseline_eval._render_official_mmlu_chat_prompt(examples[1])
        model = _FakeModel(
            tokenizer,
            {
                (prompt_1, 64): "Paris",
                (prompt_2, 64): "Mars",
            },
        )

        result = baseline_eval.evaluate_mcq(
            model,
            tokenizer,
            examples,
            max_length=256,
            max_new_tokens=64,
            batch_size=2,
        )

        self.assertEqual(result["num_correct"], 2)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["batch_size"], 2)

    def test_evaluate_gsm8k_batches_generation(self):
        tokenizer = _FakeTokenizer()
        examples = [
            GSM8KExample(sample_id="g1", question="1 + 1 = ?", answer_text="#### 2", final_answer="2"),
            GSM8KExample(sample_id="g2", question="6 * 3 = ?", answer_text="#### 18", final_answer="18"),
        ]
        model = _FakeModel(
            tokenizer,
            {
                ("1 + 1 = ?", 64): "The answer is 2.",
                ("6 * 3 = ?", 64): "The answer is 18.",
            },
        )

        result = baseline_eval.evaluate_gsm8k(
            model,
            tokenizer,
            examples,
            max_length=128,
            max_new_tokens=64,
            batch_size=2,
        )

        self.assertEqual(result["num_correct"], 2)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["batch_size"], 2)

    def test_evaluate_code_generation_batches_generation(self):
        tokenizer = _FakeTokenizer()
        examples = [
            CodeExample(
                task_id="mbpp/1",
                prompt="Return 1.",
                tests=["assert solve() == 1"],
                entry_point="solve",
            ),
            CodeExample(
                task_id="mbpp/2",
                prompt="Return 2.",
                tests=["assert solve_two() == 2"],
                entry_point="solve_two",
            ),
        ]
        prompt_1 = baseline_eval._build_mbpp_officialish_prompt(examples[0])
        prompt_2 = baseline_eval._build_mbpp_officialish_prompt(examples[1])
        model = _FakeModel(
            tokenizer,
            {
                (prompt_1, 128): "def solve():\n    return 1",
                (prompt_2, 128): "def solve_two():\n    return 2",
            },
        )

        original_run_code = baseline_eval._run_code_program
        baseline_eval._run_code_program = lambda program, timeout_seconds: (True, "")
        try:
            result = baseline_eval.evaluate_code_generation(
                model,
                tokenizer,
                examples,
                dataset_name="mbpp",
                max_length=256,
                max_new_tokens=128,
                exec_timeout_seconds=1,
                batch_size=2,
            )
        finally:
            baseline_eval._run_code_program = original_run_code

        self.assertEqual(result["num_passed"], 2)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
