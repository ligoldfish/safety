import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.datasets import (
    extract_multiple_choice_prediction,
    extract_official_gsm8k_prediction,
    extract_official_mmlu_prediction,
    sanitize_code_generation,
)
from src.baselines.eval import (
    _build_humaneval_official_prompt,
    _extract_humaneval_completion,
    _gsm8k_answers_match,
    _trim_to_code_start,
    _truncate_to_longest_python_prefix,
)
from src.baselines.datasets import CodeExample
from src.data.template_qwen import strip_qwen_thinking_content


class QwenThinkingPostprocessTests(unittest.TestCase):
    def test_strip_qwen_thinking_returns_final_response(self):
        text = "<think>\nreasoning\n</think>\n\nFinal answer: 42"
        self.assertEqual(
            strip_qwen_thinking_content(text, require_final_response=True),
            "Final answer: 42",
        )

    def test_strip_qwen_thinking_requires_closed_final_segment(self):
        text = "<think>\nreasoning without closing tag"
        self.assertEqual(
            strip_qwen_thinking_content(text, require_final_response=True),
            "",
        )

    def test_extract_multiple_choice_prediction_from_json(self):
        text = '{"answer": "C"}'
        self.assertEqual(
            extract_multiple_choice_prediction(text, ["A", "B", "C", "D"]),
            "C",
        )

    def test_extract_official_mmlu_prediction_from_choice_text(self):
        text = "After reasoning, the capital city answer is Paris."
        self.assertEqual(
            extract_official_mmlu_prediction(
                text,
                ["A", "B", "C", "D"],
                {
                    "A": "London",
                    "B": "Paris",
                    "C": "Rome",
                    "D": "Berlin",
                },
            ),
            "B",
        )

    def test_extract_official_gsm8k_prediction_and_numeric_match(self):
        text = "We compute the total carefully. Therefore the answer is 18.0000"
        prediction = extract_official_gsm8k_prediction(text)
        self.assertEqual(prediction, "18.0000")
        self.assertTrue(_gsm8k_answers_match("18", prediction))

    def test_sanitize_code_generation_strips_qwen_thinking_and_preamble(self):
        text = "<think>\nreasoning\n</think>\n\nHere is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertEqual(
            sanitize_code_generation(text, require_final_response=True),
            "def add(a, b):\n    return a + b",
        )

    def test_trim_and_truncate_python_prefix_remove_trailing_explanation(self):
        text = """Here is the code:
def add(a, b):
    return a + b

This function adds two numbers."""
        trimmed = _trim_to_code_start(text)
        truncated = _truncate_to_longest_python_prefix(trimmed)
        self.assertEqual(
            truncated,
            "def add(a, b):\n    return a + b",
        )

    def test_humaneval_official_prompt_and_completion_extraction(self):
        example = CodeExample(
            task_id="HumanEval/0",
            prompt=(
                "def add(a, b):\n"
                "    \"\"\"Return the sum of a and b.\"\"\"\n"
            ),
            tests=["assert add(1, 2) == 3"],
            entry_point="add",
        )
        prompt = _build_humaneval_official_prompt(example)
        self.assertIn("Write a Python function `add(a, b)`", prompt)
        completion = _extract_humaneval_completion(
            "<think>\nreasoning\n</think>\n```python\ndef add(a, b):\n    return a + b\n```",
            "add",
            require_final_response=True,
        )
        self.assertEqual(completion.strip(), "return a + b")


if __name__ == "__main__":
    unittest.main()
