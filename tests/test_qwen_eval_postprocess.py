import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.datasets import extract_multiple_choice_prediction, sanitize_code_generation
from src.baselines.eval import _trim_to_code_start, _truncate_to_longest_python_prefix
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


if __name__ == "__main__":
    unittest.main()
