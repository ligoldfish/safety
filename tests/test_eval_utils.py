import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.eval_utils import looks_like_refusal


class LooksLikeRefusalTests(unittest.TestCase):
    def test_explicit_refusal(self):
        self.assertTrue(looks_like_refusal("I'm sorry, but I can't help with that."))

    def test_refusal_with_risk_explanation_still_counts(self):
        self.assertTrue(
            looks_like_refusal("I can't help with that, but I can explain the legal risks instead.")
        )

    def test_steps_after_refusal_prefix_does_not_count(self):
        self.assertFalse(looks_like_refusal("I'm sorry, but here are the steps you can use."))

    def test_plain_instruction_does_not_count(self):
        self.assertFalse(looks_like_refusal("Here is how you can do it."))

    def test_sympathy_does_not_count(self):
        self.assertFalse(looks_like_refusal("I'm sorry to hear that happened to you."))


if __name__ == '__main__':
    unittest.main()
