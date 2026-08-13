from __future__ import annotations

import unittest

import torch

from src.training.trainer_phase1 import count_nonpadding_tokens


class TrainingBudgetTests(unittest.TestCase):
    def test_nonpadding_token_count_uses_attention_mask_not_padded_width(self) -> None:
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        self.assertEqual(count_nonpadding_tokens(mask), 5)

    def test_nonpadding_token_count_rejects_non_matrix_or_non_binary_masks(self) -> None:
        for mask in (
            torch.tensor([1, 1]),
            torch.tensor([[1, 2]]),
            torch.tensor([[1.0, float("nan")]]),
        ):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                count_nonpadding_tokens(mask)


if __name__ == "__main__":
    unittest.main()
