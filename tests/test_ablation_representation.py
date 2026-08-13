from __future__ import annotations

import unittest

import torch

from src.ablations.strategies.representation import extract_position_hidden
from src.features.first_gen_token import generated_token_mask
from src.training.trainer_phase1 import select_training_representations


class AblationRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden = torch.tensor(
            [
                [[1.0], [2.0], [3.0], [10.0], [20.0], [0.0]],
                [[4.0], [5.0], [30.0], [40.0], [50.0], [60.0]],
            ]
        )
        self.prompt_mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long
        )
        self.generated_mask = torch.tensor(
            [[0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.long
        )

    def test_last_and_mean_prompt_positions(self) -> None:
        last = extract_position_hidden(self.hidden, self.prompt_mask, mode="last_prompt")
        mean = extract_position_hidden(self.hidden, self.prompt_mask, mode="mean_prompt")
        torch.testing.assert_close(last, torch.tensor([[3.0], [5.0]]))
        torch.testing.assert_close(mean, torch.tensor([[2.0], [4.5]]))

    def test_generated_positions_use_only_valid_generated_tokens(self) -> None:
        first = extract_position_hidden(
            self.hidden, self.prompt_mask, mode="first_generated", generated_mask=self.generated_mask
        )
        first_four = extract_position_hidden(
            self.hidden,
            self.prompt_mask,
            mode="first_4_generated_mean",
            generated_mask=self.generated_mask,
        )
        torch.testing.assert_close(first, torch.tensor([[10.0], [30.0]]))
        torch.testing.assert_close(first_four, torch.tensor([[15.0], [45.0]]))

    def test_generated_mode_requires_generated_tokens_per_sample(self) -> None:
        bad = self.generated_mask.clone()
        bad[0].zero_()
        with self.assertRaisesRegex(ValueError, "no generated tokens"):
            extract_position_hidden(
                self.hidden, self.prompt_mask, mode="first_generated", generated_mask=bad
            )

    def test_invalid_shape_and_unknown_mode_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            extract_position_hidden(self.hidden, self.prompt_mask[:, :-1], mode="last_prompt")
        with self.assertRaisesRegex(ValueError, "unsupported representation mode"):
            extract_position_hidden(self.hidden, self.prompt_mask, mode="mystery")

    def test_generated_mask_starts_after_padded_input_width(self) -> None:
        prompt = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        sequences = torch.ones((2, 7), dtype=torch.long)
        full_mask, generated_mask = generated_token_mask(prompt, sequences, generated_count=3)
        torch.testing.assert_close(full_mask, torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]))
        torch.testing.assert_close(generated_mask, torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]]))

    def test_generated_mask_stops_after_first_eos_even_when_pad_equals_eos(self) -> None:
        prompt = torch.tensor([[0, 1, 1], [1, 1, 1]])
        sequences = torch.tensor([[0, 4, 5, 9, 2, 2, 2], [6, 7, 8, 3, 4, 5, 2]])
        full_mask, generated_mask = generated_token_mask(
            prompt,
            sequences,
            generated_count=4,
            pad_token_id=2,
            eos_token_id=2,
        )
        torch.testing.assert_close(generated_mask[0], torch.tensor([0, 0, 0, 1, 1, 0, 0]))
        torch.testing.assert_close(generated_mask[1], torch.tensor([0, 0, 0, 1, 1, 1, 1]))
        torch.testing.assert_close(full_mask[0], torch.tensor([0, 1, 1, 1, 1, 0, 0]))

    def test_phasef_teacher_forced_positions_share_representation_semantics(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0], [2.0], [3.0], [10.0], [20.0], [0.0]],
                [[4.0], [5.0], [30.0], [40.0], [0.0], [0.0]],
            ]
        )
        attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
        prompt_lengths = torch.tensor([3, 2])
        torch.testing.assert_close(
            select_training_representations(hidden, attention, prompt_lengths, mode="last_prompt"),
            torch.tensor([[3.0], [5.0]]),
        )
        torch.testing.assert_close(
            select_training_representations(hidden, attention, prompt_lengths, mode="mean_prompt"),
            torch.tensor([[2.0], [4.5]]),
        )
        torch.testing.assert_close(
            select_training_representations(hidden, attention, prompt_lengths, mode="first_generated"),
            torch.tensor([[10.0], [30.0]]),
        )
        torch.testing.assert_close(
            select_training_representations(hidden, attention, prompt_lengths, mode="first_4_generated_mean"),
            torch.tensor([[15.0], [35.0]]),
        )


if __name__ == "__main__":
    unittest.main()
