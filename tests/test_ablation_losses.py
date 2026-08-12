from __future__ import annotations

import unittest

import torch

from src.ablations.strategies.losses import layer_alignment_loss, supervision_weights
from src.training.losses import cosine_layer_alignment_loss


class AblationLossTests(unittest.TestCase):
    def test_four_losses_have_known_zero_or_positive_values(self) -> None:
        predicted = {0: torch.tensor([[1.0, 0.0], [0.0, 2.0]])}
        target = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        for kind in ("cosine", "normalized_mse", "raw_mse"):
            loss, _ = layer_alignment_loss(predicted, target, kind=kind)
            expected = 0.0 if kind != "raw_mse" else 0.25
            self.assertAlmostEqual(float(loss.item()), expected, places=6)
        contrastive, _ = layer_alignment_loss(predicted, target, kind="margin_contrastive", margin=0.2)
        self.assertGreaterEqual(float(contrastive.item()), 0.0)

    def test_zero_vectors_and_all_zero_weights_do_not_nan(self) -> None:
        zeros = {0: torch.zeros(2, 3, dtype=torch.float16, requires_grad=True)}
        for kind in ("cosine", "normalized_mse", "raw_mse", "margin_contrastive"):
            loss, _ = layer_alignment_loss(zeros, zeros, kind=kind, sample_weights=torch.zeros(2))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(loss.item()), 0.0)

    def test_supervision_policies_produce_expected_weights(self) -> None:
        labels = ["harmful", "harmless", "other"]
        self.assertIsNone(supervision_weights(labels, mode="all"))
        torch.testing.assert_close(supervision_weights(labels, mode="harmful_only"), torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(
            supervision_weights(labels, mode="label_weighted", harmful_weight=1.0, harmless_weight=0.5),
            torch.tensor([1.0, 0.5, 0.0]),
        )

    def test_legacy_cosine_api_delegates_without_changing_weighted_metric(self) -> None:
        predicted = {0: torch.tensor([[1.0, 0.0], [1.0, 0.0]])}
        target = {0: torch.tensor([[1.0, 0.0], [-1.0, 0.0]])}
        loss, metrics = cosine_layer_alignment_loss(
            predicted,
            target,
            sample_weights=torch.tensor([1.0, 0.0]),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(metrics[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
