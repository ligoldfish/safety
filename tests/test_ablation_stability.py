from __future__ import annotations

import unittest

import torch

from src.ablations.stability import (
    bootstrap_indices,
    layer_jaccard,
    principal_angles,
    projection_overlap,
)


class AblationStabilityTests(unittest.TestCase):
    def test_identical_and_orthogonal_subspaces_have_known_metrics(self) -> None:
        left = torch.eye(4)[:, :2]
        same = left.clone()
        orthogonal = torch.eye(4)[:, 2:]
        torch.testing.assert_close(principal_angles(left, same), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            principal_angles(left, orthogonal), torch.full((2,), torch.pi / 2), atol=1e-6, rtol=0
        )
        self.assertAlmostEqual(projection_overlap(left, same), 1.0, places=6)
        self.assertAlmostEqual(projection_overlap(left, orthogonal), 0.0, places=6)

    def test_layer_jaccard_handles_empty_sets_explicitly(self) -> None:
        self.assertEqual(layer_jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertEqual(layer_jaccard(set(), set()), 1.0)

    def test_bootstrap_indices_are_reproducible_and_draw_specific(self) -> None:
        first = bootstrap_indices(10, draws=20, seed=42)
        self.assertEqual(len(first), 20)
        self.assertTrue(all(tuple(index.shape) == (10,) for index in first))
        for index in first:
            self.assertTrue(bool(((index >= 0) & (index < 10)).all().item()))
        second = bootstrap_indices(10, draws=20, seed=42)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], first[1]))


if __name__ == "__main__":
    unittest.main()
