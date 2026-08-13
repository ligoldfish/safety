from __future__ import annotations

import unittest

import torch

from src.ablations.strategies.targets import permute_target_map


class AblationTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = {
            "h1": {0: torch.tensor([1.0])},
            "h2": {0: torch.tensor([2.0])},
            "h3": {0: torch.tensor([5.0])},
            "b1": {0: torch.tensor([3.0])},
            "b2": {0: torch.tensor([4.0])},
            "b3": {0: torch.tensor([6.0])},
        }
        self.labels = {
            "h1": "harmful", "h2": "harmful", "h3": "harmful",
            "b1": "harmless", "b2": "harmless", "b3": "harmless",
        }

    def test_within_label_permutation_preserves_label_source(self) -> None:
        result, manifest = permute_target_map(self.targets, self.labels, mode="within_label_permutation", seed=42)
        self.assertEqual(set(result), set(self.targets))
        for destination, source in manifest.items():
            self.assertEqual(self.labels[destination], self.labels[source])
            torch.testing.assert_close(result[destination][0], self.targets[source][0])

    def test_cross_label_permutation_uses_opposite_label(self) -> None:
        result, manifest = permute_target_map(self.targets, self.labels, mode="cross_label_permutation", seed=42)
        for destination, source in manifest.items():
            self.assertNotEqual(self.labels[destination], self.labels[source])
            torch.testing.assert_close(result[destination][0], self.targets[source][0])

    def test_permutation_is_order_independent_and_seeded(self) -> None:
        reversed_targets = dict(reversed(list(self.targets.items())))
        one = permute_target_map(self.targets, self.labels, mode="within_label_permutation", seed=7)[1]
        two = permute_target_map(reversed_targets, self.labels, mode="within_label_permutation", seed=7)[1]
        three = permute_target_map(self.targets, self.labels, mode="within_label_permutation", seed=8)[1]
        self.assertEqual(one, two)
        self.assertNotEqual(one, three)


if __name__ == "__main__":
    unittest.main()
