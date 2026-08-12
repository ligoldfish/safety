from __future__ import annotations

import unittest

from src.ablations.strategies.layers import LayerCandidate, select_layers


class AblationLayerSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = tuple(
            LayerCandidate(layer_idx=index, effect_size=float(index), probe_accuracy=float(7 - index))
            for index in range(8)
        )

    def test_scored_modes_use_requested_metric(self) -> None:
        self.assertEqual(select_layers(self.candidates, k=2, mode="effect_only"), (7, 6))
        self.assertEqual(select_layers(self.candidates, k=2, mode="probe_only"), (0, 1))
        self.assertEqual(select_layers(self.candidates, k=2, mode="effect_probe_sum"), (0, 1))

    def test_structured_layer_controls(self) -> None:
        self.assertEqual(select_layers(self.candidates, k=3, mode="evenly_spaced"), (0, 4, 7))
        self.assertEqual(select_layers(self.candidates, k=3, mode="last_k"), (5, 6, 7))

    def test_random_is_reproducible_and_seed_sensitive(self) -> None:
        first = select_layers(self.candidates, k=4, mode="random_k", seed=123)
        self.assertEqual(first, select_layers(self.candidates, k=4, mode="random_k", seed=123))
        self.assertNotEqual(first, select_layers(self.candidates, k=4, mode="random_k", seed=124))
        self.assertEqual(len(first), len(set(first)))

    def test_invalid_k_and_duplicate_layer_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 <= k"):
            select_layers(self.candidates, k=0, mode="last_k")
        duplicate = self.candidates + (self.candidates[0],)
        with self.assertRaisesRegex(ValueError, "unique"):
            select_layers(duplicate, k=2, mode="last_k")


if __name__ == "__main__":
    unittest.main()
