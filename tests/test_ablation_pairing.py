from __future__ import annotations

import unittest

import torch

from src.ablations.strategies.pairing import linear_cka_matrix, pair_layers


class AblationPairingTests(unittest.TestCase):
    def test_relative_and_same_index_modes(self) -> None:
        layers = (0, 3, 7)
        self.assertEqual(pair_layers(layers, teacher_layers=8, student_layers=4, mode="relative_depth"), (0, 1, 3))
        self.assertEqual(pair_layers(layers, teacher_layers=8, student_layers=4, mode="same_index_clamped"), (0, 3, 3))

    def test_cka_nearest_uses_max_similarity_per_teacher(self) -> None:
        cka = torch.tensor([[0.1, 0.8, 0.2], [0.9, 0.2, 0.1]])
        self.assertEqual(pair_layers((2, 6), teacher_layers=8, student_layers=3, mode="cka_nearest", cka=cka), (1, 0))

    def test_random_permutation_is_reproducible(self) -> None:
        first = pair_layers((0, 2, 4), teacher_layers=6, student_layers=5, mode="random_permutation", seed=8)
        self.assertEqual(first, pair_layers((0, 2, 4), teacher_layers=6, student_layers=5, mode="random_permutation", seed=8))
        self.assertNotEqual(first, pair_layers((0, 2, 4), teacher_layers=6, student_layers=5, mode="random_permutation", seed=9))
        self.assertEqual(len(first), len(set(first)))

    def test_linear_cka_identifies_equivalent_representation(self) -> None:
        teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        equivalent = teacher @ torch.tensor([[0.0, 2.0], [-2.0, 0.0]])
        unrelated = torch.tensor([[1.0], [-1.0], [1.0]])
        matrix = linear_cka_matrix(
            {2: teacher},
            {0: unrelated, 1: equivalent},
            teacher_key_layers=(2,),
        )
        self.assertEqual(tuple(matrix.shape), (1, 2))
        self.assertGreater(float(matrix[0, 1]), float(matrix[0, 0]))
        self.assertAlmostEqual(float(matrix[0, 1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
