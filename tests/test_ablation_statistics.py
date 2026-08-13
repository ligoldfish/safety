from __future__ import annotations

import unittest

from src.ablations.statistics import (
    cohen_kappa,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
)


class AblationStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_requires_aligned_ids(self) -> None:
        ids = ["a", "b", "c", "d"]
        result = paired_bootstrap([1, 1, 0, 0], [0, 1, 0, 1], ids, ids, draws=500, seed=7)
        self.assertEqual(result, paired_bootstrap([1, 1, 0, 0], [0, 1, 0, 1], ids, ids, draws=500, seed=7))
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mean_difference"], 0.0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 0], [0, 1], ["a", "b"], ["b", "a"], draws=10, seed=1)

    def test_mcnemar_exact_known_table(self) -> None:
        result = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
        self.assertEqual(result["b"], 3)
        self.assertEqual(result["c"], 0)
        self.assertAlmostEqual(result["p_value"], 0.25)

    def test_holm_adjust_preserves_input_order(self) -> None:
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_cohen_kappa_known_labels(self) -> None:
        self.assertAlmostEqual(cohen_kappa(["safe", "unsafe"], ["safe", "unsafe"]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)
        with self.assertRaises(ValueError):
            cohen_kappa([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
