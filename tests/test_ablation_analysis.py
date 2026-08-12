from __future__ import annotations

import unittest

from src.ablations.analysis import pan_bucket, spearman_correlation, summarize_corpus_matrix


class AblationAnalysisTests(unittest.TestCase):
    def test_pan_bucket_separates_jailbreak_and_benign(self) -> None:
        self.assertEqual(pan_bucket({"label": "harmful", "method": "PAIR"}), "jailbreak")
        self.assertEqual(pan_bucket({"label": "harmless"}), "benign")
        self.assertEqual(pan_bucket({"label": "harmful"}), "harmful_other")

    def test_corpus_matrix_has_every_requested_cell(self) -> None:
        rows = [
            {"train_corpus": "a", "test_suite": "common", "score": 0.2},
            {"train_corpus": "a", "test_suite": "common", "score": 0.4},
            {"train_corpus": "b", "test_suite": "common", "score": 0.8},
        ]
        matrix = summarize_corpus_matrix(rows, corpora=["a", "b"], suites=["common"])
        self.assertAlmostEqual(matrix["a"]["common"]["mean"], 0.3)
        self.assertEqual(matrix["b"]["common"]["n"], 1)

    def test_spearman_known_monotonic_and_constant(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [4, 5, 9]), 1.0)
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
