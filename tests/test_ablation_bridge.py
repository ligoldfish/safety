from __future__ import annotations

import unittest

import torch

from src.ablations.strategies.bridge import (
    apply_linear_bridge,
    fit_orthogonal_procrustes,
    fit_ridge,
    hidden_bridge_targets,
    match_embedding_nearest,
    match_token_strings,
    remap_sparse_coefficients,
    validate_bridge_mode,
)


class AblationBridgeTests(unittest.TestCase):
    def test_ridge_recovers_known_linear_mapping(self) -> None:
        generator = torch.Generator().manual_seed(7)
        teacher = torch.randn(40, 3, generator=generator)
        mapping = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 1.0]])
        student = teacher @ mapping
        fitted = fit_ridge(teacher, student, alpha=1e-8)
        torch.testing.assert_close(fitted, mapping, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(apply_linear_bridge(teacher, fitted), student, atol=1e-5, rtol=1e-5)

    def test_procrustes_recovers_known_orthogonal_mapping(self) -> None:
        generator = torch.Generator().manual_seed(9)
        teacher = torch.randn(50, 3, generator=generator)
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        student = teacher @ q
        fitted = fit_orthogonal_procrustes(teacher, student)
        torch.testing.assert_close(fitted.T @ fitted, torch.eye(3), atol=1e-5, rtol=0)
        torch.testing.assert_close(teacher @ fitted, student, atol=1e-5, rtol=1e-5)

    def test_procrustes_supports_cross_scale_hidden_dimensions(self) -> None:
        generator = torch.Generator().manual_seed(11)
        teacher = torch.randn(80, 5, generator=generator)
        _, _, vh = torch.linalg.svd(teacher.to(torch.float64), full_matrices=False)
        reducer = vh[:3].T.to(torch.float32)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        expected = reducer @ rotation
        student = teacher @ expected
        fitted = fit_orthogonal_procrustes(teacher, student)
        self.assertEqual(tuple(fitted.shape), (5, 3))
        torch.testing.assert_close(fitted.T @ fitted, torch.eye(3), atol=1e-5, rtol=0)
        torch.testing.assert_close(teacher @ fitted, student, atol=1e-5, rtol=1e-5)

    def test_token_string_mapping_audits_special_duplicates_and_unmatched(self) -> None:
        teacher = {"a": 0, "b": 1, "dup": 2, "<s>": 3, "missing": 4}
        student = {"b": 8, "a": 7, "dup": 6, "<s>": 5}
        result = match_token_strings(
            teacher,
            student,
            teacher_special_ids={3},
            student_special_ids={5},
        )
        self.assertEqual(result.teacher_to_student, {0: 7, 1: 8, 2: 6})
        self.assertAlmostEqual(result.coverage, 3 / 4)
        self.assertEqual(result.unmatched_teacher_ids, (4,))
        self.assertEqual(result.conflicts, 0)

    def test_embedding_nearest_reports_conflicts_and_threshold(self) -> None:
        teacher = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = match_embedding_nearest(teacher, student, min_cosine=0.8)
        self.assertEqual(result.teacher_to_student, {0: 0, 1: 0, 2: 1})
        self.assertEqual(result.conflicts, 1)
        self.assertEqual(result.unmatched_teacher_ids, ())

    def test_cross_tokenizer_forbids_vocabulary_index_bridge(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross-tokenizer"):
            validate_bridge_mode("vocabulary", tokenizer_shared=False)
        self.assertEqual(validate_bridge_mode("token_string", tokenizer_shared=False), "token_string")

    def test_sparse_coefficients_are_remapped_and_unmatched_terms_zeroed(self) -> None:
        indices = torch.tensor([[0, 1, 4], [1, 0, 4]])
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mapped_indices, mapped_values, audit = remap_sparse_coefficients(
            indices, values, {0: 7, 1: 8}
        )
        torch.testing.assert_close(mapped_indices, torch.tensor([[7, 8, 0], [8, 7, 0]]))
        torch.testing.assert_close(mapped_values, torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]))
        self.assertEqual(audit, {"total_terms": 6, "matched_terms": 4, "unmatched_terms": 2})

    def test_hidden_bridge_targets_follow_pair_indices_and_teacher_layers(self) -> None:
        safe = {2: torch.tensor([[1.0, 2.0]]), 5: torch.tensor([[3.0, 4.0]])}
        pairs = (
            {"pair_idx": 0, "teacher_layer": 2, "student_layer": 1},
            {"pair_idx": 1, "teacher_layer": 5, "student_layer": 3},
        )
        mappings = {
            0: torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            1: torch.tensor([[2.0], [-1.0]]),
        }
        targets = hidden_bridge_targets(safe, pairs, mappings)
        torch.testing.assert_close(targets[0], torch.tensor([[1.0, 2.0, 3.0]]))
        torch.testing.assert_close(targets[1], torch.tensor([[2.0]]))


if __name__ == "__main__":
    unittest.main()
