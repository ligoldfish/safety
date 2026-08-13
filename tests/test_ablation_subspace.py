from __future__ import annotations

import unittest

import torch
import importlib.util
from pathlib import Path

from src.ablations.strategies.subspace import build_control_subspace, project_with_mode


class AblationSubspaceTests(unittest.TestCase):
    def test_random_basis_is_rank_matched_orthogonal_and_reproducible(self) -> None:
        learned = torch.eye(8)[:, :3]
        first = build_control_subspace(learned, mode="random_orthogonal", seed=42)
        second = build_control_subspace(learned, mode="random_orthogonal", seed=42)
        other = build_control_subspace(learned, mode="random_orthogonal", seed=43)
        self.assertEqual(first.shape, learned.shape)
        torch.testing.assert_close(first.T @ first, torch.eye(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.allclose(first, other))

    def test_none_is_strict_identity_and_learned_projects(self) -> None:
        hidden = torch.tensor([[1.0, 2.0, 3.0]])
        basis = torch.tensor([[1.0], [0.0], [0.0]])
        torch.testing.assert_close(project_with_mode(hidden, basis, mode="none"), hidden)
        torch.testing.assert_close(
            project_with_mode(hidden, basis, mode="learned"), torch.tensor([[1.0, 0.0, 0.0]])
        )

    def test_invalid_basis_rank_and_mode_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            build_control_subspace(torch.empty(3, 0), mode="random_orthogonal", seed=1)
        with self.assertRaisesRegex(ValueError, "unsupported subspace mode"):
            build_control_subspace(torch.eye(3), mode="magic", seed=1)

    def test_phase06_control_projection_preserves_raw_hidden_for_none(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "06_project_teacher_safe_component.py"
        spec = importlib.util.spec_from_file_location("project06", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        hidden = torch.tensor([[1.0, 2.0, 3.0]])
        payload = {"basis": torch.empty(3, 0), "subspace_mode": "none"}
        safe, coeff = module._project_control(hidden, payload)
        torch.testing.assert_close(safe, hidden)
        self.assertEqual(tuple(coeff.shape), (1, 0))


if __name__ == "__main__":
    unittest.main()
