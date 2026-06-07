"""Tests for per-layer effective-rank K in `build_teacher_safe_subspace`.

Constructs a Delta with a known singular-value spectrum (harmless_mean = 0, so
Delta == harmful_hidden) and checks the energy-threshold rank selection +
rank_cap + the fixed-rank fallback.
"""
import unittest

import torch

from src.features.subspace import build_teacher_safe_subspace


def _diag_hidden(sv):
    # harmful_hidden = diag(sv) -> singular values are exactly |sv| (orthogonal rows).
    return torch.diag(torch.tensor(sv, dtype=torch.float32))


class EffectiveRankTest(unittest.TestCase):
    def setUp(self):
        # sigma = [10, 1, 0.1, 0.01] -> energy = [100, 1, 0.01, 1e-4], total ~= 101.0101
        # cum_ratio ~= [0.9901, 0.99999, ...]
        self.harmful = _diag_hidden([10.0, 1.0, 0.1, 0.01])
        self.harmless = torch.zeros(4, 4, dtype=torch.float32)

    def _build(self, **kw):
        return build_teacher_safe_subspace(
            layer_idx=0, harmful_hidden=self.harmful, harmless_hidden=self.harmless,
            normalize_hidden=False, **kw,
        )

    def test_tau_090_picks_one(self):
        # first component already carries ~99% energy -> K=1
        res = self._build(energy_threshold=0.9, rank_cap=64)
        self.assertEqual(res.k, 1)
        self.assertEqual(tuple(res.basis.shape), (4, 1))

    def test_tau_0999_picks_two(self):
        res = self._build(energy_threshold=0.999, rank_cap=64)
        self.assertEqual(res.k, 2)

    def test_rank_cap_clamps(self):
        # tau=1.0 would want all 4 dims; cap to 2.
        res = self._build(energy_threshold=1.0, rank_cap=2)
        self.assertEqual(res.k, 2)

    def test_fixed_rank_fallback(self):
        res = self._build(energy_threshold=None, k=3)
        self.assertEqual(res.k, 3)


if __name__ == "__main__":
    unittest.main()
