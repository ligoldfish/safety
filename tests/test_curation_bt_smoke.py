"""Tests for the BeaverTails Phase1 consensus prefilter (`_bt_pre_filter`).

Verifies the Q7 cleaning (Phase1-only, via the per-prompt response counts the
BT builder writes into `metadata`, which survive 20_split's passthrough):
  - consensus-harmful kept (has >=1 unsafe response);
  - missing-category / any-True over-labelled harmful with NO unsafe response dropped;
  - over-refusal-bait benign kept (harmless with >=1 safe response);
  - harmless with no safe response dropped;
  - rows lacking the counts dropped.
Also confirms BT is now wired to `strict` mode so the prefilter actually runs.
"""
import unittest

from src.data.curation import DEFAULT_MODE, PRE_FILTERS, _bt_pre_filter


def _rec(label, *, safe=None, unsafe=None, rid="x"):
    meta = {}
    if safe is not None:
        meta["prompt_group_safe_count"] = safe
    if unsafe is not None:
        meta["prompt_group_unsafe_count"] = unsafe
    return {"id": rid, "label": label, "messages": [], "metadata": meta}


class BtPreFilterTest(unittest.TestCase):
    def test_keeps_consensus_harmful_and_orbait_benign(self):
        kept = _bt_pre_filter([
            _rec("harmful", safe=0, unsafe=3, rid="h-consensus"),   # keep
            _rec("harmless", safe=4, unsafe=0, rid="b-orbait"),     # keep (safe response exists)
        ])
        ids = {r["id"] for r in kept}
        self.assertEqual(ids, {"h-consensus", "b-orbait"})

    def test_drops_forced_harmful_without_unsafe(self):
        # missing-category -> forced harmful, or any-True mislabel: no unsafe response.
        kept = _bt_pre_filter([_rec("harmful", safe=5, unsafe=0, rid="h-mislabel")])
        self.assertEqual(kept, [])

    def test_drops_harmless_without_safe(self):
        kept = _bt_pre_filter([_rec("harmless", safe=0, unsafe=2, rid="b-allunsafe")])
        self.assertEqual(kept, [])

    def test_drops_rows_missing_counts(self):
        kept = _bt_pre_filter([
            _rec("harmful", unsafe=3, rid="no-safe-field"),   # safe count missing
            _rec("harmless", safe=2, rid="no-unsafe-field"),  # unsafe count missing
            {"id": "no-meta", "label": "harmful"},            # no metadata at all
        ])
        self.assertEqual(kept, [])

    def test_bt_registered_and_strict(self):
        self.assertIn("beavertails_category", PRE_FILTERS)
        self.assertIs(PRE_FILTERS["beavertails_category"], _bt_pre_filter)
        self.assertEqual(DEFAULT_MODE["beavertails_category"], "strict")
        self.assertEqual(DEFAULT_MODE["beavertails"], "strict")


if __name__ == "__main__":
    unittest.main()
