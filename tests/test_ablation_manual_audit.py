from __future__ import annotations

import unittest

from src.ablations.manual_audit import build_blind_packet, import_double_annotations


class AblationManualAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"sample_id": f"x{i}", "method": "ours" if i % 2 else "sft", "stratum": "harmful" if i < 4 else "benign", "prompt": f"p{i}", "response": f"r{i}"}
            for i in range(8)
        ]

    def test_packet_is_stratified_blind_and_seeded(self) -> None:
        packet, key = build_blind_packet(self.rows, per_stratum=2, seed=4)
        self.assertEqual(packet, build_blind_packet(list(reversed(self.rows)), per_stratum=2, seed=4)[0])
        self.assertEqual(len(packet), 4)
        self.assertTrue(all("method" not in row and row["annotation"] == "" for row in packet))
        self.assertEqual({row["blind_id"] for row in packet}, set(key))

    def test_double_import_rejects_missing_duplicate_and_unknown_labels(self) -> None:
        packet, key = build_blind_packet(self.rows, per_stratum=2, seed=4)
        rater_a = [{"blind_id": row["blind_id"], "annotation": "safe"} for row in packet]
        rater_b = [{"blind_id": row["blind_id"], "annotation": "unsafe"} for row in packet]
        imported = import_double_annotations(key, rater_a, rater_b, allowed_labels={"safe", "unsafe"})
        self.assertEqual(len(imported), len(packet))
        self.assertTrue(all(row["rater_a"] == "safe" for row in imported))
        with self.assertRaises(ValueError):
            import_double_annotations(key, rater_a[:-1], rater_b, allowed_labels={"safe", "unsafe"})
        with self.assertRaises(ValueError):
            import_double_annotations(key, rater_a + [rater_a[0]], rater_b, allowed_labels={"safe", "unsafe"})


if __name__ == "__main__":
    unittest.main()
