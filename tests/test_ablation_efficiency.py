from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.ablations.efficiency import StageProfiler, summarize_efficiency


class AblationEfficiencyTests(unittest.TestCase):
    def test_profile_has_stable_cost_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profiler = StageProfiler("extract", output_root=root, device_count=2)
            profiler.start()
            (root / "artifact.bin").write_bytes(b"x" * 17)
            profiler.update_peak_memory(4096)
            time.sleep(0.002)
            record = profiler.finish(exit_code=0)
        self.assertGreater(record.wall_seconds, 0.0)
        self.assertEqual(record.peak_memory_bytes, 4096)
        self.assertEqual(record.disk_delta_bytes, 17)
        self.assertAlmostEqual(record.device_hours, record.wall_seconds * 2 / 3600.0)
        self.assertEqual(record.exit_code, 0)

    def test_summary_rejects_mixed_cells(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = StageProfiler("a", output_root=root, cell_id="one").profile_noop()
            second = StageProfiler("b", output_root=root, cell_id="two").profile_noop()
        with self.assertRaisesRegex(ValueError, "same cell"):
            summarize_efficiency([first, second])


if __name__ == "__main__":
    unittest.main()
