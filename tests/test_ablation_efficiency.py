from __future__ import annotations

import tempfile
import time
import unittest
import json
import os
import sys
from pathlib import Path

from src.ablations.efficiency import (
    StageProfiler,
    append_efficiency_record,
    phase_for_script,
    run_profiled_subprocess,
    summarize_efficiency,
)


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

    def test_pipeline_scripts_map_to_the_six_paper_phases(self) -> None:
        expected = {
            "01_extract_hidden_states.py": "extract",
            "02_analyze_teacher_layers.py": "subspace",
            "03_build_teacher_safe_subspace.py": "subspace",
            "04_pair_layers.py": "semantic_basis",
            "05_build_semantic_bases.py": "semantic_basis",
            "06_project_teacher_safe_component.py": "decompose",
            "07_decompose_teacher_semantics.py": "decompose",
            "08_recompose_student_targets.py": "recompose",
            "09_train_student_semalign.py": "train",
        }
        self.assertEqual({name: phase_for_script(name) for name in expected}, expected)
        self.assertIsNone(phase_for_script("12_eval_baseline_suite.py"))

    def test_profiled_subprocess_records_real_wall_disk_and_nonfabricated_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "generated.bin"
            returncode, record = run_profiled_subprocess(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,time; pathlib.Path(r'%s').write_bytes(b'x'*31); time.sleep(.15)"
                    % output,
                ],
                cwd=root,
                env=os.environ.copy(),
                stage="extract",
                script="01_extract_hidden_states.py",
                output_root=root,
                cell_id="cell-real",
                device_count=2,
            )
            log = root / "runtime.jsonl"
            append_efficiency_record(log, record)
            payload = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(returncode, 0)
        self.assertGreater(record.wall_seconds, 0.0)
        self.assertGreaterEqual(record.disk_delta_bytes, 31)
        self.assertEqual(record.script, "01_extract_hidden_states.py")
        self.assertIn(record.memory_measurement, {"process_tree_rss", "process_rss", "unavailable"})
        if record.memory_measurement == "unavailable":
            self.assertIsNone(record.peak_memory_bytes)
        else:
            self.assertGreater(record.peak_memory_bytes, 0)


if __name__ == "__main__":
    unittest.main()
