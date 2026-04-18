import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.datasets import load_code_examples
from src.baselines.debug import collect_error_predictions, export_error_predictions


def _make_mbpp_row(task_id: int) -> dict:
    return {
        "task_id": task_id,
        "prompt": f"Write function {task_id}",
        "test_list": [f"assert solve_{task_id}() == {task_id}"],
        "entry_point": f"solve_{task_id}",
        "code": f"def solve_{task_id}():\n    return {task_id}",
    }


class BaselineEvalDebugTests(unittest.TestCase):
    def test_mbpp_test_split_is_normalized_to_standard_500_tasks(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            dataset_path = Path(tmpdir) / "mbpp_all.jsonl"
            rows = [_make_mbpp_row(task_id) for task_id in range(1, 975)]
            with dataset_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            examples = load_code_examples("mbpp", str(dataset_path), split="test")

        self.assertEqual(len(examples), 500)
        self.assertEqual(examples[0].task_id, "11")
        self.assertEqual(examples[-1].task_id, "510")

    def test_collect_and_export_error_predictions(self):
        predictions = [
            {"id": "q1", "correct": True, "generated_text": "A"},
            {"id": "q2", "correct": False, "generated_text": "B"},
        ]
        errors = collect_error_predictions("mmlu", predictions)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["id"], "q2")

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            summary = export_error_predictions(
                tmpdir,
                "mmlu",
                {"status": "ok", "predictions": predictions},
            )
            self.assertIsNotNone(summary)
            self.assertEqual(summary["num_error_samples"], 1)

            debug_path = Path(summary["path"])
            self.assertTrue(debug_path.exists())
            rows = [
                json.loads(line)
                for line in debug_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "q2")


if __name__ == "__main__":
    unittest.main()
