from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_jsonl, write_json, write_jsonl


class AtomicIoTests(unittest.TestCase):
    def test_json_and_jsonl_replace_without_leaving_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "value.json"
            jsonl_path = root / "rows.jsonl"
            write_json(json_path, {"version": 1})
            write_json(json_path, {"version": 2})
            write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(read_jsonl(jsonl_path), [{"row": 1}, {"row": 2}])
            self.assertEqual(tuple(root.glob(".*.tmp")), ())


if __name__ == "__main__":
    unittest.main()
