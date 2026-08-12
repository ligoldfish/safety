from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ablations.ledger import ExperimentLedger, LedgerError, RunState


class AblationLedgerTests(unittest.TestCase):
    def test_valid_state_machine_and_atomic_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = ExperimentLedger(Path(td), "cell-a", config_hash="cfg-a")
            ledger.initialize()
            with ledger.acquire_lock():
                ledger.transition(RunState.READY)
                ledger.transition(RunState.RUNNING)
                ledger.transition(RunState.COMPLETED, artifact_hash="artifact-a")
            state = ledger.read()
            self.assertEqual(state["state"], "COMPLETED")
            self.assertEqual(state["config_hash"], "cfg-a")
            self.assertEqual(state["artifact_hash"], "artifact-a")
            self.assertFalse((Path(td) / "cell-a" / "status.json.tmp").exists())

    def test_dry_run_cannot_be_marked_completed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = ExperimentLedger(Path(td), "cell-a", config_hash="cfg-a")
            ledger.initialize(dry_run=True)
            with ledger.acquire_lock():
                ledger.transition(RunState.READY)
                ledger.transition(RunState.RUNNING)
                with self.assertRaisesRegex(LedgerError, "dry-run"):
                    ledger.transition(RunState.COMPLETED, artifact_hash="fake")

    def test_illegal_transition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = ExperimentLedger(Path(td), "cell-a", config_hash="cfg-a")
            ledger.initialize()
            with ledger.acquire_lock():
                with self.assertRaisesRegex(LedgerError, "illegal state transition"):
                    ledger.transition(RunState.COMPLETED, artifact_hash="fake")

    def test_transition_requires_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = ExperimentLedger(Path(td), "cell-a", config_hash="cfg-a")
            ledger.initialize()
            with self.assertRaisesRegex(LedgerError, "writer lock"):
                ledger.transition(RunState.READY)

    def test_existing_config_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ExperimentLedger(root, "cell-a", config_hash="cfg-a").initialize()
            with self.assertRaisesRegex(LedgerError, "configuration fingerprint"):
                ExperimentLedger(root, "cell-a", config_hash="cfg-b").initialize()

    def test_lock_rejects_second_writer_and_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = ExperimentLedger(Path(td), "cell-a", config_hash="cfg-a")
            ledger.initialize()
            with ledger.acquire_lock():
                with self.assertRaisesRegex(LedgerError, "already locked"):
                    with ledger.acquire_lock():
                        pass
            with ledger.acquire_lock():
                self.assertTrue(ledger.lock_path.exists())
            self.assertFalse(ledger.lock_path.exists())


if __name__ == "__main__":
    unittest.main()
