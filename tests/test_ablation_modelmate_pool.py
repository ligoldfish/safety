from __future__ import annotations

import threading
import time
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ablations.catalog import load_catalog
from src.ablations.modelmate_pool import (
    CAMPAIGN_WAVES,
    FINAL_ROUND_ORDER,
    ROUND_SPECS,
    PoolLayout,
    build_shard_command,
    derive_pool_layout,
    run_shard_pool,
    select_round_cells,
)
from src.ablations.planner import build_catalog_plan


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")
COMPLETE_PLAN = build_catalog_plan(CATALOG, output_root="/persistent/cells", scope="full")


def _load_pool_script():
    path = ROOT / "scripts" / "35_modelmate_8card_pool.py"
    spec = importlib.util.spec_from_file_location("modelmate_8card_pool_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_final_gate_script():
    path = ROOT / "scripts" / "36_modelmate_ablation_final_gate.py"
    spec = importlib.util.spec_from_file_location("modelmate_final_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_campaign_script():
    path = ROOT / "scripts" / "37_modelmate_ablation_campaign.py"
    spec = importlib.util.spec_from_file_location("modelmate_campaign_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelMateRoundSelectionTests(unittest.TestCase):
    def test_final_rounds_cover_every_cell_exactly_once(self) -> None:
        expected = {
            "p0-core": 54,
            "p0-wjb": 6,
            "p0-fairness": 12,
            "p0-evaluate": 2,
            "p0-analyze": 154,
            "p0-manual": 3,
            "p1-mechanism": 52,
            "p1-data": 10,
            "p1-evaluate": 5,
            "p1-analyze": 5,
            "p2-generalization": 6,
            "p2-evaluate": 24,
            "p2-analyze": 27,
        }
        seen: set[str] = set()
        for name, count in expected.items():
            with self.subTest(round=name):
                cells = select_round_cells(CATALOG, COMPLETE_PLAN, ROUND_SPECS[name])
                ids = {cell.cell_id for cell in cells}
                self.assertEqual(len(cells), count)
                self.assertFalse(seen & ids)
                seen.update(ids)

        self.assertEqual(tuple(expected), FINAL_ROUND_ORDER)
        self.assertEqual(seen, {cell.cell_id for cell in COMPLETE_PLAN.cells})
        self.assertEqual(len(seen), 360)

    def test_production_rounds_form_one_explicit_dependency_chain(self) -> None:
        self.assertEqual(ROUND_SPECS["p0-smoke"].prerequisites, ())
        previous = "p0-smoke"
        for name in FINAL_ROUND_ORDER:
            with self.subTest(round=name):
                self.assertEqual(ROUND_SPECS[name].prerequisites, (previous,))
            previous = name

    def test_smoke_is_an_isolated_eight_card_pan_canary(self) -> None:
        smoke = select_round_cells(CATALOG, COMPLETE_PLAN, ROUND_SPECS["p0-smoke"])
        core = select_round_cells(CATALOG, COMPLETE_PLAN, ROUND_SPECS["p0-core"])
        self.assertEqual(len(smoke), 8)
        self.assertEqual(ROUND_SPECS["p0-smoke"].state_group, "p0-smoke")
        self.assertEqual(ROUND_SPECS["p0-smoke"].execution_profile, "canary")
        self.assertEqual({cell.axes["dataset"] for cell in smoke}, {"pan"})
        self.assertLessEqual(
            {cell.cell_id for cell in smoke},
            {cell.cell_id for cell in core},
        )

    def test_six_campaign_waves_preserve_every_internal_round(self) -> None:
        self.assertEqual(tuple(CAMPAIGN_WAVES), ("canary", "p0", "p0-manual", "p1", "p2", "final"))
        internal = tuple(
            round_name
            for wave in CAMPAIGN_WAVES.values()
            for round_name in wave.rounds
        )
        self.assertEqual(internal, ("p0-smoke", *FINAL_ROUND_ORDER))
        self.assertTrue(CAMPAIGN_WAVES["final"].final_gate)


class ModelMatePoolLayoutTests(unittest.TestCase):
    def test_sixteen_logical_shards_use_at_most_eight_devices(self) -> None:
        self.assertEqual(
            derive_pool_layout(cell_count=54, requested_shards=16, requested_devices=8),
            PoolLayout(cell_count=54, shard_count=16, max_cells_per_shard=4, device_count=8),
        )

    def test_small_round_does_not_create_empty_shards_or_workers(self) -> None:
        self.assertEqual(
            derive_pool_layout(cell_count=2, requested_shards=16, requested_devices=8),
            PoolLayout(cell_count=2, shard_count=2, max_cells_per_shard=1, device_count=2),
        )

    def test_shard_command_keeps_each_experiment_single_device(self) -> None:
        command = build_shard_command(
            python_executable="python",
            project_root=Path("/snapshot"),
            plan_path=Path("/persistent/plan.jsonl"),
            state_root=Path("/persistent/state"),
            asset_manifest=Path("/persistent/assets.json"),
            layout=PoolLayout(54, 16, 4, 8),
            shard_index=9,
            device="npu",
            device_id=3,
            dry_run=False,
            execution_profile="formal",
            foundation_cache_root=Path("/persistent/foundation-cache"),
        )
        self.assertEqual(command[command.index("--shard-index") + 1], "9")
        self.assertEqual(command[command.index("--shard-count") + 1], "16")
        self.assertEqual(command[command.index("--max-cells") + 1], "4")
        self.assertEqual(command[command.index("--device-id") + 1], "3")
        self.assertEqual(command[command.index("--num-devices") + 1], "1")
        self.assertEqual(command[command.index("--execution-profile") + 1], "formal")
        self.assertEqual(
            command[command.index("--foundation-cache-root") + 1],
            str(Path("/persistent/foundation-cache")),
        )


class ModelMateDynamicPoolTests(unittest.TestCase):
    def test_dynamic_pool_runs_each_shard_once_without_device_overlap(self) -> None:
        lock = threading.Lock()
        active = 0
        peak = 0
        active_devices: set[int] = set()
        calls: list[tuple[int, int]] = []

        def worker(shard_index: int, device_id: int) -> int:
            nonlocal active, peak
            with lock:
                self.assertNotIn(device_id, active_devices)
                active_devices.add(device_id)
                active += 1
                peak = max(peak, active)
                calls.append((shard_index, device_id))
            time.sleep(0.01 if shard_index % 3 else 0.02)
            with lock:
                active -= 1
                active_devices.remove(device_id)
            return 0

        results = run_shard_pool(
            shard_count=16,
            device_ids=tuple(range(8)),
            worker=worker,
            stagger_seconds=0,
        )

        self.assertEqual({result.shard_index for result in results}, set(range(16)))
        self.assertEqual(len(calls), 16)
        self.assertLessEqual(peak, 8)
        self.assertTrue(all(result.returncode == 0 for result in results))

    def test_failure_stops_assigning_pending_shards(self) -> None:
        calls: list[int] = []
        lock = threading.Lock()

        def worker(shard_index: int, device_id: int) -> int:
            del device_id
            with lock:
                calls.append(shard_index)
            if shard_index == 0:
                return 7
            time.sleep(0.05)
            return 0

        results = run_shard_pool(
            shard_count=12,
            device_ids=(0, 1),
            worker=worker,
            stagger_seconds=0.02,
        )

        self.assertTrue(any(result.returncode == 7 for result in results))
        self.assertLess(len(calls), 12)

    def test_first_failure_requests_running_sibling_termination_once(self) -> None:
        notifications: list[str] = []

        def worker(shard_index: int, device_id: int) -> int:
            del device_id
            return 9 if shard_index == 0 else 0

        results = run_shard_pool(
            shard_count=8,
            device_ids=(0, 1),
            worker=worker,
            stagger_seconds=0,
            on_failure=lambda: notifications.append("terminate"),
        )

        self.assertTrue(any(result.returncode == 9 for result in results))
        self.assertEqual(notifications, ["terminate"])


class ModelMatePoolEntrypointTests(unittest.TestCase):
    def test_formal_pool_requires_prepared_offline_data(self) -> None:
        module = _load_pool_script()
        environment = module._runtime_environment(
            Path("/persistent/models"),
            Path("/persistent/data"),
            Path("/persistent/output"),
        )
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["SAFETY_REQUIRE_PREPARED_DATA"], "1")

    def test_parser_accepts_only_known_modelmate_injected_arguments(self) -> None:
        module = _load_pool_script()
        args = module.build_parser().parse_args(
            [
                "--round=p0-smoke",
                "--checkpoint_url=",
                "--data_url=s3://platform-placeholder/dataset",
            ]
        )
        self.assertEqual(args.round, "p0-smoke")
        self.assertEqual(args.checkpoint_url, "")
        self.assertEqual(args.data_url, "s3://platform-placeholder/dataset")

        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(["--unknown-platform-argument=1"])

    def test_worker_environment_isolates_one_scheduler_visible_npu_per_process(self) -> None:
        module = _load_pool_script()
        scheduler = {
            "ASCEND_RT_VISIBLE_DEVICES": "7,3,5,1,6,2,4,0",
            "SAFETY_SENTINEL": "kept",
        }
        mapped = [
            module._worker_environment(scheduler, device="npu", device_id=index)[
                "ASCEND_RT_VISIBLE_DEVICES"
            ]
            for index in range(8)
        ]
        self.assertEqual(mapped, ["7", "3", "5", "1", "6", "2", "4", "0"])
        self.assertEqual(scheduler["ASCEND_RT_VISIBLE_DEVICES"], "7,3,5,1,6,2,4,0")
        self.assertEqual(
            module._worker_environment({}, device="npu", device_id=4)[
                "ASCEND_RT_VISIBLE_DEVICES"
            ],
            "4",
        )
        self.assertNotIn(
            "ASCEND_RT_VISIBLE_DEVICES",
            module._worker_environment({}, device="cpu", device_id=4),
        )

    def test_worker_environment_rejects_unmapped_scheduler_device(self) -> None:
        module = _load_pool_script()
        with self.assertRaisesRegex(RuntimeError, "only 2 visible NPU entries"):
            module._worker_environment(
                {"ASCEND_RT_VISIBLE_DEVICES": "4,5"},
                device="npu",
                device_id=2,
            )

    def test_subprocess_receives_the_isolated_worker_environment(self) -> None:
        module = _load_pool_script()
        captured: dict[str, object] = {}

        class FakeProcess:
            pid = 731

            def wait(self) -> int:
                return 0

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["environment"] = kwargs["env"]
            return FakeProcess()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            executor = module.SubprocessShardExecutor(
                project_root=ROOT,
                round_root=root / "round",
                environment={"ASCEND_RT_VISIBLE_DEVICES": "6,2"},
                plan_path=root / "plan.jsonl",
                state_root=root / "state",
                asset_manifest=root / "assets.json",
                layout=PoolLayout(2, 2, 1, 2),
                python_executable=sys.executable,
                device="npu",
                dry_run=False,
            )
            with patch.object(module.subprocess, "Popen", side_effect=fake_popen):
                result = executor(1, 1)
        self.assertEqual(result, 0)
        self.assertEqual(
            captured["environment"]["ASCEND_RT_VISIBLE_DEVICES"],
            "2",
        )

    def test_preflight_only_writes_auditable_eight_card_smoke_layout(self) -> None:
        module = _load_pool_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            output = root / "outputs"
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            existing_pool_summary = output / "jobs" / "p0-smoke" / "pool-summary.json"
            existing_pool_summary.parent.mkdir(parents=True, exist_ok=True)
            existing_pool_summary.write_text('{"status":"READY","sentinel":true}\n')
            probe = {
                "schema_version": 1,
                "torch_device_count": 8,
                "requested_device_ids": list(range(8)),
                "probes": [],
            }
            with patch.object(module, "_run_logged", return_value=0), patch.object(
                module, "verify_npu_devices", return_value=probe
            ):
                result = module.main(
                    [
                        "--round",
                        "p0-smoke",
                        "--model-root",
                        str(root / "models"),
                        "--data-root",
                        str(root / "datasets"),
                        "--output-root",
                        str(output),
                        "--asset-manifest",
                        str(manifest),
                        "--preflight-only",
                    ]
                )
            summary = json.loads(
                (output / "jobs" / "p0-smoke" / "preflight-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            preserved = json.loads(existing_pool_summary.read_text(encoding="utf-8"))
            metadata = json.loads(
                (output / "jobs" / "p0-smoke" / "job-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(result, 0)
        self.assertEqual(summary["status"], "PREFLIGHT_READY")
        self.assertTrue(preserved["sentinel"])
        self.assertEqual(metadata["layout"]["cell_count"], 8)
        self.assertEqual(metadata["layout"]["shard_count"], 8)
        self.assertEqual(metadata["device_ids"], list(range(8)))

    def test_production_cannot_skip_the_real_device_check(self) -> None:
        module = _load_pool_script()
        with self.assertRaisesRegex(ValueError, "only with --dry-run"):
            module.main(["--skip-device-check"])

    def test_cpu_analysis_round_does_not_import_or_probe_npu(self) -> None:
        module = _load_pool_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            output = root / "outputs"
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with patch.object(module, "_run_logged", return_value=0), patch.object(
                module, "verify_npu_devices", side_effect=AssertionError("unexpected NPU probe")
            ) as probe:
                result = module.main(
                    [
                        "--round",
                        "p0-analyze",
                        "--model-root",
                        str(root / "models"),
                        "--data-root",
                        str(root / "datasets"),
                        "--output-root",
                        str(output),
                        "--asset-manifest",
                        str(manifest),
                        "--preflight-only",
                    ]
                )
        self.assertEqual(result, 0)
        probe.assert_not_called()

    def test_real_round_rejects_missing_or_dry_run_prerequisite(self) -> None:
        module = _load_pool_script()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td).resolve()
            with self.assertRaisesRegex(RuntimeError, "p0-smoke.*missing"):
                module._require_completed_prerequisites(
                    output, ROUND_SPECS["p0-core"], CATALOG, COMPLETE_PLAN
                )

            summary = output / "jobs" / "p0-smoke" / "pool-summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"status": "READY", "dry_run": True}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "real READY run"):
                module._require_completed_prerequisites(
                    output, ROUND_SPECS["p0-core"], CATALOG, COMPLETE_PLAN
                )

            summary.write_text(
                json.dumps(
                    {
                        "status": "READY",
                        "dry_run": False,
                        "expected_cells": 8,
                        "failed_shards": [],
                        "pending_shards": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "status is missing"):
                module._require_completed_prerequisites(
                    output, ROUND_SPECS["p0-core"], CATALOG, COMPLETE_PLAN
                )
            smoke_ids = [
                cell.cell_id
                for cell in select_round_cells(
                    CATALOG, COMPLETE_PLAN, ROUND_SPECS["p0-smoke"]
                )
            ]
            (summary.parent / "status.json").write_text(
                json.dumps(
                    {
                        "cells": [
                            {"cell_id": cell_id, "state": "COMPLETED"}
                            for cell_id in smoke_ids
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            module._require_completed_prerequisites(
                output, ROUND_SPECS["p0-core"], CATALOG, COMPLETE_PLAN
            )

    def test_npu_probe_failure_is_recorded_before_pool_launch(self) -> None:
        module = _load_pool_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            output = root / "outputs"
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with patch.object(module, "_run_logged", return_value=0), patch.object(
                module, "verify_npu_devices", side_effect=RuntimeError("device 3 failed")
            ):
                result = module.main(
                    [
                        "--round",
                        "p0-smoke",
                        "--model-root",
                        str(root / "models"),
                        "--data-root",
                        str(root / "datasets"),
                        "--output-root",
                        str(output),
                        "--asset-manifest",
                        str(manifest),
                        "--preflight-only",
                    ]
                )
            probe = json.loads(
                (output / "jobs" / "p0-smoke" / "device-preflight.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(result, 3)
        self.assertEqual(probe["status"], "FAILED")
        self.assertIn("device 3 failed", probe["error"])

    def test_bootstrap_does_not_override_scheduler_device_visibility(self) -> None:
        content = (ROOT / "boot_safety_8card.sh").read_text(encoding="utf-8")
        self.assertNotIn("export ASCEND_RT_VISIBLE_DEVICES", content)
        self.assertIn('SAFETY_LOGICAL_SHARDS:-16', content)
        self.assertIn('SAFETY_POOL_DEVICES:-8', content)


class ModelMateCampaignTests(unittest.TestCase):
    def test_wave_builds_ordered_internal_round_commands(self) -> None:
        module = _load_campaign_script()
        args = module.build_parser().parse_args(["--wave", "p1"])
        commands = module.build_campaign_commands(args)
        self.assertEqual(
            [command[command.index("--round") + 1] for command in commands],
            ["p1-mechanism", "p1-data", "p1-evaluate", "p1-analyze"],
        )
        self.assertTrue(all("--logical-shards" in command for command in commands))

    def test_wave_stops_after_first_failed_internal_round(self) -> None:
        module = _load_campaign_script()
        args = module.build_parser().parse_args(["--wave", "p0"])
        calls: list[list[str]] = []

        class Result:
            def __init__(self, returncode: int) -> None:
                self.returncode = returncode

        def runner(command, **kwargs):
            del kwargs
            calls.append(command)
            return Result(7 if len(calls) == 2 else 0)

        exit_code, payload = module.run_campaign(args, runner=runner)
        self.assertEqual(exit_code, 7)
        self.assertEqual(len(calls), 2)
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["executed_stage_count"], 2)
        self.assertEqual(payload["planned_stage_count"], 5)

    def test_final_wave_invokes_only_the_360_cell_full_gate(self) -> None:
        module = _load_campaign_script()
        args = module.build_parser().parse_args(["--wave", "final"])
        commands = module.build_campaign_commands(args)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0][1].endswith("36_modelmate_ablation_final_gate.py"))
        self.assertIn("--output-root", commands[0])


class ModelMateFinalGateTests(unittest.TestCase):
    def _write_completed_rounds(self, output: Path) -> None:
        for round_name in FINAL_ROUND_ORDER:
            spec = ROUND_SPECS[round_name]
            cells = select_round_cells(CATALOG, COMPLETE_PLAN, spec)
            root = output / "jobs" / round_name
            root.mkdir(parents=True, exist_ok=True)
            (root / "pool-summary.json").write_text(
                json.dumps(
                    {
                        "status": "READY",
                        "dry_run": False,
                        "expected_cells": len(cells),
                        "failed_shards": [],
                        "pending_shards": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "cells": [
                            {"cell_id": cell.cell_id, "state": "COMPLETED"}
                            for cell in cells
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    def test_final_gate_requires_and_counts_all_360_real_cells(self) -> None:
        module = _load_final_gate_script()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td).resolve()
            self._write_completed_rounds(output)
            result = module.main(["--output-root", str(output)])
            report = json.loads(
                (output / "final-completion.json").read_text(encoding="utf-8")
            )
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["expected_cells"], 360)
        self.assertEqual(report["covered_cells"], 360)
        self.assertEqual(report["training_cells"], 140)

    def test_final_gate_rejects_dry_run_or_noncompleted_round(self) -> None:
        module = _load_final_gate_script()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td).resolve()
            self._write_completed_rounds(output)
            root = output / "jobs" / "p1-data"
            summary = json.loads((root / "pool-summary.json").read_text())
            summary["dry_run"] = True
            (root / "pool-summary.json").write_text(json.dumps(summary) + "\n")
            status = json.loads((root / "status.json").read_text())
            status["cells"][0]["state"] = "FAILED"
            (root / "status.json").write_text(json.dumps(status) + "\n")
            report = module.audit_completion(output)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["covered_cells"], 359)
        self.assertTrue(any("p1-data" in item for item in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
