from __future__ import annotations

import json
import importlib.util
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.ablations.benchmarks import BenchmarkRequest, DecodeConfig, preflight_benchmark
from src.ablations.catalog import load_catalog
from src.ablations.planner import build_catalog_plan
from src.ablations.runner import (
    AblationRunner,
    RunnerContext,
    RunnerError,
    compile_cell_commands,
    executable_handlers,
    phasef_updates_for_cell,
    validate_completion,
)
from tests.fairness_evidence import attach_validation_evidence


ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "configs" / "ablations" / "catalog.yaml")


def _cell(experiment_id: str, **axes):
    plan = build_catalog_plan(CATALOG, output_root="/persistent/out", scope="all")
    return next(
        cell
        for cell in plan.cells
        if cell.experiment_id == experiment_id
        and all(cell.axes.get(key) == value for key, value in axes.items())
    )


class AblationCompileTests(unittest.TestCase):
    def test_p002_cells_share_phase1_but_keep_cell_owned_phasef_outputs(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_cache_staging_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def stage(cell_id: str, method: str):
                args = SimpleNamespace(
                    output_dir=str(root / "cells" / cell_id),
                    pair="qwen35_9b_to_08b",
                    dataset="pan",
                    device="npu",
                    teacher_variant="",
                    execution_profile="formal",
                    foundation_cache_root=str(root / "foundation"),
                    experiment_id="P0-02",
                    cell_id=cell_id,
                    disable_dataset_overrides=False,
                )
                return module._stage_configs(
                    args,
                    {},
                    {
                        "seed": 42 if method == "ours" else 44,
                        "target.mode": "semantic" if method == "ours" else "random_same_norm",
                    },
                    {},
                )

            phase1_a, phasef_a = stage("cell-a", "ours")
            phase1_b, phasef_b = stage("cell-b", "random")
            p1_a = yaml.safe_load(phase1_a.read_text(encoding="utf-8"))
            p1_b = yaml.safe_load(phase1_b.read_text(encoding="utf-8"))
            pf_a = yaml.safe_load(phasef_a.read_text(encoding="utf-8"))
            pf_b = yaml.safe_load(phasef_b.read_text(encoding="utf-8"))

        self.assertEqual(p1_a["extraction"]["output_root"], p1_b["extraction"]["output_root"])
        self.assertNotEqual(pf_a["output"]["output_root"], pf_b["output"]["output_root"])
        self.assertTrue(pf_a["output"]["output_root"].endswith("training_cells\\cell-a") or pf_a["output"]["output_root"].endswith("training_cells/cell-a"))

    def test_canary_staging_uses_one_epoch_and_short_sequences(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_canary_staging_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                output_dir=str(Path(td) / "cell"),
                pair="qwen35_9b_to_08b",
                dataset="pan",
                device="npu",
                teacher_variant="",
                execution_profile="canary",
                foundation_cache_root="",
                experiment_id="P0-02",
                cell_id="canary-0",
                disable_dataset_overrides=False,
            )
            phase1_path, phasef_path = module._stage_configs(args, {}, {}, {})
            phase1 = yaml.safe_load(phase1_path.read_text(encoding="utf-8"))
            phasef = yaml.safe_load(phasef_path.read_text(encoding="utf-8"))

        self.assertEqual(phase1["extraction"]["max_length"], 512)
        self.assertEqual(phasef["optim"]["epochs"], 1)
        self.assertEqual(phasef["optim"]["max_length"], 512)
        self.assertEqual(phasef["optim"]["max_new_tokens"], 32)

    def test_canary_profile_isolated_from_formal_completion_contract(self) -> None:
        command = compile_cell_commands(
            CATALOG,
            _cell("P0-02", dataset="pan", method="ours", seed=42),
            RunnerContext(
                ROOT,
                Path("/state"),
                "python",
                "npu",
                0,
                execution_profile="canary",
            ),
        )[0]
        self.assertEqual(command.completion_artifacts, ("canary_manifest.json",))
        self.assertIn("--canary", command.argv)
        self.assertEqual(
            command.argv[command.argv.index("--execution-profile") + 1], "canary"
        )

    def test_formal_training_command_receives_foundation_cache_root(self) -> None:
        command = compile_cell_commands(
            CATALOG,
            _cell("P0-02", dataset="pan", method="ours", seed=42),
            RunnerContext(
                ROOT,
                Path("/state"),
                "python",
                "npu",
                0,
                foundation_cache_root=Path("/persistent/foundation-cache"),
            ),
        )[0]
        self.assertEqual(
            command.argv[command.argv.index("--foundation-cache-root") + 1],
            str(Path("/persistent/foundation-cache").resolve()),
        )

    def test_analysis_command_receives_declared_paths_from_asset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "assets.json"
            manifest.write_text(json.dumps({"aligned_sample_predictions": "/data/pairs.jsonl"}), encoding="utf-8")
            context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0, asset_manifest=manifest)
            command = compile_cell_commands(CATALOG, _cell("P0-04", seed=42), context)[0]
        spec_arg = next(token for token in command.argv if token.startswith("--cell-spec="))
        spec = json.loads(spec_arg.split("=", 1)[1])
        self.assertEqual(
            spec["inputs"]["aligned_sample_predictions"],
            str((manifest.parent / Path("/data/pairs.jsonl")).resolve()),
        )

    def test_worker_anchors_relative_manifest_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "assets" / "pairs.jsonl"
            data.parent.mkdir()
            data.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps({"aligned_sample_predictions": {"path": "assets/pairs.jsonl", "kind": "file"}}),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, root / "state", "python", "npu", 0, asset_manifest=manifest)
            command = compile_cell_commands(CATALOG, _cell("P0-04", seed=42), context)[0]
        spec = json.loads(next(x for x in command.argv if x.startswith("--cell-spec=")).split("=", 1)[1])
        self.assertEqual(spec["inputs"]["aligned_sample_predictions"], str(data.resolve()))

    def test_compiled_cell_uses_the_same_expanded_manifest_path_as_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "pan-predictions.jsonl"
            data.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "pan_predictions": {
                            "path": "${SAFETY_DATA_ROOT}/pan-predictions.jsonl",
                            "kind": "file",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"SAFETY_DATA_ROOT": str(root)}, clear=False):
                command = compile_cell_commands(
                    CATALOG,
                    _cell("P1-18", grouping="attack_family"),
                    RunnerContext(
                        ROOT,
                        root / "state",
                        "python",
                        "npu",
                        0,
                        asset_manifest=manifest,
                    ),
                )[0]
            spec = json.loads(
                next(x for x in command.argv if x.startswith("--cell-spec=")).split("=", 1)[1]
            )
        self.assertEqual(spec["inputs"]["pan_predictions"], str(data.resolve()))

    def test_worker_extracts_paths_from_strict_asset_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "aligned_sample_predictions": {
                            "path": "/data/pairs.jsonl",
                            "kind": "file",
                        }
                    }
                ),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0, asset_manifest=manifest)
            command = compile_cell_commands(CATALOG, _cell("P0-04", seed=42), context)[0]
        spec_arg = next(token for token in command.argv if token.startswith("--cell-spec="))
        spec = json.loads(spec_arg.split("=", 1)[1])
        self.assertEqual(
            spec["inputs"]["aligned_sample_predictions"],
            str((manifest.parent / Path("/data/pairs.jsonl")).resolve()),
        )

    def test_training_command_receives_declared_manifest_inputs_with_anchored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "assets" / "search.jsonl"
            ledger.parent.mkdir()
            ledger.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps({"search_ledger": {"path": "assets/search.jsonl", "kind": "file"}}),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, root / "state", "python", "npu", 0, asset_manifest=manifest)
            command = compile_cell_commands(
                CATALOG,
                _cell("P0-07", dataset="pan", config="global", method="ours"),
                context,
            )[0]
        spec_arg = next(token for token in command.argv if token.startswith("--cell-spec="))
        spec = json.loads(spec_arg.split("=", 1)[1])
        self.assertEqual(spec["inputs"]["search_ledger"], str(ledger.resolve()))

    def test_all_catalog_handlers_are_executable(self) -> None:
        self.assertEqual(
            executable_handlers(),
            {definition.handler for definition in CATALOG.experiments.values()},
        )

    def test_representation_strategy_is_forwarded_to_extraction_as_json(self) -> None:
        cell = _cell("P1-08", mode="first_4_generated_mean")
        context = RunnerContext(
            project_root=ROOT,
            state_root=Path("/persistent/state"),
            python_executable="python with spaces",
            device="npu",
            device_id=3,
        )
        command = compile_cell_commands(CATALOG, cell, context)[0]
        self.assertEqual(command.argv[0], "python with spaces")
        stage_flag = next(arg for arg in command.argv if arg.startswith("--phase1-stage-extras="))
        extras = json.loads(stage_flag.split("=", 1)[1])
        self.assertEqual(
            extras["extract"],
            ["--representation-mode", "first_4_generated_mean"],
        )
        self.assertEqual(command.argv[command.argv.index("--device-id") + 1], "3")

    def test_each_core_strategy_reaches_its_own_phase(self) -> None:
        cases = (
            ("P1-03", {"draw": 4}, "subspace", "--subspace-mode", "random_orthogonal"),
            ("P1-04", {"bridge_mode": "ridge"}, "bridge", "--bridge-mode", "ridge"),
            ("P1-07", {"mode": "cka_nearest"}, "pairing", "--pairing-mode", "cka_nearest"),
            ("P1-09", {"top_m": 512}, "decompose", "--top-k", "512"),
        )
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for experiment_id, axes, phase, flag, expected in cases:
            with self.subTest(experiment_id=experiment_id, axes=axes):
                command = compile_cell_commands(CATALOG, _cell(experiment_id, **axes), context)[0]
                encoded = next(arg for arg in command.argv if arg.startswith("--phase1-stage-extras="))
                extras = json.loads(encoded.split("=", 1)[1])
                self.assertEqual(extras[phase][extras[phase].index(flag) + 1], expected)

    def test_teacher_scale_cells_select_the_matching_model_pair(self) -> None:
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        expected = {
            "same_size_base": "qwen3_4b_to_06b",
            "safety_tuned": "qwen3_4b_to_06b",
            "qwen3_4b": "qwen3_4b_to_06b",
            "qwen3_8b": "qwen3_8b_to_06b",
        }
        for teacher, pair in expected.items():
            with self.subTest(teacher=teacher):
                command = compile_cell_commands(CATALOG, _cell("P2-03", teacher=teacher), context)[0]
                self.assertEqual(command.argv[command.argv.index("--pair") + 1], pair)
                if teacher in {"same_size_base", "safety_tuned"}:
                    self.assertEqual(
                        command.argv[command.argv.index("--teacher-variant") + 1],
                        teacher,
                    )
                else:
                    self.assertNotIn("--teacher-variant", command.argv)

    def test_cross_tokenizer_cells_use_a_real_cross_family_pair(self) -> None:
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        command = compile_cell_commands(
            CATALOG,
            _cell("P2-04", bridge_mode="token_string"),
            context,
        )[0]
        pair = command.argv[command.argv.index("--pair") + 1]
        self.assertEqual(pair, "qwen3_8b_to_llama32_1b")
        self.assertTrue((ROOT / "configs" / f"{pair}_phase1_npu.yaml").is_file())
        self.assertTrue((ROOT / "configs" / f"{pair}_phaseF_npu.yaml").is_file())

    def test_teacher_registry_paths_respect_the_persistent_model_root(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_cell_backend_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                output_dir=str(root / "out"),
                pair="qwen3_4b_to_06b",
                device="npu",
                teacher_variant="same_size_base",
            )
            model_root = root / "persistent-models"
            with patch.dict(os.environ, {"SAFETY_MODEL_ROOT": str(model_root)}):
                phase1_path, _ = module._stage_configs(args, {}, {})
            import yaml

            phase1 = yaml.safe_load(phase1_path.read_text(encoding="utf-8"))
        self.assertEqual(
            phase1["models"]["teacher"]["path"],
            str((model_root / "teacher-controls" / "same-size-base").resolve()),
        )

    def test_staged_pan_source_respects_the_persistent_data_root(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_cell_pan_path_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                output_dir=str(root / "out"),
                pair="qwen35_9b_to_08b",
                device="npu",
                teacher_variant="",
            )
            data_root = root / "persistent-data"
            with patch.dict(os.environ, {"SAFETY_DATA_ROOT": str(data_root)}):
                phase1_path, _ = module._stage_configs(args, {}, {})
            payload = yaml.safe_load(phase1_path.read_text(encoding="utf-8"))
        self.assertEqual(
            Path(payload["dataset"]["pan_repo_dir"]),
            (data_root / "external" / "safety-residual-space").resolve(),
        )

    def test_every_train_cell_compiles_only_supported_phasef_fields(self) -> None:
        plan = build_catalog_plan(CATALOG, output_root="/persistent/out", scope="all")
        allowed = {
            "seed",
            "target.mode",
            "target.representation_mode",
            "target.loss_kind",
            "target.layer_loss_policy",
            "target.harmful_layer_weight",
            "target.harmless_layer_weight",
            "optim.layer_loss_weight",
            "lora.rank",
            "lora.placement",
            "inputs.max_samples_per_label",
        }
        for cell in plan.cells:
            if CATALOG.experiments[cell.experiment_id].execution_kind.value != "train":
                continue
            with self.subTest(cell=cell.cell_id, experiment=cell.experiment_id):
                self.assertLessEqual(set(phasef_updates_for_cell(cell)), allowed)

    def test_wildjailbreak_method_axis_changes_the_actual_phasef_objective(self) -> None:
        expected = {
            "sft1": {"target.mode": "semantic", "optim.layer_loss_weight": 0.0},
            "random": {"target.mode": "random_same_norm", "optim.layer_loss_weight": 0.25},
            "ours": {"target.mode": "semantic", "optim.layer_loss_weight": 0.25},
        }
        for method, subset in expected.items():
            with self.subTest(method=method):
                updates = phasef_updates_for_cell(
                    _cell("P0-06", config="override", curation="off", method=method)
                )
                self.assertEqual({key: updates[key] for key in subset}, subset)

    def test_fairness_method_axis_changes_the_actual_phasef_objective(self) -> None:
        expected = {
            "sft1": {"target.mode": "semantic", "optim.layer_loss_weight": 0.0},
            "random": {"target.mode": "random_same_norm", "optim.layer_loss_weight": 0.25},
            "ours": {"target.mode": "semantic", "optim.layer_loss_weight": 0.25},
        }
        for method, subset in expected.items():
            with self.subTest(method=method):
                updates = phasef_updates_for_cell(
                    _cell("P0-07", dataset="pan", config="global", method=method)
                )
                self.assertEqual({key: updates[key] for key in subset}, subset)

    def test_fairness_cells_disable_implicit_dataset_overrides_for_both_policies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "search.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps({"search_ledger": {"path": str(ledger), "kind": "file"}}),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, root / "state", "python", "npu", 0, asset_manifest=manifest)
            for dataset, config in (("pan", "global"), ("wildjailbreak", "validation_selected")):
                with self.subTest(dataset=dataset, config=config):
                    command = compile_cell_commands(
                        CATALOG,
                        _cell("P0-07", dataset=dataset, config=config, method="ours"),
                        context,
                    )[0]
                    self.assertIn("--disable-dataset-overrides", command.argv)

    def test_worker_applies_validation_selected_winner_before_staging(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_fairness_worker_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "search.jsonl"
            rows = []
            for method in ("sft1", "random", "ours"):
                rows.append(
                    {
                        "trial_id": f"{method}-winner",
                        "dataset": "wildguardmix",
                        "config": "validation_selected",
                        "method": method,
                        "selection_split": "validation",
                        "selected": True,
                        "validation_metric": 0.8,
                        "hyperparameters": {
                            "top_k": 7,
                            "energy_threshold": 0.9,
                            "rank_cap": 32,
                            "layer_loss_weight": 0.0 if method == "sft1" else 0.5,
                            "epochs": 3,
                        },
                    }
                )
            attach_validation_evidence(rows, Path(td))
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
            args = SimpleNamespace(
                experiment_id="P0-07",
                cell_spec=json.dumps(
                    {
                        "experiment_id": "P0-07",
                        "axes": {
                            "dataset": "wildguardmix",
                            "config": "validation_selected",
                            "method": "ours",
                        },
                        "inputs": {"search_ledger": str(ledger)},
                    }
                ),
                phase1_updates="{}",
                phasef_updates=json.dumps(
                    {"target.mode": "semantic", "optim.layer_loss_weight": 0.25}
                ),
                phase1_stage_extras="{}",
            )
            raw_spec, phase1_updates, phasef_updates, stage_extras = (
                module._prepare_training_configuration(args)
            )
        self.assertEqual(phase1_updates, {})
        self.assertEqual(phasef_updates["optim.layer_loss_weight"], 0.5)
        self.assertEqual(phasef_updates["optim.epochs"], 3)
        self.assertEqual(stage_extras["analyze"], ["--top-k", "7"])
        self.assertEqual(
            stage_extras["subspace"],
            ["--energy-threshold", "0.9", "--rank-cap", "32"],
        )
        self.assertEqual(
            raw_spec["fairness_configuration"]["selected_trial_id"],
            "ours-winner",
        )
        self.assertEqual(
            raw_spec["fairness_configuration"]["search_ledger_sha256"],
            ledger_hash,
        )

    def test_wildjailbreak_pair_axis_selects_every_declared_backend_pair(self) -> None:
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for pair in CATALOG.formal_pairs:
            with self.subTest(pair=pair):
                command = compile_cell_commands(
                    CATALOG,
                    _cell(
                        "P0-06",
                        pair=pair,
                        config="override",
                        curation="strict",
                        method="ours",
                    ),
                    context,
                )[0]
                self.assertEqual(command.argv[command.argv.index("--pair") + 1], pair)

    def test_wildjailbreak_curation_axis_reaches_curation_script(self) -> None:
        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_wjb_curation_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        calls = []
        module._run_script = lambda script, argv, **kwargs: calls.append((script, list(argv)))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase1 = root / "phase1.yaml"
            phase1.write_text(
                "dataset:\n  curation_mode: minimal\nmodels:\n  teacher:\n    path: /models/teacher\n",
                encoding="utf-8",
            )
            module._invoke_phase1_curation(
                baseline_name="wildjailbreak",
                processed_dir=root / "processed",
                phase1_yaml=phase1,
                dry_run=True,
                env_overrides={},
            )
        command = calls[0][1]
        self.assertEqual(command[command.index("--mode") + 1], "minimal")

    def test_safety_training_cells_use_cell_owned_derived_data_directories(self) -> None:
        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_cell_data_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first_config = root / "first" / "phase1.yaml"
            second_config = root / "second" / "phase1.yaml"
            for config in (first_config, second_config):
                config.parent.mkdir(parents=True)
                config.write_text(
                    yaml.safe_dump(
                        {"extraction": {"output_root": str(config.parent / "pipeline" / "phase1")}}
                    ),
                    encoding="utf-8",
                )
            first = module._resolve_safety_full_roots(
                baseline_name="wildjailbreak",
                device="npu",
                cell_id="cell-one",
                phase1_config_path=str(first_config),
                phasef_config_path="",
            )
            second = module._resolve_safety_full_roots(
                baseline_name="wildjailbreak",
                device="npu",
                cell_id="cell-two",
                phase1_config_path=str(second_config),
                phasef_config_path="",
            )
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[0], first[2].parent / "processed")
        self.assertEqual(second[0], second[2].parent / "processed")

    def test_matched_control_updates_survive_into_the_staged_backend_yaml(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_cell_staging_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        signatures = {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for method in ("sft1", "random", "ours"):
                cell = _cell("P0-02", method=method, seed=42)
                args = SimpleNamespace(
                    output_dir=str(root / method),
                    pair="qwen35_9b_to_08b",
                    device="npu",
                    teacher_variant="",
                )
                _, phasef_path = module._stage_configs(
                    args, {}, phasef_updates_for_cell(cell)
                )
                import yaml

                payload = yaml.safe_load(phasef_path.read_text(encoding="utf-8"))
                signatures[method] = (
                    payload["target"]["mode"],
                    payload["optim"]["layer_loss_weight"],
                    payload["seed"],
                )
        self.assertEqual(
            signatures,
            {
                "sft1": ("semantic", 0.0, 42),
                "random": ("random_same_norm", 0.25, 42),
                "ours": ("semantic", 0.25, 42),
            },
        )

    def test_no_projection_control_keeps_bridge_mapped_semantic_targets(self) -> None:
        updates = phasef_updates_for_cell(_cell("P1-02", subspace_mode="none"))
        self.assertEqual(updates.get("target.mode", "semantic"), "semantic")

    def test_bridge_control_uses_the_same_mode_for_fit_and_recompose(self) -> None:
        context = RunnerContext(ROOT, Path("/state"), "python", "npu", 0)
        for experiment_id, axes in (
            ("P1-04", {"bridge_mode": "ridge"}),
            ("P2-04", {"bridge_mode": "embedding_nearest"}),
        ):
            with self.subTest(experiment=experiment_id):
                command = compile_cell_commands(CATALOG, _cell(experiment_id, **axes), context)[0]
                encoded = next(arg for arg in command.argv if arg.startswith("--phase1-stage-extras="))
                extras = json.loads(encoded.split("=", 1)[1])
                self.assertEqual(extras["bridge"], ["--bridge-mode", axes["bridge_mode"]])
                self.assertEqual(extras["recompose"], ["--bridge-mode", axes["bridge_mode"]])

    def test_oneclick_forwards_every_stage_extra_to_the_correct_phase(self) -> None:
        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_ablation_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        calls = []
        module._run_script = lambda script, argv, **kwargs: calls.append((script, list(argv)))
        module._run_phase1_precompute(
            Path("phase one.yaml"),
            smoke=False,
            dry_run=True,
            stage_extras={
                "extract": ["--representation-mode", "mean_prompt"],
                "analyze": ["--selection-mode", "effect_only"],
                "subspace": ["--subspace-mode", "none"],
                "pairing": ["--pairing-mode", "cka_nearest"],
                "bridge": ["--bridge-mode", "ridge"],
                "decompose": ["--top-k", "32"],
                "recompose": ["--bridge-mode", "ridge"],
            },
        )
        by_script = {}
        for script, argv in calls:
            by_script.setdefault(script, []).append(argv)
        self.assertIn("--representation-mode", by_script["01_extract_hidden_states.py"][0])
        self.assertIn("effect_only", by_script["02_analyze_teacher_layers.py"][0])
        self.assertIn("none", by_script["03_build_teacher_safe_subspace.py"][0])
        self.assertIn("cka_nearest", by_script["04_pair_layers.py"][0])
        self.assertIn("ridge", by_script["05_build_semantic_bases.py"][0])
        self.assertIn("32", by_script["07_decompose_teacher_semantics.py"][0])
        self.assertIn("ridge", by_script["08_recompose_student_targets.py"][0])

    def test_safety_full_explicit_configs_preserve_cell_outputs_and_shared_source_data(self) -> None:
        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_safety_roots_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cell_phase1 = root / "outputs" / "cell" / "pipeline" / "phase1"
            cell_phasef = cell_phase1 / "training"
            phase1_path = root / "phase1.yaml"
            phasef_path = root / "phaseF.yaml"
            phase1_path.write_text(
                "extraction:\n  output_root: " + json.dumps(str(cell_phase1)) + "\n",
                encoding="utf-8",
            )
            phasef_path.write_text(
                "output:\n  output_root: " + json.dumps(str(cell_phasef)) + "\n",
                encoding="utf-8",
            )
            data_root = root / "persistent-data"
            with patch.dict(os.environ, {"SAFETY_DATA_ROOT": str(data_root)}):
                processed, pan, phase1_root, phasef_root = module._resolve_safety_full_roots(
                    baseline_name="c5",
                    device="npu",
                    cell_id="cell-123",
                    phase1_config_path=str(phase1_path),
                    phasef_config_path=str(phasef_path),
                )
        self.assertEqual(processed, (cell_phase1.parent / "processed").resolve())
        self.assertEqual(pan, (data_root / "processed").resolve())
        self.assertEqual(phase1_root, cell_phase1.resolve())
        self.assertEqual(phasef_root, cell_phasef.resolve())

    def test_oneclick_profiles_each_internal_paper_phase_when_cell_log_is_declared(self) -> None:
        from src.ablations.efficiency import StageEfficiency

        path = ROOT / "scripts" / "15_run_oneclick.py"
        spec = importlib.util.spec_from_file_location("oneclick_profile_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        calls = []
        records = []

        def fake_profile(command, **kwargs):
            calls.append((command, kwargs))
            return 0, StageEfficiency(
                1, "cell-9", kwargs["stage"], 1.0, 123, 4, 1, 1 / 3600, 0,
                script=kwargs["script"], memory_measurement="process_tree_rss",
            )

        module.run_profiled_subprocess = fake_profile
        module.append_efficiency_record = lambda path, record: records.append((path, record))
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {
                "SAFETY_ABLATION_RUNTIME_LOG": str(Path(td) / "runtime.jsonl"),
                "SAFETY_ABLATION_PROFILE_OUTPUT_ROOT": str(Path(td) / "pipeline"),
                "SAFETY_ABLATION_CELL_ID": "cell-9",
                "SAFETY_ABLATION_DEVICE_COUNT": "1",
            },
        ):
            module._run_script("05_build_semantic_bases.py", [], dry_run=False)
        self.assertEqual(calls[0][1]["stage"], "semantic_basis")
        self.assertEqual(calls[0][1]["cell_id"], "cell-9")
        self.assertEqual(records[0][1].script, "05_build_semantic_bases.py")

    def test_training_backend_declares_cell_owned_runtime_log(self) -> None:
        path = ROOT / "scripts" / "30_run_ablation_cell.py"
        spec = importlib.util.spec_from_file_location("ablation_profile_env_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(output_dir=str(Path(td) / "artifacts"), cell_id="cell-42")
            environment = module._training_environment(args)
        self.assertTrue(environment["SAFETY_ABLATION_RUNTIME_LOG"].endswith("phase_runtime_logs.jsonl"))
        profile_root = Path(environment["SAFETY_ABLATION_PROFILE_OUTPUT_ROOT"])
        self.assertEqual(profile_root.parts[-2:], ("pipeline", "phase1"))
        self.assertEqual(environment["SAFETY_ABLATION_CELL_ID"], "cell-42")

    def test_benchmark_missing_assets_is_blocked_and_decode_is_shared(self) -> None:
        decode = DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=1024)
        with tempfile.TemporaryDirectory() as td:
            request = BenchmarkRequest(
                name="mmlu",
                asset_path=Path(td) / "missing.jsonl",
                decode=decode,
            )
            report = preflight_benchmark(request)
        self.assertEqual(report.status, "BLOCKED")
        self.assertEqual(report.decode, decode)

    def test_evaluation_cells_dispatch_to_the_real_evaluation_worker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "assets.json"
            intervention_model = root / "model"
            intervention_model.mkdir()
            inputs = root / "inputs"
            inputs.mkdir()
            artifact = root / "subspace.pt"
            artifact.write_bytes(b"x")
            manifest.write_text(
                json.dumps(
                    {
                        "subspace_artifact": str(artifact),
                        "intervention_data": str(inputs),
                        "intervention_model": str(intervention_model),
                    }
                ),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, root / "state", "python", "npu", 2, asset_manifest=manifest)
            command = compile_cell_commands(
                CATALOG,
                _cell("P2-02", layers="random", sign=-1, strength=0.5),
                context,
            )[0]
        self.assertIn("--evaluation-handler", command.argv)
        self.assertEqual(command.argv[command.argv.index("--evaluation-handler") + 1], "causal_intervention")
        self.assertEqual(command.argv[command.argv.index("--device-id") + 1], "2")

    def test_cross_corpus_dispatches_to_evaluation_and_requires_independent_judge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("common", "wildguard"):
                (root / name).mkdir()
            registry = root / "checkpoints.jsonl"
            registry.write_text("{}\n", encoding="utf-8")
            manifest = root / "assets.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoint_registry": {"path": str(registry), "kind": "file"},
                        "common_test": str(root / "common"),
                        "wildguard_model": {"path": str(root / "wildguard"), "kind": "model"},
                    }
                ),
                encoding="utf-8",
            )
            context = RunnerContext(ROOT, root / "state", "python", "npu", 0, asset_manifest=manifest)
            command = compile_cell_commands(
                CATALOG,
                _cell("P0-08", test_suite="pan_heldout"),
                context,
            )[0]
        self.assertIn("--evaluation-handler", command.argv)
        spec = json.loads(next(x for x in command.argv if x.startswith("--cell-spec=")).split("=", 1)[1])
        self.assertEqual(spec["inputs"]["checkpoint_registry"], str(registry.resolve()))
        self.assertIn("wildguard_model", spec["inputs"])


class AblationRunnerStateTests(unittest.TestCase):
    def test_real_runner_blocks_before_execution_when_assets_are_missing(self) -> None:
        cell = _cell("P0-07", dataset="pan", config="global", method="ours")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "assets.json"
            manifest.write_text("{}\n", encoding="utf-8")
            called = []
            runner = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, root / "state", "python", "npu", 0, asset_manifest=manifest),
                executor=lambda *args, **kwargs: called.append(args),
                enforce_preflight=True,
            )
            status = runner.run_cell(cell)
        self.assertEqual(status["state"], "BLOCKED")
        self.assertIn("search_ledger", status["reason"])
        self.assertEqual(called, [])

    def test_ready_manifest_allows_execution_and_is_part_of_fingerprint(self) -> None:
        cell = _cell("P1-11", layer_loss_weight=0.25)
        definition = CATALOG.experiments[cell.experiment_id]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phasef = root / "phasef"
            phasef.mkdir()
            manifest = root / "assets.json"
            manifest.write_text(json.dumps({"phasef_data": str(phasef)}), encoding="utf-8")
            state = root / "state"

            def success(command, *, cwd, env):
                out = state / cell.cell_id / "artifacts"
                out.mkdir(parents=True, exist_ok=True)
                for name in definition.completion_artifacts:
                    (out / name).write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            runner = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, state, "python", "npu", 0, asset_manifest=manifest),
                executor=success,
                enforce_preflight=True,
            )
            self.assertEqual(runner.run_cell(cell)["state"], "COMPLETED")
    def test_completion_validation_rejects_empty_or_invalid_structured_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cell_dir = Path(td)
            artifacts = cell_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "bad.json").write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "invalid JSON"):
                validate_completion(cell_dir, ["bad.json"])
            (artifacts / "bad.json").write_text("{}\n", encoding="utf-8")
            (artifacts / "empty.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "empty"):
                validate_completion(cell_dir, ["bad.json", "empty.jsonl"])

    def test_success_failure_dry_run_and_recovery(self) -> None:
        cell = _cell("P1-11", layer_loss_weight=0.25)
        definition = CATALOG.experiments[cell.experiment_id]
        with tempfile.TemporaryDirectory() as td:
            state_root = Path(td) / "state"
            context = RunnerContext(ROOT, state_root, "python", "npu", 0)
            calls = []

            def success(command, *, cwd, env):
                calls.append((command, cwd, env))
                out = state_root / cell.cell_id / "artifacts"
                out.mkdir(parents=True, exist_ok=True)
                for name in definition.completion_artifacts:
                    (out / name).write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            runner = AblationRunner(CATALOG, context, executor=success)
            dry = runner.run_cell(cell, dry_run=True)
            self.assertNotEqual(dry["state"], "COMPLETED")

            # A dry-run ledger is intentionally immutable as a real run; use a
            # fresh state root to prove a real execution completes.
            context = RunnerContext(ROOT, Path(td) / "real", "python", "npu", 0)
            state_root = context.state_root
            runner = AblationRunner(CATALOG, context, executor=success)
            complete = runner.run_cell(cell)
            self.assertEqual(complete["state"], "COMPLETED")
            self.assertTrue(calls)
            self.assertIsInstance(calls[-1][0], list)

            fail_root = Path(td) / "failure"
            failed_once = {"value": True}

            def flaky(command, *, cwd, env):
                if failed_once["value"]:
                    failed_once["value"] = False
                    return subprocess.CompletedProcess(command, 9)
                out = fail_root / cell.cell_id / "artifacts"
                out.mkdir(parents=True, exist_ok=True)
                for name in definition.completion_artifacts:
                    (out / name).write_text("{}\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            recovery = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, fail_root, "python", "npu", 0),
                executor=flaky,
            )
            with self.assertRaisesRegex(RunnerError, "exit code 9"):
                recovery.run_cell(cell)
            self.assertEqual(recovery.status(cell)["state"], "FAILED")
            self.assertEqual(recovery.run_cell(cell)["state"], "COMPLETED")

    def test_unknown_cell_and_missing_completion_artifacts_fail_closed(self) -> None:
        cell = _cell("P1-13", loss_kind="cosine")
        with tempfile.TemporaryDirectory() as td:
            runner = AblationRunner(
                CATALOG,
                RunnerContext(ROOT, Path(td), "python", "npu", 0),
                executor=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
            )
            with self.assertRaisesRegex(RunnerError, "completion artifact"):
                runner.run_cell(cell)
            self.assertEqual(runner.status(cell)["state"], "FAILED")
            with self.assertRaisesRegex(RunnerError, "unknown cell"):
                runner.select_cell([cell], "not-a-cell")


if __name__ == "__main__":
    unittest.main()
