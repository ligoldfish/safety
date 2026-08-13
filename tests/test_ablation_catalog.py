from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.ablations.catalog import CatalogError, load_catalog
from src.ablations.schema import EXPECTED_EXPERIMENT_IDS


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "ablations" / "catalog.yaml"


class AblationCatalogTests(unittest.TestCase):
    def test_catalog_covers_exactly_all_html_experiment_ids(self) -> None:
        catalog = load_catalog(CATALOG_PATH)
        self.assertEqual(set(catalog.experiments), EXPECTED_EXPERIMENT_IDS)
        self.assertEqual(len(catalog.experiments), 35)

    def test_formal_axes_are_exact_and_stable(self) -> None:
        catalog = load_catalog(CATALOG_PATH)
        self.assertEqual(
            catalog.formal_pairs,
            (
                "qwen35_9b_to_08b",
                "llama31_8b_to_1b",
                "qwen3_8b_to_06b",
                "qwen3_8b_to_4b",
                "qwen3_4b_to_06b",
            ),
        )
        self.assertEqual(
            catalog.formal_datasets,
            (
                "pan",
                "safety_tuned_llamas",
                "coconot",
                "c5",
                "wildjailbreak",
                "wildguardmix",
            ),
        )
        self.assertEqual(catalog.main_methods, ("ours", "sft1", "sft", "distill", "nosft"))

    def test_every_experiment_has_an_executable_contract(self) -> None:
        catalog = load_catalog(CATALOG_PATH)
        for item in catalog.experiments.values():
            with self.subTest(item.id):
                self.assertTrue(item.question.strip())
                self.assertTrue(item.handler.strip())
                self.assertNotEqual(item.handler, "document_only")
                self.assertTrue(item.metrics)
                self.assertTrue(item.completion_artifacts)

    def test_cross_corpus_registry_is_not_aliased_to_single_checkpoint_directory(self) -> None:
        catalog = load_catalog(CATALOG_PATH)
        self.assertIn("checkpoint_registry", catalog.experiments["P0-08"].requires)
        self.assertNotIn("trained_checkpoints", catalog.experiments["P0-08"].requires)
        self.assertIn("trained_checkpoints", catalog.experiments["P1-19"].requires)
        self.assertIn("trained_checkpoints", catalog.experiments["P2-05"].requires)

    def test_duplicate_ids_are_rejected(self) -> None:
        raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
        raw["experiments"].append(dict(raw["experiments"][0]))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "catalog.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "duplicate experiment id"):
                load_catalog(path)

    def test_unknown_strategy_is_rejected(self) -> None:
        raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
        raw["experiments"][0]["overrides"] = {"target.mode": "magic_target"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "catalog.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "unknown target.mode"):
                load_catalog(path)


if __name__ == "__main__":
    unittest.main()
