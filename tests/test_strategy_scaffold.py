from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import new_strategy  # noqa: E402

from trading_lab.strategies import list_strategies  # noqa: E402


class RegisteredConfigTests(unittest.TestCase):
    def test_registered_configs_are_well_formed(self):
        # Universal contract: every registered config loads as a dict with a
        # "version" key. The full forecast-template schema (validate_config) is
        # only required of forecast-family strategies (those with "horizon").
        for definition in list_strategies():
            with self.subTest(strategy=definition.strategy_id):
                config = json.loads(
                    definition.config_path.read_text(encoding="utf-8")
                )
                self.assertIsInstance(config, dict)
                self.assertIn("version", config)
                if "horizon" in config:
                    self.assertEqual(new_strategy.validate_config(config), [])


class ScaffoldTests(unittest.TestCase):
    def _scaffold(self, tmp: Path, **kwargs):
        with (
            patch.object(new_strategy, "CONFIG_DIR", tmp / "configs"),
            patch.object(
                new_strategy, "STRATEGY_DOCS", tmp / "docs",
            ),
        ):
            (tmp / "configs").mkdir(parents=True, exist_ok=True)
            (tmp / "docs").mkdir(parents=True, exist_ok=True)
            for name in (
                "STRATEGY_SPEC_TEMPLATE.md", "STRATEGY_TEST_CHECKLIST.md",
            ):
                target = tmp / "docs" / name
                source = ROOT / "docs" / "strategy" / name
                target.write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8"
                )
            with (
                patch.object(
                    new_strategy, "SPEC_TEMPLATE",
                    tmp / "docs" / "STRATEGY_SPEC_TEMPLATE.md",
                ),
                patch.object(
                    new_strategy, "MASTER_CHECKLIST",
                    tmp / "docs" / "STRATEGY_TEST_CHECKLIST.md",
                ),
            ):
                return new_strategy.scaffold("demo-strat-v1", "demo", **kwargs)

    def test_scaffold_creates_valid_artifacts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            paths = self._scaffold(tmp)
            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            self.assertEqual(new_strategy.validate_config(config), [])
            self.assertEqual(config["version"], "demo-strat-v1")
            spec = paths["spec"].read_text(encoding="utf-8")
            self.assertIn("demo-strat-v1", spec)
            self.assertNotIn("{STRATEGY_ID}", spec)
            checklist = paths["checklist"].read_text(encoding="utf-8")
            self.assertIn("Gate 0", checklist)
            self.assertIn("Gate 4", checklist)

    def test_scaffold_refuses_overwrite(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._scaffold(tmp)
            with self.assertRaises(FileExistsError):
                self._scaffold(tmp)
            self._scaffold(tmp, force=True)

    def test_rejects_bad_strategy_id(self):
        with self.assertRaises(ValueError):
            new_strategy.scaffold("Bad_Name", "x")


if __name__ == "__main__":
    unittest.main()
