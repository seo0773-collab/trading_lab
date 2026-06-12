from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.storage import RunStore


class FailedRunTests(unittest.TestCase):
    def test_failure_keeps_error_artifact_and_database_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "runs.sqlite3")
            service = BacktestService(store)
            with (
                patch("trading_lab.artifacts.runs_dir", return_value=root / "runs"),
                patch(
                    "trading_lab.service.load_market_data",
                    side_effect=RuntimeError("provider unavailable"),
                ),
            ):
                run_id = service.run(BacktestRequest(
                    strategy_id="h72-price-v1",
                    symbol="FAIL",
                ))

            run = store.get_run(run_id)
            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual(run["status"], "failed")
            self.assertIn("provider unavailable", run["error"])
            kinds = {item["kind"] for item in run["artifacts"]}
            self.assertTrue({"manifest", "config", "error"}.issubset(kinds))


if __name__ == "__main__":
    unittest.main()
