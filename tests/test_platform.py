from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_lab.artifacts import ArtifactWriter
from trading_lab.execution import DisabledBrokerAdapter, OrderRequest
from trading_lab.models import RunRecord, utc_now
from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.storage import RunStore
from trading_lab.strategies import get_strategy, list_strategies


class RunStoreTests(unittest.TestCase):
    def test_run_lifecycle_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "runs.sqlite3")
            record = RunRecord(
                run_id="run-1",
                strategy_id="h72-price-v1",
                mode="backtest",
                status="created",
                symbol="TEST",
                phase="validation",
                created_at=utc_now(),
                run_number=1,
                run_name="1_랜덤_랜덤_261200",
                chart_type="random",
                chart_detail="랜덤",
            )
            store.create_run(record)
            artifact = root / "metrics.json"
            artifact.write_text("{}", encoding="utf-8")
            store.add_artifact("run-1", "metrics", artifact)
            store.add_event("run-1", "test", "event")
            store.update_status(
                "run-1", "succeeded", metrics={"trades": 1}
            )

            run = store.get_run("run-1")
            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual(run["status"], "succeeded")
            self.assertEqual(run["run_name"], "1_랜덤_랜덤_261200")
            self.assertEqual(run["chart_type"], "random")
            self.assertEqual(run["metrics"], {"trades": 1})
            self.assertEqual(run["artifacts"][0]["kind"], "metrics")
            self.assertEqual(run["events"][0]["event_type"], "test")

    def test_existing_runs_table_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        strategy_id TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT,
                        metrics_json TEXT
                    )
                    """
                )
            store = RunStore(path)
            with store.connect() as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(runs)")
                }
            self.assertTrue({
                "run_number", "run_name", "chart_type", "chart_detail"
            }.issubset(columns))

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "runs.sqlite3")
            with self.assertRaises(ValueError):
                store.create_run(RunRecord(
                    run_id="bad",
                    strategy_id="h72-price-v1",
                    mode="backtest",
                    status="unknown",
                    symbol="TEST",
                    phase="validation",
                    created_at=utc_now(),
                ))


class ArtifactTests(unittest.TestCase):
    def test_writer_creates_run_directory_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("trading_lab.artifacts.runs_dir", return_value=Path(directory)):
                writer = ArtifactWriter("run-2")
                path = writer.write_json("manifest.json", {"run_id": "run-2"})
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8"))["run_id"],
                    "run-2",
                )


class BrokerTests(unittest.TestCase):
    def test_disabled_broker_rejects_orders(self):
        adapter = DisabledBrokerAdapter()
        with self.assertRaisesRegex(RuntimeError, "live trading is disabled"):
            adapter.submit_order(OrderRequest("BTC-USD", "buy", 1.0))
        self.assertEqual(adapter.positions(), [])


class StrategyTests(unittest.TestCase):
    def test_registered_strategies_are_active_and_not_live_eligible(self):
        ids = [item.strategy_id for item in list_strategies()]
        self.assertIn("h72-price-v1", ids)
        self.assertIn("di-kalman-mw-v1", ids)
        for strategy in list_strategies():
            with self.subTest(strategy=strategy.strategy_id):
                self.assertTrue(strategy.enabled)
                self.assertFalse(strategy.live_eligible)
                self.assertTrue(strategy.config_path.exists())


class RunNamingTests(unittest.TestCase):
    def test_run_name_uses_number_strategy_type_detail_and_yyddhh(self):
        name = BacktestService._run_name(
            12, "h72-price-v1", "crypto", "BTC", "2026-06-12T14:30:00+00:00"
        )
        self.assertEqual(name, "12_h72-price-v1_크립토_BTC_261214")

    def test_run_name_labels_synthetic_chart(self):
        name = BacktestService._run_name(
            3, "di-kalman-mw-v1", "random", "랜덤", "2026-06-12T09:00:00+00:00"
        )
        self.assertEqual(name, "3_di-kalman-mw-v1_합성_랜덤_261209")

    def test_crypto_detail_removes_quote_currency(self):
        self.assertEqual(
            BacktestService._chart_detail("BTC-USD", "crypto", None), "BTC"
        )


class ServiceValidationTests(unittest.TestCase):
    def test_test_phase_is_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "runs.sqlite3")
            service = BacktestService(store)
            with self.assertRaisesRegex(ValueError, "test phase is locked"):
                service.run(BacktestRequest(
                    strategy_id="h72-price-v1",
                    symbol="TEST",
                    phase="test",
                    synthetic=True,
                ))


if __name__ == "__main__":
    unittest.main()
