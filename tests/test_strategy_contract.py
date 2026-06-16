"""파이프라인 계약 회귀 테스트 (run_dashboard.sh 구조 보호).

등록된 모든 전략을 합성 데이터로 ``BacktestService.run``에 통과시켜, 새 전략이
공통 대시보드 파이프라인 계약(StrategyArtifacts 스키마 · run_name 규칙 ·
필수 아티팩트)을 깨뜨리면 즉시 실패하도록 만든다. 전략을 추가하면 자동으로
그 전략이 이 테스트에 포함된다 — CLAUDE.md 2~4절 참조.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.storage import RunStore
from trading_lab.strategies import get_handler, list_strategies

# {몇번째}_{전략이름}_{타입}_{세부타입}_{YYDDHH}
RUN_NAME_RE = re.compile(
    r"^\d+_[0-9A-Za-z가-힣_-]+_(주식|크립토|합성)_[0-9A-Za-z가-힣_-]+_\d{6}$"
)

# StrategyArtifacts.trades 가 반드시 노출해야 하는 컬럼 (DASHBOARD_GUIDE 1절).
REQUIRED_TRADE_COLUMNS = (
    "direction", "entry_time", "entry_price", "exit_time",
    "exit_price", "net_return", "exit_reason",
)
# 대시보드 지표 행이 직접 읽는 키.
REQUIRED_METRIC_KEYS = (
    "trades", "hit_rate", "total_return", "sharpe", "max_drawdown",
)


def _enabled_strategies():
    return [d for d in list_strategies() if d.enabled]


class StrategyPipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # var_dir()는 호출 시점에 환경변수를 읽으므로 모듈 reload 없이 격리된다.
        self._prev_var = os.environ.get("TRADING_LAB_VAR")
        os.environ["TRADING_LAB_VAR"] = self._tmp.name
        self.addCleanup(self._restore_var)
        self.store = RunStore(Path(self._tmp.name) / "db.sqlite3")
        self.service = BacktestService(self.store)

    def _restore_var(self) -> None:
        if self._prev_var is None:
            os.environ.pop("TRADING_LAB_VAR", None)
        else:
            os.environ["TRADING_LAB_VAR"] = self._prev_var

    def test_at_least_one_strategy_registered(self) -> None:
        self.assertTrue(_enabled_strategies(), "활성 전략이 하나도 없습니다.")

    def test_every_strategy_runs_through_dashboard_pipeline(self) -> None:
        for definition in _enabled_strategies():
            with self.subTest(strategy=definition.strategy_id):
                run_id = self.service.run(BacktestRequest(
                    strategy_id=definition.strategy_id,
                    symbol="RANDOM",
                    phase="validation",
                    chart_type="random",
                    synthetic=True,
                ))
                run = self.store.get_run(run_id)
                self.assertIsNotNone(run)
                self.assertEqual(
                    run["status"], "succeeded", msg=run.get("error")
                )
                # run_name 규칙 (var/runs/<run_name>/ 폴더명으로도 쓰인다).
                self.assertRegex(run["run_name"], RUN_NAME_RE)
                # 대시보드 결과 화면이 요구하는 아티팩트가 모두 기록됐는지.
                kinds = {a["kind"] for a in run["artifacts"]}
                for kind in ("forecast", "trades", "equity", "metrics"):
                    self.assertIn(kind, kinds, f"{kind} 아티팩트 누락")
                # 기록된 아티팩트 경로가 실제로 존재하는지.
                for artifact in run["artifacts"]:
                    self.assertTrue(
                        Path(artifact["path"]).exists(),
                        f"누락된 파일: {artifact['path']}",
                    )

    def test_handlers_emit_contract_compliant_artifacts(self) -> None:
        for definition in _enabled_strategies():
            with self.subTest(strategy=definition.strategy_id):
                handler = get_handler(definition.strategy_id)
                config = json.loads(
                    definition.config_path.read_text(encoding="utf-8")
                )
                raw = handler.load_data("RANDOM", config, synthetic=True)
                artifacts = handler.build_artifacts(
                    raw, config, symbol="RANDOM",
                    phase="validation", bars_per_year=8760,
                )
                # forecast: 최소 close 컬럼 + DatetimeIndex.
                self.assertIn("close", artifacts.forecast.columns)
                # equity: 비어있지 않고 첫 값이 양의 유한값(1.0 기준 정규화).
                self.assertFalse(artifacts.equity.empty)
                self.assertGreater(float(artifacts.equity.iloc[0]), 0.0)
                # trades: 체결이 있을 때 필수 컬럼을 모두 노출.
                if not artifacts.trades.empty:
                    for column in REQUIRED_TRADE_COLUMNS:
                        self.assertIn(column, artifacts.trades.columns)
                # metrics: 대시보드가 읽는 키가 모두 존재(값은 None 허용).
                for key in REQUIRED_METRIC_KEYS:
                    self.assertIn(key, artifacts.metrics)


if __name__ == "__main__":
    unittest.main()
