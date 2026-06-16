from __future__ import annotations

import json
import platform
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from .artifacts import ArtifactWriter
from .models import RunRecord, utc_now
from .paths import ROOT
from .reporting import build_equity_html, build_markdown_report
from .storage import RunStore
from .strategies import get_handler, get_strategy, list_strategies
from .ui.presentation import (
    account_value_series,
    build_trade_overview,
    build_trade_report,
)


CHART_TYPE_LABELS = {
    "crypto": "크립토",
    "stock": "주식",
    "random": "합성",
}


@dataclass(frozen=True)
class BacktestRequest:
    strategy_id: str
    symbol: str
    phase: str = "validation"
    chart_type: str | None = None
    chart_detail: str | None = None
    bars_per_year: int = 8760
    initial_capital: float = 10_000.0
    csv_path: Path | None = None
    synthetic: bool = False
    config_overrides: dict[str, Any] | None = None


class BacktestService:
    def __init__(self, store: RunStore | None = None):
        self.store = store or RunStore()
        for strategy in list_strategies():
            self.store.register_strategy(
                strategy.strategy_id,
                strategy.version,
                strategy.description,
                enabled=strategy.enabled,
                live_eligible=strategy.live_eligible,
            )

    def run(self, request: BacktestRequest) -> str:
        strategy = get_strategy(request.strategy_id)
        self._validate_request(request, strategy.enabled)

        run_id = uuid.uuid4().hex
        run_number = self.store.next_run_number()
        chart_type = request.chart_type or (
            "random" if request.synthetic else "crypto"
        )
        chart_detail = self._chart_detail(
            request.symbol, chart_type, request.chart_detail
        )
        created_at = utc_now()
        run_name = self._run_name(
            run_number, strategy.strategy_id, chart_type, chart_detail, created_at
        )
        writer = ArtifactWriter(run_id, dir_name=run_name)
        record = RunRecord(
            run_id=run_id,
            run_number=run_number,
            run_name=run_name,
            strategy_id=strategy.strategy_id,
            mode="backtest",
            status="created",
            symbol=request.symbol,
            chart_type=chart_type,
            chart_detail=chart_detail,
            phase=request.phase,
            created_at=created_at,
        )
        self.store.create_run(record)
        self.store.update_status(run_id, "running")
        self.store.add_event(run_id, "run_started", "Backtest started")

        config = json.loads(strategy.config_path.read_text(encoding="utf-8"))
        if request.config_overrides:
            config.update(request.config_overrides)
        manifest = {
            "run_id": run_id,
            "run_number": run_number,
            "run_name": run_name,
            "strategy_id": strategy.strategy_id,
            "mode": "backtest",
            "symbol": request.symbol,
            "chart_type": chart_type,
            "chart_detail": chart_detail,
            "phase": request.phase,
            "initial_capital": request.initial_capital,
            "created_at": record.created_at,
            "source": self._source_name(request),
            "provenance": self._provenance(),
            "config": config,
        }
        self._register(run_id, "manifest", writer.write_json("manifest.json", manifest))
        self._register(
            run_id, "config", writer.write_json("config_snapshot.json", config)
        )

        try:
            handler = get_handler(strategy.strategy_id)
            raw = handler.load_data(
                request.symbol,
                config,
                csv_path=request.csv_path,
                synthetic=request.synthetic,
            )
            self.store.add_event(
                run_id, "data_loaded", f"Loaded {len(raw)} bars",
                payload={"bars": len(raw)},
            )
            artifacts = handler.build_artifacts(
                raw,
                config,
                symbol=request.symbol,
                phase=request.phase,
                bars_per_year=request.bars_per_year,
            )
            forecast = artifacts.forecast
            metadata = artifacts.metadata
            trades = artifacts.trades
            equity_series = artifacts.equity
            metrics = artifacts.metrics
            metrics.setdefault("symbol", request.symbol)
            metrics.setdefault("phase", request.phase)

            self._register(
                run_id, "forecast", writer.write_frame("forecast", forecast)
            )
            self._register(
                run_id, "forecast_metadata",
                writer.write_json("forecast_metadata.json", metadata),
            )
            self._register(
                run_id, "trades", writer.write_frame("trades", trades)
            )
            equity = equity_series.rename("equity").to_frame()
            self._register(run_id, "equity", writer.write_frame("equity", equity))
            account_value = account_value_series(
                equity_series, request.initial_capital
            ).to_frame()
            self._register(
                run_id,
                "account_value",
                writer.write_frame("account_value", account_value),
            )
            trade_report = build_trade_report(
                trades,
                equity_series,
                initial_capital=request.initial_capital,
                horizon=artifacts.horizon,
                execution=str(config.get("execution", "next_open")),
            )
            self._register(
                run_id,
                "trade_report",
                writer.write_frame("trade_report", trade_report),
            )
            for kind, table in (artifacts.extras or {}).items():
                self._register(
                    run_id, kind,
                    writer.write_json(
                        f"{kind}.json", table.to_dict(orient="records")
                    ),
                )
            trade_overview = build_trade_overview(trades)
            metrics.update({
                **trade_overview,
                "initial_capital": float(request.initial_capital),
                "final_account_value": float(account_value.iloc[-1, 0]),
                "profit_abs": float(
                    account_value.iloc[-1, 0] - request.initial_capital
                ),
            })
            self._register(
                run_id, "metrics", writer.write_json("metrics.json", metrics)
            )
            self._register(
                run_id, "report",
                writer.write_text(
                    "report.md",
                    build_markdown_report(
                        run_id, strategy.strategy_id, config, metrics, metadata
                    ),
                ),
            )
            self._register(
                run_id, "equity_chart",
                writer.write_text(
                    "equity.html",
                    build_equity_html(equity_series, request.symbol),
                ),
            )
            self.store.update_status(run_id, "succeeded", metrics=metrics)
            self.store.add_event(
                run_id, "run_succeeded", "Backtest completed",
                payload={"trades": metrics["trades"]},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._register(
                run_id, "error",
                writer.write_text("error.log", traceback.format_exc()),
            )
            self.store.update_status(run_id, "failed", error=error)
            self.store.add_event(run_id, "run_failed", error, level="error")
        return run_id

    @staticmethod
    def _validate_request(request: BacktestRequest, enabled: bool) -> None:
        if not enabled:
            raise ValueError(f"strategy is disabled: {request.strategy_id}")
        if request.phase not in {"validation", "test", "all"}:
            raise ValueError("phase must be validation, test, or all")
        if request.phase == "test":
            raise ValueError("test phase is locked in platform v1")
        if request.chart_type not in {None, "crypto", "stock", "random"}:
            raise ValueError("chart_type must be crypto, stock, or random")
        if request.chart_type == "random" and not request.synthetic:
            raise ValueError("random chart_type requires synthetic data")
        if request.bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")
        if request.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

    @staticmethod
    def _source_name(request: BacktestRequest) -> str:
        if request.synthetic:
            return "synthetic"
        if request.csv_path:
            return str(request.csv_path)
        return "yfinance"

    @staticmethod
    def _chart_detail(
        symbol: str, chart_type: str, requested: str | None,
    ) -> str:
        if chart_type == "random":
            return "랜덤"
        detail = requested or symbol
        if chart_type == "crypto":
            detail = re.sub(r"[-/](USD|USDT)$", "", detail, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "", detail)
        return cleaned or "UNKNOWN"

    @staticmethod
    def _run_name(
        run_number: int,
        strategy_id: str,
        chart_type: str,
        chart_detail: str,
        created_at: str,
    ) -> str:
        timestamp = datetime.fromisoformat(created_at).strftime("%y%d%H")
        type_label = CHART_TYPE_LABELS[chart_type]
        strategy_label = re.sub(r"[^0-9A-Za-z가-힣_-]+", "", strategy_id) or "strategy"
        return (
            f"{run_number}_{strategy_label}_{type_label}_{chart_detail}_{timestamp}"
        )

    def _register(self, run_id: str, kind: str, path: Path) -> None:
        self.store.add_artifact(run_id, kind, path)

    @staticmethod
    def _provenance() -> dict[str, Any]:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": completed.stdout.strip() or None,
        }
