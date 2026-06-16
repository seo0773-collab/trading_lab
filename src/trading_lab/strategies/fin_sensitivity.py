"""Handler for fin-sensitivity-v1 (finance_plan.txt §19·§20·§24).

scripts/finance_sensitivity 의 데이터→모델→신호 파이프라인을 공통 대시보드 데이터
(yfinance / synthetic / csv 가격 + 분기재무)에 돌리고, 결과를 공통 StrategyArtifacts
계약에 매핑한다. forecast의 보조 컬럼(pred_ret_20d/60d, quality_score, valuation_z,
sens_*)은 대시보드 파형 패널에 자동 노출된다(전략 전용 차트 코드 없음).

분기재무 출처: var/fundamentals/<symbol>.parquet (scripts/fetch_fundamentals.py).
없고 심볼이 합성(RANDOM/SYNTH)이면 합성 재무를 결정적으로 재생성한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.paths import var_dir
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from finance_sensitivity.availability import AVAILABLE_DATE  # noqa: E402
from finance_sensitivity.config import config_from_dict  # noqa: E402
from finance_sensitivity.dataset import build_event_table  # noqa: E402
from finance_sensitivity.fundamentals import feature_columns  # noqa: E402
from finance_sensitivity.macro import make_synthetic_rates  # noqa: E402
from finance_sensitivity.model import rolling_predict  # noqa: E402
from finance_sensitivity.signals import build_trades  # noqa: E402
from finance_sensitivity.synthetic import (  # noqa: E402
    make_synthetic_fundamentals, make_synthetic_ohlcv,
)

_SYNTHETIC_SYMBOLS = {"RANDOM", "SYNTH", "SYNTHETIC"}


class FinSensitivityHandler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        cfg = config_from_dict(config)
        if synthetic:
            return make_synthetic_ohlcv(cfg.synthetic_bars, cfg.seed, cfg=cfg)
        if csv_path is not None:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        return load_yfinance(symbol, cfg.interval, cfg.period)

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        cfg = config_from_dict(config)
        fundamentals = self._load_fundamentals(symbol, raw, cfg)
        market_close = self._load_market(symbol, cfg)
        rates = self._load_rates(symbol, raw, cfg)

        events = build_event_table(
            raw, fundamentals, cfg, market_close=market_close, rates=rates
        )
        predicted = rolling_predict(events, cfg)
        table = predicted["table"]
        trades_all = build_trades(table, raw, cfg, market_close=market_close)

        index = self._naive_index(raw)
        window = self._phase_window(index, phase, cfg)
        forecast = self._forecast_frame(raw, table, cfg, index).loc[window]
        trades = self._slice_trades(trades_all, window)
        equity = self._equity_series(trades, window, forecast["close"])
        metrics = self._metrics(trades, equity, phase)

        n_pred = int(predicted["n_predicted"])
        metadata = {
            "n_events": int(len(table)),
            "n_predicted": n_pred,
            "insufficient_train_data": n_pred == 0,
            "timeframe": cfg.interval,
            "fundamentals_source": self._source_label(symbol),
        }
        extras = {
            "sensitivity_table": self._sensitivity_table(predicted["coef20"]),
            "pred_vs_real": self._pred_vs_real(table),
            "learning_summary": self._learning_summary(table),
            "learning_events": self._learning_events(table, cfg),
        }
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            extras=extras,
        )

    # ----- fundamentals -------------------------------------------------
    def _load_fundamentals(
        self, symbol: str, raw: pd.DataFrame, cfg
    ) -> pd.DataFrame:
        path = var_dir() / "fundamentals" / f"{symbol.upper()}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        if symbol.upper() in _SYNTHETIC_SYMBOLS:
            return make_synthetic_fundamentals(self._naive_index(raw), cfg)
        raise FileNotFoundError(
            f"분기재무 없음: {path}. 먼저 "
            f"`python scripts/fetch_fundamentals.py --symbols {symbol}` 실행."
        )

    @staticmethod
    def _source_label(symbol: str) -> str:
        return "synthetic" if symbol.upper() in _SYNTHETIC_SYMBOLS else "parquet"

    def _load_market(self, symbol: str, cfg) -> pd.Series | None:
        """시장 지수(기본 SPY) 종가 — 초과수익 타깃·시장 필터용. 합성/실패 시 None."""
        if symbol.upper() in _SYNTHETIC_SYMBOLS:
            return None
        try:
            from run_kalman_pipeline import load_yfinance  # noqa: E402
            mkt = load_yfinance(cfg.market_filter.symbol, cfg.interval, cfg.period)
            return mkt["close"]
        except Exception:
            return None

    def _load_rates(self, symbol: str, raw: pd.DataFrame, cfg) -> pd.Series | None:
        """미국 금리(기본 ^IRX) 일별 종가 — 금리 발표 민감도 피처용.

        합성 심볼은 결정적 합성 금리, 실데이터는 yfinance. 로드 실패 시 None을
        반환하면 dataset가 rate 피처를 0으로 채워 파이프라인은 중단되지 않는다.
        """
        if not cfg.use_rate_feature:
            return None
        if symbol.upper() in _SYNTHETIC_SYMBOLS:
            return make_synthetic_rates(self._naive_index(raw), cfg)
        try:
            from run_kalman_pipeline import load_yfinance  # noqa: E402
            rate = load_yfinance(cfg.rate_symbol, cfg.interval, cfg.period)
            return rate["close"]
        except Exception:
            return None

    # ----- frames -------------------------------------------------------
    @staticmethod
    def _naive_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        return idx.tz_localize(None) if idx.tz is not None else idx

    def _forecast_frame(self, raw, table, cfg, index) -> pd.DataFrame:
        frame = pd.DataFrame(index=index)
        for col in ("open", "high", "low", "close"):
            if col in raw:
                frame[col] = np.asarray(raw[col], dtype=float)
        # 이벤트 지표를 진입봉에 놓고 ffill(발표 사이 step 함수) → 보조 자동노출.
        ind_cols = ["pred_ret_20d", "pred_ret_60d", "quality_score",
                    "valuation_z"]
        if cfg.use_rate_feature:
            ind_cols += ["rate_level", "d_rate"]
        ind_cols += [f"sens_{f}" for f in feature_columns(cfg)]
        if not table.empty:
            stamped = table.set_index(
                pd.DatetimeIndex(pd.to_datetime(table["entry_time"]))
            ).sort_index()
            stamped = stamped[~stamped.index.duplicated(keep="last")]
            for col in ind_cols:
                if col in stamped:
                    series = stamped[col].reindex(index, method="ffill")
                    frame[col] = series.to_numpy()
        return frame

    @staticmethod
    def _phase_window(index, phase: str, cfg) -> pd.DatetimeIndex:
        if phase == "all":
            return index
        n = len(index)
        t_end = int(n * cfg.train_frac)
        v_end = int(n * (cfg.train_frac + cfg.validation_frac))
        if phase == "validation":
            return index[t_end:v_end]
        if phase == "test":
            return index[v_end:]
        return index

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window) -> pd.DataFrame:
        if trades.empty:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        lo, hi = window[0], window[-1]
        mask = np.asarray((entries >= lo) & (entries <= hi))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _equity_series(
        trades: pd.DataFrame, window, close: pd.Series
    ) -> pd.Series:
        """평가자산(mark-to-market) 일별 equity — 1.0 기준.

        보유 중에는 종가로 매일 평가손익을 반영(미실현 포함)하고, 청산일에
        실현 net_return(수수료 포함)으로 확정해 다음 구간의 기준 자본으로 쓴다.
        포지션이 없는 구간은 현금으로 평탄하게 유지한다. 단일 종목·중첩 없음.
        """
        idx = pd.DatetimeIndex(window)
        equity = pd.Series(1.0, index=idx, name="equity")
        if trades.empty:
            return equity
        px = pd.Series(np.asarray(close, dtype=float), index=idx).ffill().bfill()
        realized = 1.0
        ordered = trades.sort_values("entry_time")
        for _, tr in ordered.iterrows():
            entry_t = pd.Timestamp(tr["entry_time"])
            exit_t = pd.Timestamp(tr["exit_time"])
            entry_price = float(tr["entry_price"])
            net = float(tr["net_return"])
            # 보유 구간(진입 다음날 ~ 청산일): 종가로 미실현 평가.
            held = idx[(idx > entry_t) & (idx < exit_t)]
            if len(held) and entry_price:
                equity.loc[held] = realized * (px.loc[held].to_numpy() / entry_price)
            realized = realized * (1.0 + net)
            # 청산일 이후(다음 진입 전까지)는 실현 자본으로 평탄 유지.
            equity.loc[idx >= exit_t] = realized
        return equity

    # ----- metrics & extras --------------------------------------------
    @staticmethod
    def _metrics(trades: pd.DataFrame, equity: pd.Series, phase: str) -> dict:
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        hit_rate = float((rets > 0).mean()) if n else None
        total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
        if n > 1 and rets.std(ddof=0) > 0:
            sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(n))
        else:
            sharpe = None
        if len(equity):
            dd = (equity / equity.cummax() - 1.0).min()
            max_drawdown = float(dd)
        else:
            max_drawdown = None
        return {
            "trades": n,
            "hit_rate": hit_rate,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "avg_return": float(rets.mean()) if n else None,
            "phase": phase,
        }

    @staticmethod
    def _sensitivity_table(coef20: pd.DataFrame) -> pd.DataFrame:
        if coef20 is None or coef20.empty:
            return pd.DataFrame(columns=["factor", "sensitivity_mean"])
        feats = [c for c in coef20.columns if c != "available_date"]
        out = coef20[feats].mean().reset_index()
        out.columns = ["factor", "sensitivity_mean"]
        return out

    @staticmethod
    def _pred_vs_real(table: pd.DataFrame) -> pd.DataFrame:
        cols = ["available_date", "pred_ret_20d", "ret_20d",
                "pred_ret_60d", "ret_60d"]
        present = [c for c in cols if c in table.columns]
        if table.empty or not present:
            return pd.DataFrame(columns=cols)
        out = table[present].dropna(subset=["pred_ret_20d"]).copy()
        if AVAILABLE_DATE in out:
            out[AVAILABLE_DATE] = pd.to_datetime(out[AVAILABLE_DATE]).astype(str)
        return out

    @staticmethod
    def _learning_summary(table: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for horizon in (20, 60):
            pred_col = f"pred_ret_{horizon}d"
            real_col = f"ret_{horizon}d"
            if pred_col not in table or real_col not in table:
                continue
            valid = table[[pred_col, real_col]].dropna().astype(float)
            if valid.empty:
                continue
            pred = valid[pred_col]
            real = valid[real_col]
            rows.append({
                "horizon_days": horizon,
                "samples": int(len(valid)),
                "spearman_ic": float(pred.corr(real, method="spearman")),
                "mae": float((real - pred).abs().mean()),
                "direction_accuracy": float(
                    ((pred > 0) == (real > 0)).mean()
                ),
                "mean_predicted_return": float(pred.mean()),
                "mean_actual_return": float(real.mean()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _learning_events(table: pd.DataFrame, cfg) -> pd.DataFrame:
        base = [
            AVAILABLE_DATE, "entry_time", "quality_score", "valuation_z",
            "excluded", "insufficient", "pred_ret_20d", "ret_20d",
            "pred_ret_60d", "ret_60d",
        ]
        columns = list(dict.fromkeys(
            column for column in [*base, *feature_columns(cfg)]
            if column in table.columns
        ))
        if table.empty:
            return pd.DataFrame(columns=columns)
        out = table[columns].copy()
        for column in (AVAILABLE_DATE, "entry_time"):
            if column in out:
                out[column] = pd.to_datetime(out[column]).astype(str)
        return out
