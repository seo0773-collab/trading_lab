from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from profile_sizing.engine import performance
from profile_sizing.portfolio import rebalance_dates
from run_kalman_pipeline import load_yfinance
from trading_lab.strategies import get_handler
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler
from trading_lab.ui.presentation import account_value_series, total_invested


# KIS 편도 비용(소수) — 계좌 실조건(2026-07-05 확인, 우대율 반영).
# US = 수수료 0.10%(우대) + 제비용 0.00103% + 환전 편도 0.10%(스프레드 1%×우대 90%).
# KR = 온라인 수수료+제비용(환전 없음). bp 대응 상수: scripts/profile_sizing/config.py.
DEFAULT_KIS_KR_COST = 0.000146527
DEFAULT_KIS_US_COST = 0.0020103


@dataclass(frozen=True)
class IntegerDcaSettings:
    initial_capital: float
    monthly_contribution: float
    cost_kr: float = DEFAULT_KIS_KR_COST
    cost_us: float = DEFAULT_KIS_US_COST
    min_kr_order: float = 50_000.0
    min_us_order_usd: float = 50.0


def supports_integer_dca(config: dict[str, Any]) -> bool:
    if len(config.get("sub_strategies") or []) != 2:
        return False
    currencies = {str(s.get("currency", config.get("base_currency", ""))) for s in config["sub_strategies"]}
    return {"USD", "KRW"}.issubset(currencies) and bool(config.get("fx_symbol"))


def simulate_integer_dca(
    strategy_id: str,
    config: dict[str, Any],
    symbol: str,
    phase_index: pd.Index,
    settings: IntegerDcaSettings,
) -> dict[str, Any]:
    data = _build_leaf_targets(strategy_id, config, symbol)
    phase_index = pd.DatetimeIndex(pd.to_datetime(phase_index))
    idx = pd.DatetimeIndex(data["idx"]).intersection(phase_index).sort_values()
    if len(idx) < 60:
        raise ValueError("not enough overlapping bars for integer execution simulation")
    data = {
        **data,
        "idx": idx,
        "ideal_equity": data["ideal_equity"].reindex(idx).ffill(),
        "target": data["target"].reindex(idx).ffill().fillna(0.0),
        "price": data["price"].reindex(idx).ffill(),
        "fx": data["fx"].reindex(idx).ffill(),
    }
    sim = _simulate(data, settings)
    ideal_eq = data["ideal_equity"] / data["ideal_equity"].iloc[0]
    ideal_account = account_value_series(
        ideal_eq,
        settings.initial_capital,
        monthly_contribution=settings.monthly_contribution,
    )
    invested = total_invested(
        ideal_eq,
        settings.initial_capital,
        settings.monthly_contribution,
    )
    ideal_metrics = _metric_nav(ideal_eq)
    account_diff = sim["account"] / ideal_account - 1.0
    corr = float(
        sim["integer_nav"].pct_change().fillna(0.0).corr(
            ideal_eq.pct_change().fillna(0.0)
        )
    )
    frame = pd.DataFrame({
        "ideal_account": ideal_account,
        "integer_account": sim["account"],
        "invested_principal": sim["invested"],
        "ideal_nav": ideal_eq,
        "integer_twr_nav": sim["integer_nav"],
        "account_diff_integer_vs_ideal": account_diff,
    })
    summary = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "invested_principal": float(invested),
        "ideal": {
            **ideal_metrics,
            "account_mdd": _account_mdd(ideal_account),
            "final_value": float(ideal_account.iloc[-1]),
            "profit": float(ideal_account.iloc[-1] - invested),
            "return_on_principal": float(ideal_account.iloc[-1] / invested - 1.0),
        },
        "integer": {
            **sim["metrics"],
            "account_mdd": sim["account_mdd"],
            "final_value": sim["final_value"],
            "profit": sim["profit"],
            "return_on_principal": sim["return_on_principal"],
            "rebalance_count": sim["rebalance_count"],
            "order_count": sim["order_count"],
            "turnover_per_year": sim["turnover_per_year"],
            "cost_paid": sim["cost_paid"],
            "cost_on_principal": sim["cost_paid"] / float(invested),
            "avg_target_fit": sim["avg_target_fit"],
            "min_target_fit": sim["min_target_fit"],
            "avg_tracking_error": sim["avg_tracking_error"],
            "avg_cash_weight": sim["avg_cash_weight"],
            "max_cash_weight": sim["max_cash_weight"],
        },
        "tracking": {
            "daily_return_corr": corr,
            "final_integer_vs_ideal": sim["final_value"] / float(ideal_account.iloc[-1]) - 1.0,
            "avg_account_diff": float(account_diff.mean()),
            "min_account_diff": float(account_diff.min()),
            "max_account_diff": float(account_diff.max()),
        },
        "skipped_high_price": sim["skipped_high_price"],
    }
    return {"summary": summary, "frame": frame}


def _metric_nav(nav: pd.Series) -> dict[str, float]:
    nav = pd.Series(nav).dropna().astype(float)
    eq = nav / nav.iloc[0]
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {
        "cagr": float(p["cagr"]),
        "mdd": float(p["max_drawdown"]),
        "sharpe": float(p["sharpe"]),
        "total_return": float(p["total_return"]),
        "volatility": float(p["volatility"]),
    }


def _account_mdd(account: pd.Series) -> float:
    account = pd.Series(account).dropna().astype(float)
    return float((account / account.cummax() - 1.0).min())


def _real_symbol(column: str) -> str:
    return column[:-5] if column.endswith("_SAFE") else column


def _load_sub_wave(handler: Any, raw: pd.DataFrame, sub: dict[str, Any]) -> pd.DataFrame:
    sub_id = str(sub["config"])
    cfg = handler._sub_config(sub_id, sub.get("overrides"))
    art = ProfilePortfolioHandler().build_artifacts(
        raw[sub_id], cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252
    )
    wave = art.extras["portfolio_wave"]
    if "time" in wave.columns:
        wave = wave.set_index("time")
    wave.index = pd.to_datetime(wave.index)
    return wave.astype(float)


def _build_leaf_targets(strategy_id: str, config: dict[str, Any], symbol: str) -> dict[str, Any]:
    handler = get_handler(strategy_id)
    raw = handler.load_data(symbol, config, synthetic=False)
    ideal_art = handler.build_artifacts(raw, config, symbol=symbol, phase="all", bars_per_year=252)
    idx = pd.DatetimeIndex(ideal_art.equity.index)
    top_weights = handler._target_weight_frame(
        config,
        raw,
        idx,
        [float(s.get("weight", 1.0 / len(config["sub_strategies"]))) for s in config["sub_strategies"]],
    ).reindex(idx).ffill().bfill()

    leaf = pd.DataFrame(index=idx)
    cash = pd.Series(0.0, index=idx)
    asset_currency: dict[str, str] = {}
    for sub in config["sub_strategies"]:
        sid = str(sub["config"])
        cur = str(sub.get("currency", config.get("base_currency", "KRW")))
        wave = _load_sub_wave(handler, raw, sub).reindex(idx).ffill().fillna(0.0)
        sleeve_w = top_weights[sid].reindex(idx).ffill().fillna(0.0)
        for col in wave.columns:
            weighted = sleeve_w * wave[col]
            if col == "cash":
                cash = cash.add(weighted, fill_value=0.0)
                continue
            sym = _real_symbol(str(col))
            leaf[sym] = leaf.get(sym, pd.Series(0.0, index=idx)).add(weighted, fill_value=0.0)
            asset_currency.setdefault(sym, cur)

    leaf = leaf.fillna(0.0)
    cash = cash.add((1.0 - leaf.sum(axis=1) - cash).clip(lower=0.0), fill_value=0.0)
    fx = raw["__FX__"][str(config["fx_symbol"])]["close"].reindex(idx).ffill()
    prices = {}
    for sym, cur in asset_currency.items():
        try:
            px = load_yfinance(sym, "1d", "max")["close"].reindex(idx).ffill()
        except Exception:  # noqa: BLE001
            px = pd.Series(np.nan, index=idx)
        prices[sym] = px * fx if cur == "USD" else px
    price = pd.DataFrame(prices, index=idx)
    valid = price.notna() & (price > 0.0)
    leaf = leaf.where(valid, 0.0)
    leaf["__CASH__"] = (1.0 - leaf.sum(axis=1)).clip(lower=0.0)
    return {
        "idx": idx,
        "ideal_equity": ideal_art.equity.reindex(idx).ffill(),
        "target": leaf,
        "price": price.reindex(idx).ffill(),
        "fx": fx,
        "asset_currency": asset_currency,
    }


def _current_account(shares: dict[str, float], cash: float, prices: pd.Series, symbols: list[str]) -> float:
    total = float(cash)
    for sym in symbols:
        px = prices.get(sym, np.nan)
        price = 0.0 if pd.isna(px) else float(px)
        total += shares[sym] * price
    return float(total)


def _current_weights(
    shares: dict[str, float],
    cash: float,
    prices: pd.Series,
    symbols: list[str],
    account: float,
) -> pd.Series:
    if account <= 0.0:
        return pd.Series(0.0, index=[*symbols, "__CASH__"])
    vals = {}
    for sym in symbols:
        px = prices.get(sym, np.nan)
        price = 0.0 if pd.isna(px) else float(px)
        vals[sym] = shares[sym] * price / account
    vals["__CASH__"] = cash / account
    return pd.Series(vals)


def _min_order_value(sym: str, currency: str, fx_value: float, settings: IntegerDcaSettings) -> float:
    return settings.min_us_order_usd * fx_value if currency == "USD" else settings.min_kr_order


def _cost_rate(sym: str, asset_currency: dict[str, str], settings: IntegerDcaSettings) -> float:
    return settings.cost_us if asset_currency[sym] == "USD" else settings.cost_kr


def _rebalance_integer(
    shares: dict[str, float],
    cash: float,
    account: float,
    target: pd.Series,
    prices: pd.Series,
    fx_value: float,
    asset_currency: dict[str, str],
    settings: IntegerDcaSettings,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    symbols = sorted(asset_currency)
    avail = {
        s: float(prices[s])
        for s in symbols
        if pd.notna(prices.get(s)) and float(prices[s]) > 0.0
    }
    target = target.reindex([*symbols, "__CASH__"]).fillna(0.0)
    desired = {
        s: math.floor(max(account * float(target[s]), 0.0) / avail[s])
        for s in avail
    }

    def order_value_ok(s: str, from_qty: float, to_qty: float) -> bool:
        value = abs(to_qty - from_qty) * avail[s]
        if value <= 0.0:
            return False
        if to_qty == 0.0 and from_qty > 0.0:
            return True
        return value >= _min_order_value(s, asset_currency[s], fx_value, settings)

    for s in symbols:
        if s not in avail:
            desired[s] = shares[s]
            continue
        if not order_value_ok(s, shares[s], desired.get(s, 0.0)):
            desired[s] = shares[s]

    def cash_delta(next_shares: dict[str, float]) -> float:
        buy = sum(
            max(next_shares.get(s, 0.0) - shares[s], 0.0)
            * avail[s]
            * (1.0 + _cost_rate(s, asset_currency, settings))
            for s in avail
        )
        sell = sum(
            max(shares[s] - next_shares.get(s, 0.0), 0.0)
            * avail[s]
            * (1.0 - _cost_rate(s, asset_currency, settings))
            for s in avail
        )
        return buy - sell

    guard = 0
    while cash_delta(desired) > cash + 1e-6 and guard < 50_000:
        over = max(
            avail,
            key=lambda s: (
                desired.get(s, 0.0) * avail[s] / max(account, 1.0) - float(target[s])
                if desired.get(s, 0.0) > 0.0 else -9.0
            ),
        )
        if desired.get(over, 0.0) <= 0.0:
            break
        desired[over] -= 1.0
        guard += 1

    guard = 0
    while guard < 50_000:
        candidates = []
        for s in avail:
            nxt = desired.get(s, 0.0) + 1.0
            if not order_value_ok(s, desired.get(s, 0.0), nxt):
                continue
            if cash_delta({**desired, s: nxt}) <= cash + 1e-6:
                under = float(target[s]) - desired.get(s, 0.0) * avail[s] / max(account, 1.0)
                if under > 0.0:
                    candidates.append((under, s))
        if not candidates:
            break
        _, pick = max(candidates)
        desired[pick] = desired.get(pick, 0.0) + 1.0
        guard += 1

    traded = 0.0
    cost_paid = 0.0
    orders = 0
    skipped = []
    for s in avail:
        delta_qty = desired.get(s, 0.0) - shares[s]
        if abs(delta_qty) <= 0.0:
            if float(target[s]) > 0.0 and shares[s] * avail[s] < target[s] * account:
                one_share = avail[s] * (1.0 + _cost_rate(s, asset_currency, settings))
                if one_share > cash or avail[s] < _min_order_value(s, asset_currency[s], fx_value, settings):
                    skipped.append(s)
            continue
        value = abs(delta_qty) * avail[s]
        traded += value
        cost_paid += value * _cost_rate(s, asset_currency, settings)
        orders += 1

    new_cash = cash - cash_delta(desired)
    next_shares = {s: float(desired.get(s, 0.0)) for s in symbols}
    return next_shares, float(new_cash), {
        "traded": traded,
        "cost_paid": cost_paid,
        "orders": orders,
        "skipped": skipped,
    }


def _simulate(data: dict[str, Any], settings: IntegerDcaSettings) -> dict[str, Any]:
    idx: pd.DatetimeIndex = data["idx"]
    target: pd.DataFrame = data["target"]
    price: pd.DataFrame = data["price"]
    fx: pd.Series = data["fx"]
    asset_currency: dict[str, str] = data["asset_currency"]
    symbols = sorted(asset_currency)
    month_first = ~pd.Series(idx, index=idx).index.to_period("M").duplicated()
    rebal = rebalance_dates(idx, "monthly")
    shares = {s: 0.0 for s in symbols}
    cash = 0.0
    prev_account: float | None = None
    account_values = []
    invested_values = []
    twr_returns = []
    target_fits = []
    tracking_errors = []
    cash_weights = []
    skipped: dict[str, int] = {}
    turnover = 0.0
    cost_paid = 0.0
    order_count = 0
    rebalance_count = 0
    invested = 0.0

    for i, t in enumerate(idx):
        contribution = 0.0
        if i == 0:
            contribution += settings.initial_capital
        if bool(month_first[i]):
            contribution += settings.monthly_contribution
        cash += contribution
        invested += contribution
        px = price.loc[t]
        account_before = _current_account(shares, cash, px, symbols)
        if i == 0 or t in rebal:
            shares, cash, st = _rebalance_integer(
                shares, cash, account_before, target.loc[t], px, float(fx.loc[t]),
                asset_currency, settings,
            )
            turnover += st["traded"] / max(account_before, 1.0)
            cost_paid += st["cost_paid"]
            order_count += st["orders"]
            for sym in st["skipped"]:
                skipped[sym] = skipped.get(sym, 0) + 1
            rebalance_count += 1

        account = _current_account(shares, cash, px, symbols)
        cw = _current_weights(shares, cash, px, symbols, account)
        tw = target.loc[t].reindex([*symbols, "__CASH__"]).fillna(0.0)
        tracking_error = float((cw - tw).abs().sum())
        tracking_errors.append(tracking_error)
        target_fits.append(max(0.0, 1.0 - tracking_error / 2.0))
        cash_weights.append(float(cw["__CASH__"]))
        twr_returns.append(0.0 if prev_account is None or prev_account <= 0.0 else (account - contribution) / prev_account - 1.0)
        prev_account = account
        account_values.append(account)
        invested_values.append(invested)

    account_series = pd.Series(account_values, index=idx, name="integer_account")
    invested_series = pd.Series(invested_values, index=idx, name="invested_principal")
    integer_nav = (1.0 + pd.Series(twr_returns, index=idx)).cumprod().rename("integer_twr_nav")
    return {
        "account": account_series,
        "invested": invested_series,
        "integer_nav": integer_nav,
        "metrics": _metric_nav(integer_nav),
        "account_mdd": _account_mdd(account_series),
        "final_value": float(account_series.iloc[-1]),
        "invested_principal": float(invested_series.iloc[-1]),
        "profit": float(account_series.iloc[-1] - invested_series.iloc[-1]),
        "return_on_principal": float(account_series.iloc[-1] / invested_series.iloc[-1] - 1.0),
        "rebalance_count": rebalance_count,
        "order_count": order_count,
        "turnover_per_year": turnover / (len(idx) / 252.0),
        "cost_paid": cost_paid,
        "avg_target_fit": float(np.mean(target_fits)),
        "min_target_fit": float(np.min(target_fits)),
        "avg_tracking_error": float(np.mean(tracking_errors)),
        "avg_cash_weight": float(np.mean(cash_weights)),
        "max_cash_weight": float(np.max(cash_weights)),
        "skipped_high_price": sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
    }
