"""yoon2m 유동 리밸런싱 후보 비교.

하위 yoon1j_lp/yoon1j_kr 신호는 매일 계산해 두고, 후보별 조건이 켜지는 날에만
목표비중으로 리밸런싱한다. 이렇게 하면 "월간 고정"과 이벤트 기반 리밸런싱을 같은
목표 신호/가격/비용 가정에서 나란히 비교할 수 있다.

사용:
    PYTHONPATH=src python scripts/profile_sizing/yoon2m_dynamic_rebalance_compare.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from profile_sizing.yoon2a_bootstrap import build_series  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

TOPK = 3
INITIAL_KRW = 2_000_000.0
COST_KR = 0.0015
COST_US = 0.0060


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    description: str


CANDIDATES = [
    Candidate("monthly", "월간 고정", "현행 기준선"),
    Candidate("weekly", "주간 고정", "거래 빈도 증가 비교군"),
    Candidate("quarterly", "분기 고정", "거래 절감 비교군"),
    Candidate("monthly_skip_5", "월말 gap 5%", "월말에 gross gap >= 5%일 때만"),
    Candidate("monthly_skip_10", "월말 gap 10%", "월말에 gross gap >= 10%일 때만"),
    Candidate("monthly_skip_mix", "월말 혼합스킵", "월말에 gross 5%/종목 3%p/통화 4%p 중 하나 충족"),
    Candidate("target_gap_10", "목표변화 10%", "현재 보유비중과 일별 목표의 gross gap >= 10%시에만"),
    Candidate("leaf_drift_5", "종목 드리프트 5%p", "단일 종목 목표-현재 차이 >= 5%p"),
    Candidate("currency_drift_5", "통화 드리프트 5%p", "USD/KRW 슬리브 목표-현재 차이 >= 5%p"),
    Candidate("regime_event", "레짐 전환", "월말 + SPY/KODEX200 100MA 레짐 전환"),
    Candidate("drawdown_event", "드로다운 트리거", "월말 + NAV 고점 대비 -5% 하향돌파"),
    Candidate("vol_event", "변동성 트리거", "월말 + 벤치 20일 변동성의 1년 80% 분위 상향돌파"),
    Candidate("hybrid_final", "하이브리드 최종후보", "월말 정기 점검 + gross gap 10% 이상일 때만 실행"),
]


def sleeve_wave(cfg_id: str) -> pd.DataFrame:
    cfg = json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8"))
    cfg["top_k"] = TOPK
    cfg["rebalance_freq"] = "daily"
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    w = art.extras["portfolio_wave"]
    if "time" in w.columns:
        w = w.set_index("time")
    w.index = pd.to_datetime(w.index)
    return w.drop(columns=[c for c in w.columns if c == "cash"], errors="ignore")


def real_symbol(col: str) -> str:
    return col[:-5] if col.endswith("_SAFE") else col


def metric_row(eq: pd.Series) -> dict[str, float]:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {
        "cagr": float(p["cagr"]),
        "mdd": float(p["max_drawdown"]),
        "sharpe": float(p["sharpe"]),
        "total_return": float(eq.iloc[-1] - 1.0),
    }


def market_on(symbol: str, idx: pd.DatetimeIndex, ma_len: int = 100) -> pd.Series:
    close = load_yfinance(symbol, "1d", "max")["close"].reindex(idx).ffill()
    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    below = (close <= ma).shift(1)
    warm = ma.shift(1).isna()
    return pd.Series(np.where(warm, True, below != True), index=idx)  # noqa: E712


def build_inputs() -> dict[str, Any]:
    lp = sleeve_wave("yoon1j_lp")
    kr = sleeve_wave("yoon1j_kr")
    idx = lp.index.intersection(kr.index).sort_values()

    raw_cols = list(lp.columns) + list(kr.columns)
    currency_by_col = {c: "USD" for c in lp.columns}
    currency_by_col.update({c: "KRW" for c in kr.columns})
    asset_currency: dict[str, str] = {}
    for col in raw_cols:
        asset_currency.setdefault(real_symbol(col), currency_by_col[col])

    w_raw = pd.DataFrame(0.0, index=idx, columns=raw_cols)
    w_raw.loc[:, lp.columns] = 0.5 * lp.reindex(idx).fillna(0.0).to_numpy()
    w_raw.loc[:, kr.columns] = 0.5 * kr.reindex(idx).fillna(0.0).to_numpy()

    cols = sorted({real_symbol(c) for c in raw_cols})
    target = pd.DataFrame(0.0, index=idx, columns=cols)
    for c in raw_cols:
        target[real_symbol(c)] += w_raw[c]

    fx = load_yfinance("KRW=X", "1d", "max")["close"].reindex(idx).ffill()
    prices = {}
    for sym in cols:
        try:
            px = load_yfinance(sym, "1d", "max")["close"].reindex(idx).ffill()
        except Exception:  # noqa: BLE001
            px = pd.Series(np.nan, index=idx)
        prices[sym] = px * fx if asset_currency[sym] == "USD" else px
    price = pd.DataFrame(prices).reindex(idx).ffill()

    _, bench, _ = build_series()
    bench = bench.reindex(idx).ffill()
    us_assets = [s for s, cur in asset_currency.items() if cur == "USD"]
    kr_assets = [s for s, cur in asset_currency.items() if cur == "KRW"]

    regime = pd.concat({
        "us": market_on("SPY", idx),
        "kr": market_on("069500.KS", idx),
    }, axis=1)
    regime_flip = regime.ne(regime.shift(1)).any(axis=1).fillna(False)

    bret = bench.pct_change()
    vol20 = bret.rolling(20, min_periods=20).std()
    vol_thr = vol20.rolling(252, min_periods=126).quantile(0.8).shift(1)
    high_vol = (vol20 > vol_thr).fillna(False)
    vol_cross = (high_vol & ~high_vol.shift(1, fill_value=False)).reindex(idx).fillna(False)

    return {
        "idx": idx,
        "target": target,
        "price": price,
        "bench": bench,
        "asset_currency": asset_currency,
        "cost": {s: (COST_US if asset_currency[s] == "USD" else COST_KR) for s in cols},
        "us_assets": us_assets,
        "kr_assets": kr_assets,
        "month_end": rebalance_dates(idx, "monthly"),
        "week_end": rebalance_dates(idx, "weekly"),
        "quarter_end": rebalance_dates(idx, "quarterly"),
        "regime_flip": regime_flip,
        "vol_cross": vol_cross,
    }


def current_weights(shares: dict[str, float], price: pd.Series, account: float) -> pd.Series:
    if account <= 0.0:
        return pd.Series(0.0, index=price.index)
    vals = {
        s: shares.get(s, 0.0) * (float(price[s]) if pd.notna(price[s]) else 0.0)
        for s in price.index
    }
    return pd.Series(vals, index=price.index) / account


def should_rebalance(
    key: str,
    *,
    i: int,
    t: pd.Timestamp,
    first: bool,
    days_since: int,
    month_end: set,
    week_end: set,
    quarter_end: set,
    gross_gap: float,
    max_leaf_gap: float,
    currency_gap: float,
    regime_flip: bool,
    drawdown_cross: bool,
    vol_cross: bool,
) -> bool:
    if first:
        return True
    if key == "monthly":
        return t in month_end
    if key == "weekly":
        return t in week_end
    if key == "quarterly":
        return t in quarter_end
    if key == "monthly_skip_5":
        return (t in month_end) and gross_gap >= 0.05
    if key == "monthly_skip_10":
        return (t in month_end) and gross_gap >= 0.10
    if key == "monthly_skip_mix":
        return (t in month_end) and (
            gross_gap >= 0.05 or max_leaf_gap >= 0.03 or currency_gap >= 0.04
        )

    gap_ok = days_since >= 5
    if key == "target_gap_10":
        return gap_ok and gross_gap >= 0.10
    if key == "leaf_drift_5":
        return gap_ok and max_leaf_gap >= 0.05
    if key == "currency_drift_5":
        return gap_ok and currency_gap >= 0.05
    if key == "regime_event":
        return (t in month_end) or (gap_ok and regime_flip)
    if key == "drawdown_event":
        return (t in month_end) or (gap_ok and drawdown_cross)
    if key == "vol_event":
        return (t in month_end) or (gap_ok and vol_cross)
    if key == "hybrid_final":
        return (t in month_end) and gross_gap >= 0.10
    raise ValueError(key)


def simulate_fractional(candidate: Candidate, data: dict[str, Any]) -> dict[str, Any]:
    idx: pd.DatetimeIndex = data["idx"]
    target: pd.DataFrame = data["target"]
    price: pd.DataFrame = data["price"]
    cost: dict[str, float] = data["cost"]
    us_assets: list[str] = data["us_assets"]
    kr_assets: list[str] = data["kr_assets"]

    shares = {s: 0.0 for s in target.columns}
    cash = 1.0
    nav = np.empty(len(idx))
    rebalance_count = 0
    gross_turnover = 0.0
    cost_paid = 0.0
    last_rebal_i = -10_000
    peak = 1.0
    prev_dd = 0.0

    for i, t in enumerate(idx):
        px = price.loc[t]
        account = cash + sum(
            shares[s] * (float(px[s]) if pd.notna(px[s]) else 0.0)
            for s in target.columns
        )
        peak = max(peak, account)
        dd = account / peak - 1.0 if peak > 0.0 else 0.0
        cw = current_weights(shares, px, account)
        tw = target.loc[t].reindex(target.columns).fillna(0.0)
        gap = (tw - cw).abs()
        gross_gap = float(gap.sum())
        max_leaf_gap = float(gap.max())
        cur_us = float(cw.reindex(us_assets).sum())
        tgt_us = float(tw.reindex(us_assets).sum())
        cur_kr = float(cw.reindex(kr_assets).sum())
        tgt_kr = float(tw.reindex(kr_assets).sum())
        currency_gap = max(abs(cur_us - tgt_us), abs(cur_kr - tgt_kr))
        drawdown_cross = dd <= -0.05 and prev_dd > -0.05

        do_rebal = should_rebalance(
            candidate.key,
            i=i,
            t=t,
            first=rebalance_count == 0,
            days_since=i - last_rebal_i,
            month_end=data["month_end"],
            week_end=data["week_end"],
            quarter_end=data["quarter_end"],
            gross_gap=gross_gap,
            max_leaf_gap=max_leaf_gap,
            currency_gap=currency_gap,
            regime_flip=bool(data["regime_flip"].iloc[i]),
            drawdown_cross=drawdown_cross,
            vol_cross=bool(data["vol_cross"].iloc[i]),
        )
        if do_rebal and account > 0.0:
            traded = 0.0
            paid = 0.0
            for s in target.columns:
                if pd.isna(px[s]) or px[s] <= 0:
                    continue
                target_val = float(tw[s]) * account
                cur_val = shares[s] * float(px[s])
                trade_val = target_val - cur_val
                if abs(trade_val) > 1e-12:
                    fee = cost[s] * abs(trade_val)
                    cash -= trade_val + fee
                    shares[s] = target_val / float(px[s])
                    traded += abs(trade_val)
                    paid += fee
            account = cash + sum(
                shares[s] * (float(px[s]) if pd.notna(px[s]) else 0.0)
                for s in target.columns
            )
            rebalance_count += 1
            gross_turnover += traded
            cost_paid += paid
            last_rebal_i = i

        nav[i] = account
        prev_dd = dd

    eq = pd.Series(nav, index=idx)
    m = metric_row(eq)
    return {
        **m,
        "rebalance_count": rebalance_count,
        "rebalance_per_year": rebalance_count / (len(idx) / 252.0),
        "gross_turnover": gross_turnover,
        "turnover_per_year": gross_turnover / (len(idx) / 252.0),
        "cost_paid": cost_paid,
        "cost_per_year": cost_paid / (len(idx) / 252.0),
        "equity": eq,
    }


def simulate_integer(candidate: Candidate, data: dict[str, Any], initial: float) -> dict[str, Any]:
    idx: pd.DatetimeIndex = data["idx"]
    target: pd.DataFrame = data["target"]
    price: pd.DataFrame = data["price"]
    cost: dict[str, float] = data["cost"]
    us_assets: list[str] = data["us_assets"]
    kr_assets: list[str] = data["kr_assets"]

    shares = {s: 0.0 for s in target.columns}
    cash = float(initial)
    nav = np.empty(len(idx))
    rebalance_count = 0
    gross_turnover = 0.0
    cost_paid = 0.0
    last_rebal_i = -10_000
    peak = initial
    prev_dd = 0.0

    for i, t in enumerate(idx):
        px = price.loc[t]
        account = cash + sum(
            shares[s] * (float(px[s]) if pd.notna(px[s]) else 0.0)
            for s in target.columns
        )
        peak = max(peak, account)
        dd = account / peak - 1.0 if peak > 0.0 else 0.0
        cw = current_weights(shares, px, account)
        tw = target.loc[t].reindex(target.columns).fillna(0.0)
        gap = (tw - cw).abs()
        gross_gap = float(gap.sum())
        max_leaf_gap = float(gap.max())
        currency_gap = max(
            abs(float(cw.reindex(us_assets).sum()) - float(tw.reindex(us_assets).sum())),
            abs(float(cw.reindex(kr_assets).sum()) - float(tw.reindex(kr_assets).sum())),
        )
        drawdown_cross = dd <= -0.05 and prev_dd > -0.05

        do_rebal = should_rebalance(
            candidate.key,
            i=i,
            t=t,
            first=rebalance_count == 0,
            days_since=i - last_rebal_i,
            month_end=data["month_end"],
            week_end=data["week_end"],
            quarter_end=data["quarter_end"],
            gross_gap=gross_gap,
            max_leaf_gap=max_leaf_gap,
            currency_gap=currency_gap,
            regime_flip=bool(data["regime_flip"].iloc[i]),
            drawdown_cross=drawdown_cross,
            vol_cross=bool(data["vol_cross"].iloc[i]),
        )
        if do_rebal and account > 0.0:
            avail = {
                s: float(px[s])
                for s in target.columns
                if pd.notna(px[s]) and float(px[s]) > 0.0
            }
            new_sh = {
                s: math.floor(account * float(tw[s]) / avail[s])
                for s in avail
            }

            def cash_delta(sh: dict[str, float]) -> float:
                buy = sum(
                    max(sh.get(s, 0.0) - shares[s], 0.0) * avail[s] * (1.0 + cost[s])
                    for s in avail
                )
                sell = sum(
                    max(shares[s] - sh.get(s, 0.0), 0.0) * avail[s] * (1.0 - cost[s])
                    for s in avail
                )
                return buy - sell

            guard = 0
            while cash_delta(new_sh) > cash + 1e-6 and guard < 10_000:
                over = max(
                    avail,
                    key=lambda s: (
                        new_sh.get(s, 0.0) * avail[s] / account - float(tw[s])
                        if new_sh.get(s, 0.0) > 0.0 else -9.0
                    ),
                )
                if new_sh.get(over, 0.0) <= 0.0:
                    break
                new_sh[over] -= 1.0
                guard += 1

            guard = 0
            while guard < 10_000:
                cand = [
                    s for s in avail
                    if cash_delta({**new_sh, s: new_sh.get(s, 0.0) + 1.0}) <= cash + 1e-6
                ]
                if not cand:
                    break
                pick = max(cand, key=lambda s: float(tw[s]) - new_sh.get(s, 0.0) * avail[s] / account)
                if float(tw[pick]) - new_sh.get(pick, 0.0) * avail[pick] / account <= 0.0:
                    break
                new_sh[pick] = new_sh.get(pick, 0.0) + 1.0
                guard += 1

            traded = sum(abs(new_sh.get(s, 0.0) - shares[s]) * avail[s] for s in avail)
            paid = sum(abs(new_sh.get(s, 0.0) - shares[s]) * avail[s] * cost[s] for s in avail)
            cash -= cash_delta(new_sh)
            shares = {s: float(new_sh.get(s, 0.0)) for s in target.columns}
            account = cash + sum(
                shares[s] * (float(px[s]) if pd.notna(px[s]) else 0.0)
                for s in target.columns
            )
            rebalance_count += 1
            gross_turnover += traded / max(account, 1.0)
            cost_paid += paid / initial
            last_rebal_i = i

        nav[i] = account
        prev_dd = dd

    eq = pd.Series(nav, index=idx)
    m = metric_row(eq)
    return {
        **m,
        "rebalance_count": rebalance_count,
        "rebalance_per_year": rebalance_count / (len(idx) / 252.0),
        "gross_turnover": gross_turnover,
        "turnover_per_year": gross_turnover / (len(idx) / 252.0),
        "cost_paid": cost_paid,
        "cost_per_year": cost_paid / (len(idx) / 252.0),
        "equity": eq,
    }


def write_report(rows: list[dict[str, Any]], bench_metrics: dict[str, float], out_json: Path, out_md: Path) -> None:
    payload = {
        "top_k": TOPK,
        "initial_krw": INITIAL_KRW,
        "cost_kr": COST_KR,
        "cost_us": COST_US,
        "bench": bench_metrics,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    lines = [
        "# yoon2m 유동 리밸런싱 후보 비교",
        "",
        f"- top_k={TOPK}, 비용: KR {COST_KR * 100:.2f}%/편도, US {COST_US * 100:.2f}%/편도",
        f"- 벤치: CAGR {pct(bench_metrics['cagr'])}, MDD {pct(bench_metrics['mdd'])}, Sharpe {bench_metrics['sharpe']:.3f}",
        "",
        "## 연속비중 비용차감",
        "",
        "| 후보 | CAGR | MDD | Sharpe | 리밸/년 | 회전율/년 | 비용/년 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        f = row["fractional"]
        lines.append(
            f"| {row['label']} | {pct(f['cagr'])} | {pct(f['mdd'])} | {f['sharpe']:.3f} | "
            f"{f['rebalance_per_year']:.1f} | {f['turnover_per_year']:.2f}x | {pct(f['cost_per_year'])} |"
        )
    lines += [
        "",
        f"## 정수주 {INITIAL_KRW / 10000:.0f}만원 실집행",
        "",
        "| 후보 | CAGR | MDD | Sharpe | 리밸/년 | 회전율/년 | 비용/년 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        r = row["integer"]
        lines.append(
            f"| {row['label']} | {pct(r['cagr'])} | {pct(r['mdd'])} | {r['sharpe']:.3f} | "
            f"{r['rebalance_per_year']:.1f} | {r['turnover_per_year']:.2f}x | {pct(r['cost_per_year'])} |"
        )
    lines += [
        "",
        "## 후보 설명",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['label']}: {row['description']}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_equity(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "equity"}


def main() -> None:
    data = build_inputs()
    bench = data["bench"].dropna()
    bench_metrics = metric_row(bench)
    rows = []
    print(
        f"기간 {data['idx'][0].date()}~{data['idx'][-1].date()} · "
        f"{len(data['idx'])}봉 · 후보 {len(CANDIDATES)}개"
    )
    print(f"벤치 CAGR {bench_metrics['cagr']*100:.2f}% MDD {bench_metrics['mdd']*100:.2f}% Sharpe {bench_metrics['sharpe']:.3f}\n")
    print(f"{'후보':18s}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'리밸/년':>8}{'정수CAGR':>10}{'정수MDD':>9}")
    for cand in CANDIDATES:
        frac = simulate_fractional(cand, data)
        integer = simulate_integer(cand, data, INITIAL_KRW)
        row = {
            "key": cand.key,
            "label": cand.label,
            "description": cand.description,
            "fractional": strip_equity(frac),
            "integer": strip_equity(integer),
        }
        rows.append(row)
        print(
            f"{cand.label:18s}{frac['cagr']*100:7.2f}%{frac['mdd']*100:8.2f}%"
            f"{frac['sharpe']:8.3f}{frac['rebalance_per_year']:8.1f}"
            f"{integer['cagr']*100:9.2f}%{integer['mdd']*100:8.2f}%"
        )

    out_json = ROOT / "reports" / "profile_sizing" / "yoon2m_dynamic_rebalance_compare.json"
    out_md = ROOT / "reports" / "profile_sizing" / "yoon2m_dynamic_rebalance_compare.md"
    write_report(rows, bench_metrics, out_json, out_md)
    print(f"\nJSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
