"""Probe covered-call and high-dividend universes for yoon2_final_fx.

The yoon2_final_fx top-level USD/KRW relative-momentum layer is kept intact.
Only lower sleeve universes are replaced through temporary config overrides.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_income_universe.py
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.engine import performance  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.ui.presentation import account_value_series, total_invested  # noqa: E402


INITIAL_KRW = 2_000_000.0
MONTHLY_KRW = 200_000.0


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    us_universe: list[str] | None = None
    kr_universe: list[str] | None = None
    note: str = ""


US_COVERED_CALL_CORE = ["QYLD", "XYLD", "DIVO", "NUSI"]
US_COVERED_CALL_INCOME = ["QYLD", "XYLD", "RYLD", "JEPI", "JEPQ", "DIVO", "NUSI", "SPYI"]
US_HIGH_DIV_ETF_LONG = ["VYM", "DVY", "SDY", "DHS", "DON", "FDL", "PEY", "VIG"]
US_HIGH_DIV_ETF_PLUS = [
    "VYM", "DVY", "SDY", "HDV", "SCHD", "SPYD", "DHS", "DGRW",
    "DON", "FDL", "PEY", "NOBL", "VIG", "DGRO",
]
US_HIGH_DIV_STOCKS = [
    "MO", "T", "VZ", "IBM", "XOM", "CVX", "KO", "PEP",
    "PG", "JNJ", "ABBV", "PM", "O", "KMI", "WMB",
]
KR_HIGH_DIV_ETF = ["161510.KS", "279530.KS", "211900.KS"]
KR_COVERED_CALL_RECENT = ["441680.KS", "458730.KS", "472150.KS", "453850.KS"]


VARIANTS = [
    Variant("baseline", "baseline"),
    Variant("us_covered_call_core", "US covered-call core", us_universe=US_COVERED_CALL_CORE),
    Variant("us_covered_call_income", "US covered-call income", us_universe=US_COVERED_CALL_INCOME),
    Variant("us_high_div_etf_long", "US high-div ETF long", us_universe=US_HIGH_DIV_ETF_LONG),
    Variant("us_high_div_etf_plus", "US high-div ETF plus", us_universe=US_HIGH_DIV_ETF_PLUS),
    Variant(
        "us_high_div_stocks",
        "US high-div stocks",
        us_universe=US_HIGH_DIV_STOCKS,
        note="Current high-dividend stock basket; survivorship/selection bias risk.",
    ),
    Variant("kr_high_div_etf", "KR high-div ETF", kr_universe=KR_HIGH_DIV_ETF),
    Variant("both_high_div_etf", "US+KR high-div ETF", us_universe=US_HIGH_DIV_ETF_PLUS, kr_universe=KR_HIGH_DIV_ETF),
    Variant("kr_covered_call_recent", "KR covered-call recent", kr_universe=KR_COVERED_CALL_RECENT),
    Variant(
        "both_covered_call_recent",
        "US+KR covered-call recent",
        us_universe=US_COVERED_CALL_INCOME,
        kr_universe=KR_COVERED_CALL_RECENT,
    ),
]


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def money(v: float | None) -> str:
    return "-" if v is None else f"{v:,.0f}원"


def metric_nav(nav: pd.Series) -> dict[str, float]:
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


def account_mdd(account: pd.Series) -> float:
    account = pd.Series(account).dropna().astype(float)
    return float((account / account.cummax() - 1.0).min())


def symbol_start(symbol: str) -> pd.Timestamp | None:
    try:
        close = load_yfinance(symbol, "1d", "max")["close"].dropna()
    except Exception:
        return None
    return pd.Timestamp(close.index[0]) if len(close) else None


def active_start_for(variant: Variant, base_start: pd.Timestamp) -> pd.Timestamp:
    starts = [base_start]
    for sym in (variant.us_universe or []) + (variant.kr_universe or []):
        st = symbol_start(sym)
        if st is not None:
            starts.append(st)
    return max(starts)


def variant_config(base: dict[str, Any], variant: Variant) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    for sub in cfg["sub_strategies"]:
        cur = str(sub.get("currency", cfg.get("base_currency", "KRW")))
        if cur == "USD" and variant.us_universe is not None:
            over = dict(sub.get("overrides") or {})
            over["universe"] = list(variant.us_universe)
            over["top_k"] = min(3, len(variant.us_universe))
            sub["overrides"] = over
        if cur == "KRW" and variant.kr_universe is not None:
            over = dict(sub.get("overrides") or {})
            over["universe"] = list(variant.kr_universe)
            over["top_k"] = min(3, len(variant.kr_universe))
            sub["overrides"] = over
    cfg["version"] = f"yoon2_final_fx_{variant.key}"
    return cfg


def dca_stats(nav: pd.Series) -> dict[str, float]:
    eq = nav / nav.iloc[0]
    account = account_value_series(eq, INITIAL_KRW, monthly_contribution=MONTHLY_KRW)
    invested = total_invested(eq, INITIAL_KRW, MONTHLY_KRW)
    return {
        "invested": float(invested),
        "final": float(account.iloc[-1]),
        "profit": float(account.iloc[-1] - invested),
        "return_on_principal": float(account.iloc[-1] / invested - 1.0),
        "account_mdd": account_mdd(account),
    }


def run_variant(handler: Yoon1kHandler, base: dict[str, Any], variant: Variant) -> dict[str, Any]:
    cfg = variant_config(base, variant)
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav = (art.equity / art.equity.iloc[0]).dropna()
    active_start = active_start_for(variant, pd.Timestamp(nav.index[0]))
    active_nav = nav.loc[nav.index >= active_start]
    if len(active_nav) < 252:
        active_nav = nav
        active_start = pd.Timestamp(nav.index[0])
    row = {
        "key": variant.key,
        "label": variant.label,
        "period": f"{nav.index[0].date()}~{nav.index[-1].date()}",
        "active_start": str(active_start.date()),
        "bars": int(len(nav)),
        "active_bars": int(len(active_nav)),
        "us_universe": variant.us_universe,
        "kr_universe": variant.kr_universe,
        "note": variant.note,
        **{f"full_{k}": v for k, v in metric_nav(nav).items()},
        **{f"active_{k}": v for k, v in metric_nav(active_nav).items()},
        **{f"full_dca_{k}": v for k, v in dca_stats(nav).items()},
        **{f"active_dca_{k}": v for k, v in dca_stats(active_nav).items()},
        "_nav": nav,
        "_active_nav": active_nav,
    }
    return row


def add_baseline_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(r for r in rows if r["key"] == "baseline")
    base_nav = baseline["_nav"]
    out = []
    for row in rows:
        nav = row["_nav"]
        active_nav = row["_active_nav"]
        base_period = base_nav.reindex(nav.index).ffill().dropna()
        base_active = base_nav.reindex(active_nav.index).ffill().dropna()
        if len(base_period) >= 60:
            p = metric_nav(base_period)
            d = dca_stats(base_period)
            row.update({f"base_same_period_{k}": v for k, v in p.items()})
            row.update({f"base_same_period_dca_{k}": v for k, v in d.items()})
        if len(base_active) >= 60:
            p = metric_nav(base_active)
            d = dca_stats(base_active)
            row.update({f"base_active_{k}": v for k, v in p.items()})
            row.update({f"base_active_dca_{k}": v for k, v in d.items()})
        public = {k: v for k, v in row.items() if not k.startswith("_")}
        out.append(public)
    return out


def write_report(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_income_universe.json"
    out_csv = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_income_universe.csv"
    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_income_universe.md"
    rows = add_baseline_comparisons(rows)
    out_json.write_text(json.dumps({"rows": rows, "candidates": candidates}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    lines = [
        "# yoon2_final_fx 인컴 유니버스 테스트",
        "",
        "## 가정",
        "",
        "- `yoon2_final_fx`의 상단 USD/KRW 상대모멘텀 틸트와 월간 리밸런스는 유지.",
        "- 하위 슬리브 유니버스만 임시 override로 교체. 전략 config/registry는 변경하지 않음.",
        f"- 적립식 비교: 초기 `{money(INITIAL_KRW)}`, 월 `{money(MONTHLY_KRW)}`.",
        "- Yahoo 수정주가 기반이므로 ETF 분배금은 가격에 조정 반영된 것으로 본다.",
        "- 커버드콜/국내 커버드콜 ETF는 상장 이력이 짧아 `활성 구간` 성과를 별도 표기.",
        "",
        "## 가용 기간 성과",
        "",
        "| 변형 | 기간 | 활성 시작 | NAV CAGR | NAV MDD | Sharpe | 적립식 원금 | 적립식 최종 | 원금대비 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x["full_dca_return_on_principal"], reverse=True):
        lines.append(
            f"| {r['label']} | {r['period']} | {r['active_start']} | {pct(r['full_cagr'])} | "
            f"{pct(r['full_mdd'])} | {r['full_sharpe']:.3f} | {money(r['full_dca_invested'])} | "
            f"{money(r['full_dca_final'])} | {pct(r['full_dca_return_on_principal'])} |"
        )
    lines.extend([
        "",
        "## 활성 구간 성과와 같은 기간 기준선",
        "",
        "| 변형 | 활성 시작 | bars | 후보 CAGR | 후보 MDD | 기준선 CAGR | 기준선 MDD | 후보 적립식 | 기준선 적립식 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in sorted(rows, key=lambda x: x["active_dca_return_on_principal"], reverse=True):
        lines.append(
            f"| {r['label']} | {r['active_start']} | {r['active_bars']} | "
            f"{pct(r['active_cagr'])} | {pct(r['active_mdd'])} | "
            f"{pct(r.get('base_active_cagr'))} | {pct(r.get('base_active_mdd'))} | "
            f"{pct(r['active_dca_return_on_principal'])} | "
            f"{pct(r.get('base_active_dca_return_on_principal'))} |"
        )
    lines.extend([
        "",
        "## 같은 기간 기준선 대비",
        "",
        "| 변형 | 가용기간 CAGR 차이 | 가용기간 MDD 차이 | 활성 CAGR 차이 | 활성 MDD 차이 | 활성 적립식 차이 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        if r["key"] == "baseline":
            continue
        lines.append(
            f"| {r['label']} | {pct(r['full_cagr'] - r.get('base_same_period_cagr', 0.0))}p | "
            f"{pct(r['full_mdd'] - r.get('base_same_period_mdd', 0.0))}p | "
            f"{pct(r['active_cagr'] - r.get('base_active_cagr', 0.0))}p | "
            f"{pct(r['active_mdd'] - r.get('base_active_mdd', 0.0))}p | "
            f"{pct(r['active_dca_return_on_principal'] - r.get('base_active_dca_return_on_principal', 0.0))}p |"
        )
    lines.extend([
        "",
        "## 후보 데이터",
        "",
        "| 티커 | 시작일 | 최근일 | 최근가 | 일봉수 |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for c in candidates:
        lines.append(
            f"| {c['symbol']} | {c['start']} | {c['end']} | {c['last_close']:.2f} | {c['bars']} |"
        )
    lines.extend([
        "",
        "## 판정",
        "",
        "- 고배당 ETF 유니버스는 기준선보다 수익률이 낮고 MDD 방어도 더 좋아지지 않았다.",
        "- 커버드콜 ETF는 상장 이력이 짧고 상승장 참여율이 낮아 전체 기간/활성 구간 모두 기준선 대체 후보로 약하다.",
        "- `US high-div stocks`는 성과가 좋아 보이는 구간이 있으나 현재 고배당주를 사후 선택한 survivorship/selection bias가 커서 승격 후보로 보지 않는다.",
        "- 인컴형 자산은 `yoon2_final_fx`의 목적(상승 참여 + 방어)을 대체하기보다, 별도 현금흐름/저변동성 목적 계좌로 분리 테스트하는 편이 낫다.",
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
        f"- CSV: `{out_csv.relative_to(ROOT)}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def candidate_table() -> list[dict[str, Any]]:
    symbols = sorted({
        s for variant in VARIANTS
        for s in ((variant.us_universe or []) + (variant.kr_universe or []))
    })
    out = []
    for sym in symbols:
        try:
            close = load_yfinance(sym, "1d", "max")["close"].dropna()
            out.append({
                "symbol": sym,
                "start": str(close.index[0].date()),
                "end": str(close.index[-1].date()),
                "last_close": float(close.iloc[-1]),
                "bars": int(len(close)),
            })
        except Exception:
            out.append({"symbol": sym, "start": "-", "end": "-", "last_close": 0.0, "bars": 0})
    return out


def main() -> None:
    base = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx.json").read_text("utf-8"))
    handler = Yoon1kHandler()
    rows = []
    for variant in VARIANTS:
        print(f"running {variant.key} ...", flush=True)
        rows.append(run_variant(handler, base, variant))
        r = rows[-1]
        print(
            f"  full CAGR {r['full_cagr']*100:.2f}% MDD {r['full_mdd']*100:.2f}% "
            f"Sharpe {r['full_sharpe']:.3f} final {r['full_dca_final']:,.0f}"
        )
    out_json, out_md = write_report(rows, candidate_table())
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
