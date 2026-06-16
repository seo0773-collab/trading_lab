#!/usr/bin/env python
"""분기 재무제표 수집 → 정규화 parquet (finance_plan.txt §21 데이터 계층).

데이터 소스 = **SEC EDGAR companyfacts API**. yfinance 무료 분기재무가 종목당
~5~7분기뿐이라 단일 종목 민감도 학습이 불가능했던 문제(§27)를 해결한다. EDGAR는
10-Q/10-K 전체 이력(수십 분기)과 **실제 filing date(발표일)**를 제공하므로,
availability 모듈이 45/90일 보수룰 대신 진짜 발표일을 PIT 기준으로 쓸 수 있다.

표준 스키마: period_end, report_type, announce_date(=filed), revenue,
operating_income, net_income, total_equity, total_debt, operating_cashflow,
inventory, shares_outstanding, eps. → var/fundamentals/<SYMBOL>.parquet.

흐름(flow) 항목은 XBRL의 3개월 기간(context) 사실만 골라 분기값을 직접 얻고,
10-Q가 없는 4분기는 연간(10-K)에서 앞 3개 분기를 빼서 추정한다. 잔고(instant)
항목은 기간말 시점값을 그대로 쓴다. 네트워크 의존이라 단위 테스트 대상이 아니다.

Usage:
    python scripts/fetch_fundamentals.py --symbols AAPL,MSFT
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# SEC는 식별 가능한 User-Agent를 요구한다(없으면 403).
_UA = "trading_lab research seo0773@gmail.com"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# 표준 컬럼 ← us-gaap/dei concept 후보(처음 매칭되는 것 사용).
FLOW_CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "operating_cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
INSTANT_CONCEPTS = {
    "total_equity": ["StockholdersEquity"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_debt": ["DebtCurrent", "LongTermDebtCurrent"],
    "inventory": ["InventoryNet"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding",
    ],
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cik_for(symbol: str, table: dict) -> int:
    for row in table.values():
        if str(row["ticker"]).upper() == symbol.upper():
            return int(row["cik_str"])
    raise KeyError(f"CIK를 찾지 못함: {symbol}")


def _units(facts: dict, concepts: list[str]) -> list[dict]:
    """매칭되는 모든 concept의 사실을 병합한다.

    같은 항목이라도 기업이 연도별로 XBRL concept를 바꾸므로(예: SalesRevenueNet
    →Revenues→RevenueFromContractWithCustomer) 첫 concept만 쓰면 과거 구간이
    통째로 빈다. 후보를 모두 모으고, 같은 기간 충돌은 이후 dedup(최초 신고)이 정리.
    """
    merged: list[dict] = []
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept in node:
                units = node[concept]["units"]
                merged.extend(units.get("USD") or next(iter(units.values())))
    return merged


def _flow_quarterly(items: list[dict]) -> dict[pd.Timestamp, tuple[float, pd.Timestamp]]:
    """3개월 기간 사실 → {period_end: (val, filed)}. 동일 기간은 최초 신고 채택."""
    out: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for it in items:
        if "start" not in it or "end" not in it:
            continue
        start, end = pd.Timestamp(it["start"]), pd.Timestamp(it["end"])
        if not (80 <= (end - start).days <= 100):
            continue
        filed = pd.Timestamp(it["filed"])
        prev = out.get(end)
        if prev is None or filed < prev[1]:
            out[end] = (float(it["val"]), filed)
    return out


def _flow_annual(items: list[dict]) -> dict[pd.Timestamp, float]:
    out: dict[pd.Timestamp, float] = {}
    for it in items:
        if "start" not in it or "end" not in it:
            continue
        start, end = pd.Timestamp(it["start"]), pd.Timestamp(it["end"])
        if 350 <= (end - start).days <= 380:
            out.setdefault(end, float(it["val"]))
    return out


def _instant(items: list[dict]) -> dict[pd.Timestamp, float]:
    out: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for it in items:
        if "end" not in it or "start" in it:
            continue
        end, filed = pd.Timestamp(it["end"]), pd.Timestamp(it["filed"])
        prev = out.get(end)
        if prev is None or filed < prev[1]:
            out[end] = (float(it["val"]), filed)
    return {k: v[0] for k, v in out.items()}


def _derive_q4(quarterly: dict, annual: dict) -> None:
    """10-K 연간에서 같은 회계연도 3분기를 빼 Q4를 채운다(in-place)."""
    for fy_end, total in annual.items():
        if fy_end in quarterly:
            continue
        prior = [e for e in quarterly if fy_end - pd.Timedelta(days=360) < e < fy_end]
        if len(prior) == 3:
            q4 = total - sum(quarterly[e][0] for e in prior)
            filed = max(quarterly[e][1] for e in prior)  # 보수: 마지막 신고
            quarterly[fy_end] = (q4, filed)


def normalize(symbol: str, facts: dict) -> pd.DataFrame:
    flows = {}
    for col, concepts in FLOW_CONCEPTS.items():
        items = _units(facts, concepts) or []
        q = _flow_quarterly(items)
        _derive_q4(q, _flow_annual(items))
        flows[col] = q
    eps_q = _flow_quarterly(_units(facts, EPS_CONCEPTS) or [])
    instants = {
        col: _instant(_units(facts, concepts) or [])
        for col, concepts in INSTANT_CONCEPTS.items()
    }

    # 기간말 = 흐름 항목들의 분기말 합집합.
    period_ends = sorted(set().union(*[set(q) for q in flows.values()] or [set()]))
    rows = []
    for end in period_ends:
        ni = flows["net_income"].get(end)
        row = {
            "period_end": end,
            "report_type": "quarter",
            "announce_date": ni[1] if ni else pd.NaT,
        }
        for col, q in flows.items():
            row[col] = q.get(end, (None,))[0]
        row["eps"] = eps_q.get(end, (None,))[0]
        for col, d in instants.items():
            row[col] = d.get(end)
        ltd, cur = row.pop("long_term_debt"), row.pop("current_debt")
        row["total_debt"] = (ltd or 0.0) + (cur or 0.0) if (ltd or cur) else None
        rows.append(row)
    cols = [
        "period_end", "report_type", "announce_date", "revenue",
        "operating_income", "net_income", "operating_cashflow", "total_equity",
        "total_debt", "inventory", "shares_outstanding", "eps",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


def merge_fundamentals(
    existing: pd.DataFrame, incoming: pd.DataFrame
) -> pd.DataFrame:
    """Keep accumulated quarters while preferring newly fetched SEC values."""
    keys = ["period_end", "report_type"]
    old = existing.copy()
    new = incoming.copy()
    for frame in (old, new):
        frame["period_end"] = pd.to_datetime(frame["period_end"])
        frame["announce_date"] = pd.to_datetime(frame["announce_date"])
    old = old.set_index(keys)
    new = new.set_index(keys)
    merged = new.combine_first(old)
    columns = list(dict.fromkeys([
        *existing.columns,
        *incoming.columns,
    ]))
    return (
        merged.reset_index()
        .reindex(columns=columns)
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="쉼표 구분 심볼")
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "var" / "fundamentals"
    )
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    table = _get_json(_TICKERS_URL)
    for symbol in (s.strip() for s in args.symbols.split(",") if s.strip()):
        cik = _cik_for(symbol, table)
        facts = _get_json(_FACTS_URL.format(cik=cik))
        frame = normalize(symbol, facts)
        out = args.outdir / f"{symbol.upper()}.parquet"
        if out.exists():
            frame = merge_fundamentals(pd.read_parquet(out), frame)
        _write_parquet_atomic(frame, out)
        print(f"{out}: {len(frame)} quarters "
              f"({frame['period_end'].min().date()} ~ {frame['period_end'].max().date()})")
        time.sleep(0.2)  # SEC rate limit (<10 req/s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
