"""yoon2a 로컬 시뮬 원장 — KIS 잔고/주문 미사용 전진 NAV 트래커.

KIS 모의계좌가 원/달러 예수금 풀을 고정·불균형(USD풀이 시드의 ~24배)으로 선지급해
통합 리밸런싱 NAV가 무의미한 문제(2026-06-29 확인)를 우회한다. 브로커 대신 **고정
시드(기본 1천만원)** 로 목표 저널(paper_trade가 적립)을 시장가에 마크투마켓해 전진
NAV를 결정론적으로 계산한다. 백테스트와 동일 의미의 OOS 트랙 + 룩어헤드 없음 검증.

  update : 이번 달 목표(3슬리브 저널)를 원장에 1건 기록(월 1회, 시장시간 무관).
  report : inception부터 일별 NAV 재구성 → 50/50 벤치 대비 수익·낙폭·Sharpe.

한계: 결정론적 시뮬(슬리피지·체결오차 없음=낙관 상한). 실체결 아님.

사용:  PYTHONPATH=src python scripts/profile_sizing/live/paper_ledger.py update
       PYTHONPATH=src python scripts/profile_sizing/live/paper_ledger.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

SEED_KRW = 10_000_000.0
STRATEGY = "yoon2a"
LEDGER = ROOT / "var" / "live" / "yoon2a_ledger.json"
BENCH = {"USD": "SPY", "KRW": "069500.KS"}  # 50/50 SPY원화 + KODEX200
FX_SYM = "KRW=X"
INTERVAL = "1d"


def _journal_targets(cfg_id: str) -> dict[str, float]:
    jp = ROOT / "var" / "paper_trading" / f"{cfg_id}_journal.jsonl"
    lines = [ln for ln in jp.read_text().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"{cfg_id} 저널 비어있음 → paper_trade.py --strategy {cfg_id}")
    return json.loads(lines[-1]).get("targets", {})


def combined_targets() -> tuple[dict[str, dict], str]:
    """yoon2a 3슬리브 → 종목별 KRW-base 목표비중 {sym:{weight,currency}} + 최신 기준일."""
    cfg = json.loads((ROOT / "configs" / "strategies" / f"{STRATEGY}.json").read_text("utf-8"))
    subs = cfg["sub_strategies"]
    wsum = sum(float(s["weight"]) for s in subs) or 1.0
    out: dict[str, dict] = {}
    data_through = ""
    for s in subs:
        sw = float(s["weight"]) / wsum
        cur = str(s.get("currency", "KRW"))
        jp = ROOT / "var" / "paper_trading" / f"{s['config']}_journal.jsonl"
        snap = json.loads([l for l in jp.read_text().splitlines() if l.strip()][-1])
        data_through = max(data_through, str(snap.get("data_through", "")))
        targ = snap.get("targets", {})
        # 정규화하지 않음: 목표합<1 이면 그만큼 방어 현금(전략 고유 방어버퍼) 유지.
        for sym, w in targ.items():
            out[sym] = {"weight": sw * float(w), "currency": cur}
    return out, data_through


def _load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text("utf-8"))
    return {"seed_krw": SEED_KRW, "strategy": STRATEGY, "events": []}


def cmd_update() -> int:
    led = _load_ledger()
    targets, data_through = combined_targets()
    today = pd.Timestamp.today().normalize()
    ym = today.strftime("%Y-%m")
    ev = {"date": str(today.date()), "ym": ym, "data_through": data_through,
          "targets": targets}
    # 월 1건: 같은 달 이벤트 있으면 갱신, 없으면 추가.
    events = [e for e in led["events"] if e.get("ym") != ym]
    events.append(ev)
    led["events"] = sorted(events, key=lambda e: e["date"])
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[원장] {ym} 목표 기록 ({len(targets)}종, 기준일 {data_through}). "
          f"누적 이벤트 {len(led['events'])}건 → {LEDGER}")
    for sym, t in sorted(targets.items(), key=lambda kv: -kv[1]["weight"]):
        print(f"   {sym:12} {t['weight']*100:5.2f}% [{t['currency']}]")
    return 0


def _price_panel_krw(symbols: set[str], start: pd.Timestamp) -> pd.DataFrame:
    """종목별 KRW-base 종가 패널(USD종목 × 환율)."""
    fx = load_yfinance(FX_SYM, INTERVAL, "max")["close"]
    cols = {}
    for sym in symbols:
        px = load_yfinance(sym, INTERVAL, "max")["close"].dropna()
        krw = sym.upper().endswith(".KS") or sym.upper().endswith(".KQ")
        if not krw:
            f = fx.reindex(px.index).ffill()
            px = px * f
        cols[sym] = px
    panel = pd.DataFrame(cols).sort_index()
    return panel[panel.index >= start].ffill()


def _bench_krw(start: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Series:
    """50/50 SPY원화+KODEX200, 월간 리밸런싱, SEED 기준."""
    fx = load_yfinance(FX_SYM, INTERVAL, "max")["close"]
    spy = load_yfinance(BENCH["USD"], INTERVAL, "max")["close"].dropna()
    kdx = load_yfinance(BENCH["KRW"], INTERVAL, "max")["close"].dropna()
    spy_krw = (spy * fx.reindex(spy.index).ffill()).reindex(idx).ffill()
    kdx = kdx.reindex(idx).ffill()
    reb = rebalance_dates(idx, "monthly")
    r_s = spy_krw.pct_change().fillna(0.0).to_numpy()
    r_k = kdx.pct_change().fillna(0.0).to_numpy()
    v_s = v_k = 0.5
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        v_s *= 1.0 + r_s[i]
        v_k *= 1.0 + r_k[i]
        out[i] = v_s + v_k
        if t in reb:
            tot = v_s + v_k
            v_s = v_k = tot * 0.5
    return pd.Series(out, index=idx) * SEED_KRW


def cmd_report() -> int:
    led = _load_ledger()
    events = led.get("events", [])
    if not events:
        print("원장 비어있음 → 먼저 update")
        return 1
    seed = float(led.get("seed_krw", SEED_KRW))
    syms = {s for e in events for s in e["targets"]}
    start = pd.Timestamp(events[0]["date"])
    panel = _price_panel_krw(syms, start)
    idx = panel.index
    if len(idx) < 2:
        print(f"가격 데이터 부족(이벤트 {len(events)}건, inception {start.date()}). "
              "전진 기록이 더 쌓이면 NAV가 그려집니다.")
        return 0

    # 이벤트일 → 그 날짜 이후 첫 거래일에 리밸런스 적용
    reb_targets: dict[pd.Timestamp, dict] = {}
    for e in events:
        d = pd.Timestamp(e["date"])
        future = idx[idx >= d]
        if len(future):
            reb_targets[future[0]] = e["targets"]

    units: dict[str, float] = {s: 0.0 for s in syms}
    cash = seed
    nav = np.empty(len(idx))
    for i, t in enumerate(idx):
        mtm = cash + sum(units[s] * panel.at[t, s] for s in syms
                         if pd.notna(panel.at[t, s]))
        nav[i] = mtm
        if t in reb_targets:
            tw = reb_targets[t]
            invested = 0.0
            for s in syms:
                w = tw.get(s, {}).get("weight", 0.0)
                px = panel.at[t, s]
                if w > 0 and pd.notna(px) and px > 0:
                    units[s] = mtm * w / px
                    invested += units[s] * px
                else:
                    units[s] = 0.0
            cash = mtm - invested
    nav_s = pd.Series(nav, index=idx)
    bench = _bench_krw(start, idx)

    p = performance(nav_s / nav_s.iloc[0], (nav_s / nav_s.iloc[0]).pct_change().fillna(0), INTERVAL)
    b = performance(bench / bench.iloc[0], (bench / bench.iloc[0]).pct_change().fillna(0), INTERVAL)
    days = (idx[-1] - idx[0]).days
    print(f"=== yoon2a 로컬 전진 원장 ({start.date()} ~ {idx[-1].date()}, {days}일, 시드 {seed:,.0f}원) ===")
    print(f"리밸런스 {len(reb_targets)}회 · 현재 평가 {nav_s.iloc[-1]:,.0f}원\n")
    def row(name, eq, pf):
        print(f"  {name:16} 평가 {eq.iloc[-1]:>14,.0f}원 · 수익 {pf['total_return']*100:+6.2f}% "
              f"· MDD {(pf['max_drawdown'] or 0)*100:6.2f}% "
              f"· Sharpe {pf['sharpe'] if pf['sharpe'] is not None else float('nan'):.3f}")
    row("yoon2a(전진)", nav_s, p)
    row("50/50 벤치", bench, b)
    if days < 35:
        print("\n  ⚠️ 전진 구간이 짧아 수익/낙폭은 아직 노이즈 — 매월 update로 누적됩니다.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["update", "report"])
    args = ap.parse_args(argv)
    return cmd_update() if args.cmd == "update" else cmd_report()


if __name__ == "__main__":
    raise SystemExit(main())
