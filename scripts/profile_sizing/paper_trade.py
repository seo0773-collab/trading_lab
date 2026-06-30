#!/usr/bin/env python
"""페이퍼 트레이딩(전진 검증) — 최신 데이터 기준 '지금 들고 있어야 할' 목표 포트폴리오와
레짐 상태를 출력하고, 실행마다 저널에 스냅샷을 적립한다.

전략은 월간 리밸런스라, '현재 목표'는 가장 최근 월말 리밸런스에서 산출된 비중(다음
리밸런스까지 유지)이다. 엔진의 sim["last_target"]/["last_rebal_date"]를 그대로 사용해
백테스트 로직과 100% 일치시킨다(무누수: 직전 봉 신호로 산출).

반복 실행하면 var/paper_trading/<strategy>_journal.jsonl 에 스냅샷이 쌓여 실시간 전진
기록이 된다. 직전 스냅샷과 비교해 리밸런스/비중 변화를 보여준다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1b
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import compute_universe, simulate_portfolio  # noqa: E402
from trading_lab.portfolio_universes import SECTOR_INDEX  # noqa: E402


def _load_all(universe, cfg, mf, sf):
    from run_kalman_pipeline import load_yfinance
    panels = {}
    for s in universe:
        try:
            d = load_yfinance(s, cfg.interval, cfg.period)
            if d is not None and not d.empty:
                panels[s] = d
        except Exception:  # noqa: BLE001
            continue
    spy = None
    if mf.get("enabled"):
        try:
            spy = load_yfinance(str(mf.get("symbol", "SPY")),
                                cfg.interval, cfg.period)["close"]
        except Exception:  # noqa: BLE001
            spy = None
    sect = {}
    if sf.get("enabled"):
        for tk in sorted(set(SECTOR_INDEX.values())):
            try:
                sect[tk] = load_yfinance(tk, cfg.interval, cfg.period)["close"]
            except Exception:  # noqa: BLE001
                continue
    return panels, spy, (sect or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="yoon1b")
    args = ap.parse_args(argv)

    cfg_path = ROOT / "configs" / "strategies" / f"{args.strategy}.json"
    raw = json.loads(cfg_path.read_text())
    if str(raw.get("kind", "")) == "buy_hold":
        # 단일종목 buy&hold 슬리브(예: yoon2a의 SSO). 포트폴리오 시뮬레이션 없이
        # 목표 {symbol:1.0} 을 그대로 저널에 적립한다(리밸런스 불필요·항상 보유).
        return _buy_hold_journal(args.strategy, raw)
    cfg = config_from_dict(raw)
    mf, sf = raw.get("market_filter") or {}, raw.get("sector_filter") or {}
    top_k = int(raw.get("top_k", 20))
    rebal = str(raw.get("rebalance_freq", "monthly"))
    gain = float(raw.get("exposure_gain", 1.0))

    print(f"[{args.strategy}] 최신 데이터 로드 ...", flush=True)
    panels, spy, sect = _load_all(list(raw["universe"]), cfg, mf, sf)
    scores, prices = compute_universe(panels, cfg)
    if prices.empty:
        raise RuntimeError("데이터 로드 실패")

    sim = simulate_portfolio(
        scores, prices, cfg, top_k=top_k, rebal_freq=rebal,
        market_close=spy, market_ma_len=int(mf.get("ma_len", sf.get("ma_len", 200))),
        market_off_scale=float(mf.get("off_scale", 0.5)), exposure_gain=gain,
        sector_close=sect, symbol_sector=(SECTOR_INDEX if sect else None),
        sector_off_scale=float(sf.get("off_scale", 0.5)))

    target = sim["last_target"]
    rebal_date = sim["last_rebal_date"]
    fc = sim["forecast"]
    last = fc.index[-1]
    exposure = float(fc["stock_exposure"].iloc[-1])
    cash = float(fc["cash_ratio"].iloc[-1])
    market_ok = float(fc["market_ok"].iloc[-1])

    snap = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "data_through": str(last.date()),
        "last_rebal_date": str(pd.Timestamp(rebal_date).date()),
        "exposure": round(exposure, 4),
        "cash": round(cash, 4),
        "market_regime": "정상" if market_ok >= 1.0 else f"약세(x{market_ok:g})",
        "n_holdings": len(target),
        "targets": {s: round(w, 4) for s, w in
                    sorted(target.items(), key=lambda kv: -kv[1])},
    }

    jdir = ROOT / "var" / "paper_trading"
    jdir.mkdir(parents=True, exist_ok=True)
    jpath = jdir / f"{args.strategy}_journal.jsonl"
    prev = None
    if jpath.exists():
        lines = [ln for ln in jpath.read_text().splitlines() if ln.strip()]
        if lines:
            prev = json.loads(lines[-1])
    with jpath.open("a") as fh:
        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")

    _print_report(snap, prev, sect is not None)
    print(f"\n저널 적립: {jpath}")
    return 0


def buy_hold_snapshot(strategy: str, symbol: str, data_through: str) -> dict:
    """단일종목 buy&hold 페이퍼 스냅샷(순수 — 단위테스트 대상). targets={symbol:1.0}.

    저널 키는 포트폴리오 전략과 동일(gather_sleeve 가 targets·data_through 를 소비)."""
    sym = str(symbol).upper()
    return {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "data_through": data_through,
        "last_rebal_date": data_through,
        "exposure": 1.0,
        "cash": 0.0,
        "market_regime": "정상(buy&hold)",
        "n_holdings": 1,
        "targets": {sym: 1.0},
    }


def _buy_hold_journal(strategy: str, raw: dict) -> int:
    """단일종목 buy&hold 슬리브의 페이퍼 목표를 적립한다(targets={symbol:1.0}).

    저널 키/경로 형식은 포트폴리오 전략과 동일해서 live 오케스트레이터의
    gather_sleeve(`targets` + stale 가드용 `data_through`)가 그대로 소비한다."""
    from run_kalman_pipeline import load_yfinance

    sym = str(raw["symbol"]).upper()
    interval, period = str(raw.get("interval", "1d")), str(raw.get("period", "max"))
    print(f"[{strategy}] {sym} buy&hold 최신 데이터 로드 ...", flush=True)
    data = load_yfinance(sym, interval, period)
    if data is None or data.empty:
        raise RuntimeError(f"데이터 로드 실패: {sym}")
    snap = buy_hold_snapshot(strategy, sym, str(data.index[-1].date()))

    jdir = ROOT / "var" / "paper_trading"
    jdir.mkdir(parents=True, exist_ok=True)
    jpath = jdir / f"{strategy}_journal.jsonl"
    with jpath.open("a") as fh:
        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
    print("\n" + "=" * 64)
    print(f" 페이퍼 트레이딩 현재 목표 — {strategy} (buy&hold)")
    print("=" * 64)
    print(f" 데이터 기준일 : {snap['data_through']}")
    print(f" 목표          : {sym} 100%")
    print(f"\n저널 적립: {jpath}")
    return 0


def _print_report(snap, prev, has_sector) -> None:
    print("\n" + "=" * 64)
    print(f" 페이퍼 트레이딩 현재 목표 — {snap['strategy']}")
    print("=" * 64)
    print(f" 데이터 기준일 : {snap['data_through']}")
    print(f" 최근 리밸런스 : {snap['last_rebal_date']} (다음 월말까지 유지)")
    print(f" 시장 레짐     : {snap['market_regime']}"
          + ("  + 섹터 레짐 필터 ON" if has_sector else ""))
    print(f" 주식 노출     : {snap['exposure']*100:.1f}%   "
          f"현금: {snap['cash']*100:.1f}%   보유 종목: {snap['n_holdings']}")
    print("-" * 64)
    print(f" {'종목':<8}{'목표비중':>10}   {'섹터':<8}")
    for s, w in snap["targets"].items():
        sec = SECTOR_INDEX.get(s, "-")
        print(f" {s:<8}{w*100:>9.2f}%   {sec:<8}")
    if prev:
        print("-" * 64)
        if prev["last_rebal_date"] != snap["last_rebal_date"]:
            print(f" ★ 직전 스냅샷({prev['data_through']}) 이후 리밸런스 발생 "
                  f"({prev['last_rebal_date']} → {snap['last_rebal_date']})")
            ins = set(snap["targets"]) - set(prev.get("targets", {}))
            outs = set(prev.get("targets", {})) - set(snap["targets"])
            if ins:
                print(f"   편입: {', '.join(sorted(ins))}")
            if outs:
                print(f"   편출: {', '.join(sorted(outs))}")
        else:
            print(f" (직전 스냅샷 {prev['data_through']} 이후 리밸런스 없음 — 목표 동일)")


if __name__ == "__main__":
    raise SystemExit(main())
