#!/usr/bin/env python
"""yoon1k(이중통화 fund-of-funds) dry-run 주문표 — 제출 절대 안 함.

단일통화 도구(kis_rebalance)는 yoon1k 를 못 읽는다(universe 없음). yoon1k 는
하위 슬리브 yoon1j(USD)·yoon1j_kr(KRW) 의 현재 목표를 통화별 50/50 으로 결합한다.

흐름:
  1) 각 슬리브 페이퍼 저널의 최신 목표비중 로드(없으면 paper_trade.py 로 먼저 생성)
  2) 총자본을 통화 50/50 으로 분할(USD 측은 환율로 환산)
  3) KIS 현재가(직전 일봉 종가)로 통화별 rebalance_plan → 주문표
  4) 출력만. 주문/잔고 API 는 호출하지 않음(가장 안전한 dry-run).

가격은 KIS daily_history 마지막 종가(US=USD, KR=KRW). 환율은 Yahoo KRW=X(현재 KIS 미지원).
실제 모의계좌 잔고 연동(보유 차감)·주문 제출은 다음 단계(국내 주문 메서드 추가 필요).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_dryrun.py \
      --capital-krw 10000000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_dotenv(path: Path) -> None:
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from profile_sizing.live.execution_layer import integer_execution_plan  # noqa: E402
from profile_sizing.live.kis_client import KISClient  # noqa: E402
from profile_sizing.live.rebalance import load_latest_target  # noqa: E402


def load_sleeves(strategy: str):
    """yoon1k 계열 config 의 sub_strategies → (config, currency, weight, exchange, min_trade)."""
    cfg = json.loads((ROOT / "configs" / "strategies" / f"{strategy}.json")
                     .read_text("utf-8"))
    out = []
    for sub in cfg["sub_strategies"]:
        cur = str(sub.get("currency", cfg.get("base_currency", "KRW")))
        exch = None if cur == "KRW" else "AMEX"   # 미국 ETF=NYSE Arca→AMS
        min_tv = 50000.0 if cur == "KRW" else 50.0
        out.append((sub["config"], cur, float(sub.get("weight", 0.5)), exch, min_tv))
    return out


def last_close(client: KISClient, sym: str, exch: str | None) -> float:
    end = __import__("pandas").Timestamp.today().normalize()
    start = (end - __import__("pandas").Timedelta(days=14)).strftime("%Y-%m-%d")
    df = client.daily_history(sym, interval="1d", start=start,
                              exchange=exch or "NASD")
    return float(df["close"].iloc[-1]) if not df.empty else 0.0


def fx_krw_per_usd() -> float:
    from trading_lab.market_data import load_cumulative_yfinance
    fx = load_cumulative_yfinance("KRW=X", "1d", "5d")
    return float(fx["close"].iloc[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="yoon2",
                    help="2단계 계층 config id (기본 yoon2)")
    ap.add_argument("--capital-krw", type=float, default=10_000_000,
                    help="총 투자원금(원). 통화 50/50 분할.")
    args = ap.parse_args(argv)
    sleeves = load_sleeves(args.strategy)

    try:
        client = KISClient()
    except Exception as e:  # noqa: BLE001
        print(f"[자격증명 없음] {e}")
        return 1

    fx = fx_krw_per_usd()
    print(f"=== {args.strategy} DRY-RUN (주문 미제출) ===")
    print(f"총원금 {args.capital_krw:,.0f}원 · 환율 {fx:,.1f}원/$ · 통화 50/50\n")

    grand_orders = 0
    for cfg_id, cur, w, exch, min_tv in sleeves:
        journal = ROOT / "var" / "paper_trading" / f"{cfg_id}_journal.jsonl"
        if not journal.exists():
            print(f"[{cfg_id}] 저널 없음 → 먼저 실행: "
                  f"paper_trade.py --strategy {cfg_id}\n")
            continue
        snap = load_latest_target(journal)
        targets = snap["targets"]
        sleeve_krw = args.capital_krw * w
        cash = sleeve_krw / fx if cur == "USD" else sleeve_krw  # 계좌통화 현금

        prices = {}
        for sym in targets:
            kis_sym = sym  # 국내는 .KS 포함해도 _is_domestic 가 처리
            prices[sym] = last_close(client, kis_sym, exch)

        plan = integer_execution_plan(targets, holdings={}, prices=prices, cash=cash,
                                      min_trade_value=min_tv)
        unit = "$" if cur == "USD" else "원"
        print(f"── {cfg_id} 슬리브 [{cur}] · 배정 {cash:,.2f}{unit} "
              f"(목표 기준일 {snap['data_through']}, 레짐 {snap.get('market_regime')})")
        print(f"   {'종목':12} {'side':4} {'수량':>8} {'현재가':>12} {'예상금액':>14} {'목표w':>7}")
        for o in plan["orders"]:
            print(f"   {o['symbol']:12} {o['side']:4} {o['qty']:8.0f} "
                  f"{o['price']:12,.2f} {o['est_value']:14,.2f} {o['target_w']*100:6.1f}%")
        if plan["skipped_no_price"]:
            print(f"   [가격없음 제외] {plan['skipped_no_price']}")
        print(f"   소계: 주문 {len(plan['orders'])}건 · 계좌평가 {plan['account_value']:,.2f}{unit}"
              f" · 예상현금 {plan['expected_cash']:,.2f}{unit}"
              f" · 재현율 {plan['target_fit']*100:.1f}%\n")
        grand_orders += len(plan["orders"])

    print(f"=== 총 주문 {grand_orders}건 (DRY-RUN, 제출 안 됨) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
