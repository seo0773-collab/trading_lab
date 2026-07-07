"""final0706 신용스프레드 게이트 GFC 대리검증 — 위기 표본 n=1→2 (개선계획 B1).

배경: 게이트(HYG/IEF)의 실질 개선이 22개 드로다운 에피소드 중 2020-02 코로나
1건에 집중된다는 표본 한계가 있고, final0706 백테스트에는 .KS 데이터 갭으로
2008년이 아예 없다(`final0706_data_gap_diagnosis.md`). HYG는 2007-04 상장이라
**GFC(2007-10~2009-03)는 게이트 신호가 실존하는 구간** — yoon2m_gfc_proxy와 같은
yoon1j 원본 SPDR 유니버스(1998~) US 슬리브에 게이트를 얹어 GFC를 통과시킨다.

게이트 로직은 yoon1k._credit_spread_severity와 동일하게 복제한다(원본은 2조
합성전략 내부라 단독 호출 불가): ratio=HYG/IEF, 10일 모멘텀 < -2% AND 20일선
아래, 2거래일 연속 확인, 발동 시 노출 0.5배(축소분은 IEF 수익률로 대피), 신호
1봉 지연. 상위 결합이 월간 리밸런스인 원본과 달리 여기서는 일간 적용(반응성
상한의 근사) — 대신 노출 변화 회전율에 실비용(편도 20.103bp+슬리피지 5bp)을 물린다.

판정 기준(사전 등록, 결과 보고 변경 금지 — 개선계획 B1):
  ① GFC에서 게이트가 발동하는가(에피소드 내 발동일 존재)
  ② 리드타임 — 고점 대비 첫 발동이 선행(-)인가 사후(+)인가
  ③ 게이트 적용 시 GFC MDD가 개선되는가
  ①~③ 모두 불충족이면 "게이트=코로나 특화"로 스펙에 명시하고 게이트 없는
  파생과 병행 페이퍼로 강등한다.

사용:  PYTHONPATH=src:scripts .venv/bin/python scripts/profile_sizing/final0706_credit_gate_gfc_proxy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.metrics import equity_metrics  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402

# final0706.json risk_gate와 동일 파라미터 (동결값)
LOOKBACK = 10
SMA_LOOKBACK = 20
THRESHOLD = -0.02
OFF_SCALE = 0.5
MIN_CONSECUTIVE = 2
SIGNAL_LAG = 1
COST_PER_SIDE = (20.103 + 5.0) / 10_000.0  # KIS US 편도 + 슬리피지

# SPY 기준 주요 드로다운 에피소드(HYG 상장 2007-04 이후)
EPISODES = [
    ("2007-10-09", "2009-03-09", "GFC"),
    ("2011-04-29", "2011-10-03", "2011 미국 신용등급"),
    ("2015-05-21", "2016-02-11", "2015-16 위안화/유가"),
    ("2018-09-20", "2018-12-24", "2018 Q4 금리"),
    ("2020-02-19", "2020-03-23", "2020 코로나"),
    ("2022-01-03", "2022-10-12", "2022 인플레 약세장"),
    ("2025-02-19", "2025-04-08", "2025 관세 쇼크"),
]


def load_close(sym: str) -> pd.Series:
    px = load_yfinance(sym, "1d", "max")["close"]
    idx = pd.DatetimeIndex(pd.to_datetime(px.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(np.asarray(px, dtype=float), index=idx).dropna()


def gate_weak(hyg: pd.Series, ief: pd.Series) -> pd.Series:
    """yoon1k._credit_spread_severity + min_consecutive 복제(이진 weak)."""
    idx = hyg.index.intersection(ief.index).sort_values()
    ratio = (hyg.reindex(idx) / ief.reindex(idx)).ffill()
    mom = ratio.pct_change(LOOKBACK)
    below_sma = ratio < ratio.rolling(SMA_LOOKBACK).mean()
    weak = (mom < THRESHOLD) & below_sma
    weak = weak.fillna(False)
    confirmed = weak.rolling(MIN_CONSECUTIVE).sum() >= MIN_CONSECUTIVE
    return (confirmed & weak).astype(bool)


def run_sleeve(top_k: int = 3) -> pd.Series:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / "yoon1j.json").read_text("utf-8")))
    cfg["top_k"] = top_k
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    return art.equity


def apply_gate(sleeve: pd.Series, ief: pd.Series, weak: pd.Series) -> pd.Series:
    idx = sleeve.index.intersection(ief.index)
    idx = idx[idx >= weak.index[0]]
    exposure = weak.reindex(idx).ffill().fillna(False).map({True: OFF_SCALE, False: 1.0})
    exposure = exposure.shift(SIGNAL_LAG).fillna(1.0)
    r_sleeve = sleeve.reindex(idx).ffill().pct_change().fillna(0.0)
    r_ief = ief.reindex(idx).ffill().pct_change().fillna(0.0)
    turnover = exposure.diff().abs().fillna(0.0)
    r = exposure * r_sleeve + (1.0 - exposure) * r_ief - turnover * COST_PER_SIDE
    return (1.0 + r).cumprod()


def main() -> None:
    hyg, ief, spy = load_close("HYG"), load_close("IEF"), load_close("SPY")
    weak = gate_weak(hyg, ief)
    print(f"게이트 신호 구간: {weak.index[0].date()}~{weak.index[-1].date()}, "
          f"발동일 {int(weak.sum())}일 ({weak.mean()*100:.1f}%)\n")

    sleeve = run_sleeve(3)
    gated = apply_gate(sleeve, ief, weak)
    base = sleeve.reindex(gated.index).ffill()
    base = base / base.iloc[0]

    rows = []
    for peak, trough, label in EPISODES:
        p, t = pd.Timestamp(peak), pd.Timestamp(trough)
        ep_weak = weak[(weak.index >= p) & (weak.index <= t)]
        pre_window = weak[(weak.index >= p - pd.Timedelta(days=200)) & (weak.index <= t)]
        first = pre_window[pre_window].index.min() if pre_window.any() else None
        lead = None
        if first is not None:
            trading_days = weak.index[(weak.index >= min(first, p)) & (weak.index <= max(first, p))]
            lead = int(len(trading_days) - 1) * (1 if first > p else -1)
        w_spy = spy[(spy.index >= p) & (spy.index <= t)]
        w_base = base[(base.index >= p) & (base.index <= t)]
        w_gated = gated[(gated.index >= p) & (gated.index <= t)]
        dd = lambda s: float((s / s.cummax() - 1.0).min()) if len(s) else None  # noqa: E731
        rows.append({
            "episode": label, "peak": peak, "trough": trough,
            "spy_dd": dd(w_spy), "sleeve_dd": dd(w_base), "gated_dd": dd(w_gated),
            "first_trigger": str(first.date()) if first is not None else None,
            "lead_days": lead,
            "gated_frac": float(ep_weak.mean()) if len(ep_weak) else None,
        })

    df = pd.DataFrame(rows).set_index("episode")
    fmt = {c: (lambda x: "   n/a" if pd.isna(x) else f"{x*100:7.1f}%")
           for c in ("spy_dd", "sleeve_dd", "gated_dd", "gated_frac")}
    print(df[["spy_dd", "sleeve_dd", "gated_dd", "first_trigger", "lead_days", "gated_frac"]]
          .to_string(formatters=fmt))

    print("\n=== 전 구간(HYG 상장 2007-04~) 성과 — US 슬리브 단독 대리 ===")
    perf = {}
    for label, s in (("sleeve(게이트 없음)", base), ("sleeve+신용게이트", gated)):
        m = equity_metrics(s)
        perf[label] = m
        print(f"  {label:20s} CAGR {m['cagr']*100:6.2f}%  MDD {m['max_drawdown']*100:7.2f}%  "
              f"Sharpe {m['sharpe']:.3f}")
    sw = spy[spy.index >= base.index[0]]
    m = equity_metrics(sw / sw.iloc[0])
    perf["SPY"] = m
    print(f"  {'SPY 동일기간':20s} CAGR {m['cagr']*100:6.2f}%  MDD {m['max_drawdown']*100:7.2f}%  "
          f"Sharpe {m['sharpe']:.3f}")

    out = ROOT / "reports" / "profile_sizing" / "final0706_credit_gate_gfc_proxy.json"
    out.write_text(json.dumps({
        "params": {"lookback": LOOKBACK, "sma_lookback": SMA_LOOKBACK,
                   "threshold": THRESHOLD, "off_scale": OFF_SCALE,
                   "min_consecutive": MIN_CONSECUTIVE, "cost_per_side": COST_PER_SIDE},
        "episodes": rows, "full_period": perf,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
