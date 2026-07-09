"""bb_crossing_sweep.py 연장 — 진입신호(BB length=20, mult=2.5, offset=0, 최적치)는 그대로
쓰고 포지션 **사이징만 마틴게일**로 교체한 이분법적(항상 롱 또는 숏) 평균회귀 전략.

규칙 변경 요약 (bb_crossing_sweep과의 차이):
    - bb_crossing_sweep: 하단밴드 "재상향 크로스"=롱 진입, 베이시스 크로스=청산(무포지션 허용).
    - 이 스크립트: 항상 롱/숏 중 하나를 유지해야 하므로 진입 트리거를 "밴드 이탈(breach)"로
      바꾼다 — close가 하단밴드 이하로 내려가면 롱, 상단밴드 이상으로 올라가면 숏.
      반대쪽 밴드에 닿으면 전량 청산 + 즉시 반대 포지션 신규 진입(플립).
    - 보유 중 역행(가격 기준 등간격, add_step)마다 직전 랏의 mult배 사이즈로 물타기(martingale
      add), 최대 max_levels 단계까지만. 그 이상 역행해도 추가 진입은 멈추고 손절 없이 보유
      (사용자 지정: 손절 없음, 최대 단계로만 리스크 캡). 파산(자본<=0)이면 해당 구간 이후
      equity=0으로 고정(실제 마진콜/전량손실을 정직하게 반영).

Usage:
    .venv/bin/python3 scripts/bb_martingale_meanreversion.py
출력:
    reports/bb_martingale_sweep.json  — 전체 그리드 결과(ruin_rate 포함)
    reports/bb_martingale_sweep.md    — 요약 리포트
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "profile_sizing"))
from bb_crossing_sweep import (  # noqa: E402
    DATA_DIR, ROUNDTRIP_FEE, bollinger_bands, load_ohlcv, perf_slice, slice_windows,
)

OUT_DIR = ROOT / "reports"
UNIVERSE_CSV = OUT_DIR / "bb_crossing_universe.csv"

# BB 신호 파라미터 — bb_crossing_sweep 스윕에서 찾은 최적치로 고정(이 스크립트는 스윕 안 함).
BB_LENGTH, BB_MULT, BB_OFFSET = 20, 2.5, 0

FEE_SIDE = ROUNDTRIP_FEE / 2.0       # 편도 비용(진입/추가/플립청산 각각에 적용)
BASE_LOT_FRACTION = 0.05             # 최초 진입 랏 = 자본의 5% (마틴게일 배수의 기준 단위)

# --- 마틴게일 그리드 ---
MULT_GRID = (1.5, 2.0, 3.0)             # 물타기 시 직전 랏 대비 배수
ADD_STEP_GRID = (0.02, 0.03, 0.05, 0.08)  # 최근 물타기 가격 대비 -N% 역행마다 추가 진입
MAX_LEVELS_GRID = (2, 3, 4, 5)           # 최초 진입 이후 추가 진입 최대 횟수


def load_universe() -> pd.DataFrame:
    if UNIVERSE_CSV.exists():
        return pd.read_csv(UNIVERSE_CSV)
    from bb_crossing_sweep import screen_universe  # noqa: E402
    universe = screen_universe()
    universe.to_csv(UNIVERSE_CSV, index=False)
    return universe


def position_pnl(lot_sizes: list[float], entry_prices: list[float],
                  pos_dir: int, price: float) -> float:
    return sum(s * pos_dir * (price / e - 1.0) for s, e in zip(lot_sizes, entry_prices))


def simulate_martingale(close: pd.Series, upper: pd.Series, lower: pd.Series,
                         trade_from: int, add_step: float, mult: float,
                         max_levels: int,
                         regime_ok: np.ndarray | None = None,
                         base_lot_fraction: float = BASE_LOT_FRACTION,
                         fee_side: float = FEE_SIDE) -> tuple[pd.DataFrame, pd.Series, dict]:
    """regime_ok[i]=False면 그 봉에서는 신규 진입/물타기/플립을 모두 정지한다(신규 행위만
    막음 — 이미 보유 중인 포지션은 손대지 않고 마크투마켓만 계속한다). None이면 항상 True."""
    c = close.to_numpy(float)
    up = upper.to_numpy(float)
    lo = lower.to_numpy(float)
    n = len(c)
    valid = np.isfinite(c) & np.isfinite(up) & np.isfinite(lo)
    ok = regime_ok if regime_ok is not None else np.ones(n, dtype=bool)

    equity = np.ones(n)
    flips: list[dict] = []
    ruined = False
    ruin_i: int | None = None
    max_exposure = 0.0

    pos_dir = 0
    levels = 0
    lot_sizes: list[float] = []
    entry_prices: list[float] = []
    last_add_price = np.nan
    flip_entry_i = trade_from
    cap = 1.0

    def lot_size(level: int) -> float:
        return base_lot_fraction * (mult ** level)

    def open_position(i: int, direction: int) -> None:
        nonlocal pos_dir, levels, lot_sizes, entry_prices, last_add_price, flip_entry_i, cap
        pos_dir = direction
        levels = 0
        lot_sizes = [lot_size(0)]
        entry_prices = [c[i]]
        last_add_price = c[i]
        flip_entry_i = i
        cap -= lot_sizes[0] * fee_side

    for i in range(trade_from, n):
        if i > 0 and not valid[i]:
            equity[i] = equity[i - 1]
            continue

        if pos_dir == 0:
            if ok[i] and c[i] <= lo[i]:
                open_position(i, 1)
            elif ok[i] and c[i] >= up[i]:
                open_position(i, -1)
            else:
                equity[i] = equity[i - 1] if i > 0 else 1.0
                continue
            equity[i] = cap
            max_exposure = max(max_exposure, sum(lot_sizes))
            continue

        unreal = position_pnl(lot_sizes, entry_prices, pos_dir, c[i])
        if cap + unreal <= 0:
            flips.append({
                "entry_time": close.index[flip_entry_i], "exit_time": close.index[i],
                "direction": pos_dir, "avg_entry_price": float(np.average(
                    entry_prices, weights=lot_sizes)),
                "exit_price": c[i], "levels_used": levels, "total_size": sum(lot_sizes),
                "net_return": max(unreal, -1.0), "exit_reason": "ruin",
            })
            ruined, ruin_i = True, i
            equity[i:] = 0.0
            break

        flip = ok[i] and (
            (pos_dir == 1 and c[i] >= up[i]) or (pos_dir == -1 and c[i] <= lo[i]))
        if flip:
            total_size = sum(lot_sizes)
            realized = unreal
            exit_cost = total_size * fee_side
            cap = cap + realized - exit_cost
            flips.append({
                "entry_time": close.index[flip_entry_i], "exit_time": close.index[i],
                "direction": pos_dir, "avg_entry_price": float(np.average(
                    entry_prices, weights=lot_sizes)),
                "exit_price": c[i], "levels_used": levels, "total_size": total_size,
                "net_return": realized - exit_cost, "exit_reason": "flip",
            })
            if cap <= 0:
                ruined, ruin_i = True, i
                equity[i:] = 0.0
                break
            open_position(i, -pos_dir)
            equity[i] = cap
            max_exposure = max(max_exposure, sum(lot_sizes))
            continue

        if ok[i] and levels < max_levels:
            adverse = (last_add_price - c[i]) / last_add_price * pos_dir
            if adverse >= add_step:
                levels += 1
                lot_sizes.append(lot_size(levels))
                entry_prices.append(c[i])
                last_add_price = c[i]
                cap -= lot_sizes[-1] * fee_side

        total_size = sum(lot_sizes)
        max_exposure = max(max_exposure, total_size)
        equity[i] = cap + position_pnl(lot_sizes, entry_prices, pos_dir, c[i])

    eq_series = pd.Series(equity, index=close.index)
    stats = {"ruined": ruined, "ruin_pos": ruin_i, "max_exposure": max_exposure}
    return pd.DataFrame(flips), eq_series, stats


def run_combo(universe: pd.DataFrame, add_step: float, mult: float, max_levels: int,
              regime_fn=None) -> dict:
    """regime_fn(df, full_index) -> np.ndarray[bool] 를 넘기면 그 마스크로 신규 진입/물타기/
    플립을 게이팅한다(bb_martingale_regime_gate.py 참고). None이면 항상 True(게이트 없음)."""
    train_rows, holdout_rows = [], []
    ruin_train = ruin_holdout = 0
    n_stocks = 0
    for symbol in universe["symbol"]:
        df = load_ohlcv(DATA_DIR / f"{symbol}.parquet")
        if df is None:
            continue
        close = df["close"]
        full, train_start, holdout_start = slice_windows(close)
        basis, upper, lower = bollinger_bands(full, BB_LENGTH, BB_MULT, BB_OFFSET)
        regime_ok = regime_fn(df, full.index) if regime_fn is not None else None
        trades, equity, stats = simulate_martingale(
            full, upper, lower, trade_from=train_start,
            add_step=add_step, mult=mult, max_levels=max_levels, regime_ok=regime_ok)
        n_stocks += 1

        ruin_i = stats["ruin_pos"]
        if ruin_i is not None and ruin_i < holdout_start:
            ruin_train += 1
            ruin_holdout += 1
        elif ruin_i is not None:
            ruin_holdout += 1
            tr_perf = perf_slice(equity, train_start, holdout_start)
            if tr_perf["sharpe"] is not None:
                train_rows.append({"symbol": symbol, **tr_perf})
        else:
            tr_perf = perf_slice(equity, train_start, holdout_start)
            ho_perf = perf_slice(equity, holdout_start, len(equity))
            if tr_perf["sharpe"] is not None:
                train_rows.append({"symbol": symbol, **tr_perf})
            if ho_perf["sharpe"] is not None:
                holdout_rows.append({"symbol": symbol, **ho_perf})

    tr_df = pd.DataFrame(train_rows)
    ho_df = pd.DataFrame(holdout_rows)
    return {
        "add_step": add_step, "mult": mult, "max_levels": max_levels,
        "n_stocks": n_stocks,
        "train_ruin_rate": ruin_train / n_stocks if n_stocks else None,
        "holdout_ruin_rate": ruin_holdout / n_stocks if n_stocks else None,
        "train_median_sharpe": float(tr_df["sharpe"].median()) if len(tr_df) else None,
        "train_median_mdd": float(tr_df["max_drawdown"].median()) if len(tr_df) else None,
        "train_median_total_return": float(tr_df["total_return"].median()) if len(tr_df) else None,
        "holdout_median_sharpe": float(ho_df["sharpe"].median()) if len(ho_df) else None,
        "holdout_median_mdd": float(ho_df["max_drawdown"].median()) if len(ho_df) else None,
        "holdout_median_total_return": float(ho_df["total_return"].median()) if len(ho_df) else None,
    }


def main() -> None:
    print("1) 유니버스 로드 (bb_crossing_universe.csv 재사용)...", flush=True)
    universe = load_universe()
    print(f"   {len(universe)}종목: {', '.join(universe['symbol'].tolist())}")

    total = len(MULT_GRID) * len(ADD_STEP_GRID) * len(MAX_LEVELS_GRID)
    print(f"2) 마틴게일 그리드 스윕: mult{MULT_GRID} x add_step{ADD_STEP_GRID} x "
          f"max_levels{MAX_LEVELS_GRID} = {total}개 조합 x {len(universe)}종목 "
          f"(BB length={BB_LENGTH}, stdev={BB_MULT}, offset={BB_OFFSET} 고정)", flush=True)

    rows = []
    done = 0
    for mult in MULT_GRID:
        for add_step in ADD_STEP_GRID:
            for max_levels in MAX_LEVELS_GRID:
                rows.append(run_combo(universe, add_step, mult, max_levels))
                done += 1
                if done % 12 == 0 or done == total:
                    print(f"   {done}/{total} 완료", flush=True)

    grid = pd.DataFrame(rows)
    out_json = OUT_DIR / "bb_martingale_sweep.json"
    out_json.write_text(json.dumps({
        "base_lot_fraction": BASE_LOT_FRACTION,
        "bb_params": {"length": BB_LENGTH, "mult": BB_MULT, "offset": BB_OFFSET},
        "universe": universe.to_dict(orient="records"),
        "grid": grid.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    survivors = grid[(grid["train_ruin_rate"] == 0) & (grid["holdout_ruin_rate"] == 0)]
    ranked = survivors.sort_values("train_median_sharpe", ascending=False) \
        if len(survivors) else grid.sort_values("holdout_ruin_rate")
    top = ranked.head(8)

    lines = [
        "# 볼린저밴드 마틴게일 평균회귀 (항상 롱/숏 유지, 물타기 사이징)",
        "",
        f"- 진입신호: bb_crossing_sweep 최적치 고정 (length={BB_LENGTH}, stdev={BB_MULT}, "
        f"offset={BB_OFFSET}). 하단밴드 이탈=롱, 상단밴드 이탈=숏, 반대 밴드 터치 시 전량 "
        "청산+즉시 반대 포지션 진입(플립). 평소 무포지션 없음.",
        f"- 물타기: 최근 진입가 대비 역행 -add_step%마다 직전 랏의 mult배로 추가 진입, "
        "최대 max_levels 단계까지만(그 이후는 손절 없이 보유 — 리스크는 단계 상한으로만 캡).",
        f"- 최초 랏 크기 = 자본의 {BASE_LOT_FRACTION*100:.0f}% (기준 단위, 미조정)",
        f"- 왕복비용 가정은 bb_crossing_sweep과 동일(편도 {FEE_SIDE*1e4:.1f}bp), 매 진입/추가/"
        "청산마다 편도 비용 부과.",
        f"- 파산 처리: 자본(mark-to-market)이 0 이하가 되면 그 시점부터 equity=0 고정 "
        "(실거래 마진콜/전량손실에 해당, 이후 구간 Sharpe 등은 정의 불가로 결측 처리하고 "
        "ruin_rate로 별도 집계).",
        "",
        f"- 전체 {total}개 조합 중 train/holdout 양쪽 모두 파산 0%인 조합: {len(survivors)}개",
        "",
    ]

    if len(survivors):
        lines += [
            "## 파산 없는 조합 중 train Sharpe 상위",
            "",
            "| mult | add_step | max_levels | train Sharpe | train MDD | holdout Sharpe | "
            "holdout MDD |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, r in top.iterrows():
            lines.append(
                f"| {r['mult']} | {r['add_step']*100:.0f}% | {int(r['max_levels'])} | "
                f"{r['train_median_sharpe']:.3f} | {r['train_median_mdd']*100:.1f}% | "
                f"{r['holdout_median_sharpe']:.3f} | {r['holdout_median_mdd']*100:.1f}% |"
            )
        lines.append("")
        best = top.iloc[0]
        lines += [
            "## 최적 조합 (파산 0% 중 train Sharpe 기준)",
            "",
            f"- mult={best['mult']}, add_step={best['add_step']*100:.0f}%, "
            f"max_levels={int(best['max_levels'])}",
            f"- train median Sharpe={best['train_median_sharpe']:.3f}, "
            f"MDD={best['train_median_mdd']*100:.1f}%",
            f"- holdout median Sharpe={best['holdout_median_sharpe']:.3f}, "
            f"MDD={best['holdout_median_mdd']*100:.1f}%",
            "",
        ]
    else:
        lines += [
            "## 전 조합 파산 발생 — 파산율 가장 낮은 조합",
            "",
            "| mult | add_step | max_levels | train ruin_rate | holdout ruin_rate |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, r in top.iterrows():
            lines.append(
                f"| {r['mult']} | {r['add_step']*100:.0f}% | {int(r['max_levels'])} | "
                f"{r['train_ruin_rate']*100:.0f}% | {r['holdout_ruin_rate']*100:.0f}% |"
            )
        lines.append("")

    lines += [
        "## 파산율 전체 그리드 요약 (mult별)",
        "",
        "| mult | 평균 train ruin_rate | 평균 holdout ruin_rate |",
        "| ---: | ---: | ---: |",
    ]
    for m, g in grid.groupby("mult"):
        lines.append(
            f"| {m} | {g['train_ruin_rate'].mean()*100:.1f}% | "
            f"{g['holdout_ruin_rate'].mean()*100:.1f}% |"
        )
    lines += [
        "",
        "## 주의",
        "",
        "- 손절 없음(사용자 지정) — max_levels 도달 후에도 역행이 계속되면 무제한 미실현손실을 "
        "그대로 보유한다. ruin_rate가 0%가 아닌 조합은 실거래에 쓰면 안 된다.",
        "- 마틴게일 특성상 파산하지 않는 한 승률/Sharpe가 좋아 보이는 착시가 크다 — 표에 없는 "
        "파산 조합들이 꼬리위험을 대부분 흡수한다(생존편향 주의).",
        "- BB 신호 자체는 재상향크로스(무포지션 허용) 방식에서 밴드이탈(즉시 반대 플립) 방식으로 "
        "바꿔 재정의했다(항상 시장 노출 요구사항 때문) — bb_crossing_sweep 결과와 트레이드 정의가 다르다.",
        "",
        "## 파일",
        "",
        f"- 전체 그리드: `{out_json.relative_to(ROOT)}`",
    ]
    out_md = OUT_DIR / "bb_martingale_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")
    if len(survivors):
        best = top.iloc[0]
        print(f"\n=== 파산 0% 최적: mult={best['mult']} add_step={best['add_step']*100:.0f}% "
              f"max_levels={int(best['max_levels'])} train Sharpe={best['train_median_sharpe']:.3f} ===")
    else:
        print("\n=== 파산 0%인 조합이 없습니다 — 그리드 전 구간에서 리스크 재검토 필요 ===")


if __name__ == "__main__":
    main()
