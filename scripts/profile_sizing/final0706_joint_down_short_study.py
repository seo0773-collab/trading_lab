"""final0706 레짐 파생 아이디어 스크리닝: 국장·미장 동반 하락 → 1일 숏.

사용자 제안: "국장 미장 둘다 떨어질때 숏을 잡아보는건 어때, 기간을 짧게해서"를
1일 보유 숏으로 구체화해 신호 빈도·기대값·비용 이후 순기대값을 실측한다.
final0706 자체(동결)는 건드리지 않고 이 스크립트 안에서만 검증한다.

신호 정의(인과적, look-ahead 없음):
  KR 프록시 = 069500.KS(final0706 KRW 벤치와 동일), US 프록시 = SPY(USD 벤치와 동일)
  kr_ret(d)      = KR d일 종가 수익률 (KR d일 장마감 시점에 확정)
  us_ret_prior(d)= KR d일 장마감 시점까지 "이미 종료된" 가장 최근 US 세션 수익률
                   (merge_asof backward, 당일 US 세션은 KR 장마감 시점엔 아직 미종료라 제외)
  signal(d) = kr_ret(d) < 0 AND us_ret_prior(d) < 0

숏 대상 3가지, 각각 신호일 종가 시점에 진입 가능한 "다음 미실현 수익률"을 타깃으로 함:
  KR 숏: fwd_1(KR, d)   = KR d일 종가 → d+1일 종가
  US 숏: fwd_1(US, d')  = signal(d) 시점 이후 처음 도래하는 US 세션의 1일 수익률
                          (merge_asof forward, d 포함 — 그 시점까지 미종료이므로 미래 타깃으로 유효)
  블렌드: (KR 숏 + US 숏) / 2 — final0706 구조(50:50)를 반영

비용: 왕복 20bp(편도 10bp, 이 랩의 표준 가정) 이후 순기대값도 함께 보고.
검정: Newey-West(lag=1) mean/se/t, 델타(=신호일 평균 - 전체일 평균) 유의성.
다중비교: m=3(KR/US/블렌드) Bonferroni(0.05/3).
워크포워드: ~2021 / 2022~2023 / 2024~ 3구간 부호 일치 확인.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_joint_down_short_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.cycle_state import forward_returns, newey_west_mean_test  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

KR_SYMBOL = "069500.KS"
US_SYMBOL = "SPY"
FEE_ROUNDTRIP = 20.0 / 10_000.0  # 왕복 20bp(이 랩 표준 가정)
WF_SPLITS = (("cycle1_~2021", None, "2021-12-31"),
             ("cycle2_2022~2023", "2022-01-01", "2023-12-31"),
             ("cycle3_2024~", "2024-01-01", None))


def load_close(symbol: str) -> pd.Series:
    raw = load_yfinance(symbol, "1d", "max")
    idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(np.asarray(raw["close"], dtype=float), index=idx).sort_index()


def build_frame() -> pd.DataFrame:
    kr_close = load_close(KR_SYMBOL)
    us_close = load_close(US_SYMBOL)
    kr_ret = kr_close.pct_change()
    us_ret = us_close.pct_change()
    kr_fwd1 = forward_returns(kr_close, (1,))["fwd_1"]
    us_fwd1 = forward_returns(us_close, (1,))["fwd_1"]

    kr_df = pd.DataFrame({"kr_ret": kr_ret, "kr_fwd1": kr_fwd1}).dropna(subset=["kr_ret"])
    us_ret_df = pd.DataFrame({"us_date": us_ret.index, "us_ret_prior": us_ret.to_numpy()}).dropna()
    us_fwd_df = pd.DataFrame({"us_date": us_fwd1.index, "us_fwd1": us_fwd1.to_numpy()})

    kr_df.index.name = "kr_date"
    left = kr_df.reset_index().sort_values("kr_date")

    prior = pd.merge_asof(left, us_ret_df.sort_values("us_date"),
                           left_on="kr_date", right_on="us_date",
                           direction="backward", allow_exact_matches=False)
    fwd = pd.merge_asof(left[["kr_date"]], us_fwd_df.sort_values("us_date"),
                         left_on="kr_date", right_on="us_date",
                         direction="forward", allow_exact_matches=True)

    out = left.copy()
    out["us_ret_prior"] = prior["us_ret_prior"].to_numpy()
    out["us_fwd1"] = fwd["us_fwd1"].to_numpy()
    out = out.set_index("kr_date").drop(columns=["us_date"], errors="ignore")
    out = out.dropna(subset=["kr_ret", "us_ret_prior"])
    out["signal"] = (out["kr_ret"] < 0) & (out["us_ret_prior"] < 0)
    out["blend_fwd1"] = out[["kr_fwd1", "us_fwd1"]].mean(axis=1, skipna=False)
    return out


def eval_target(df: pd.DataFrame, col: str) -> dict:
    sig = df.loc[df["signal"], col].to_numpy(dtype=float)
    sig = sig[np.isfinite(sig)]
    allv = df[col].to_numpy(dtype=float)
    allv = allv[np.isfinite(allv)]
    m_sig, se_sig, t_sig = newey_west_mean_test(sig, lag=1)
    m_all, se_all, _ = newey_west_mean_test(allv, lag=1)
    delta = m_sig - m_all if np.isfinite(m_sig) and np.isfinite(m_all) else float("nan")
    se_delta = float(np.sqrt(se_sig ** 2 + se_all ** 2)) if np.isfinite(se_sig) and np.isfinite(se_all) else float("nan")
    t_delta = delta / se_delta if se_delta and se_delta > 0 else float("nan")
    short_pnl_gross = -m_sig if np.isfinite(m_sig) else float("nan")
    short_pnl_net = short_pnl_gross - FEE_ROUNDTRIP if np.isfinite(short_pnl_gross) else float("nan")
    return {
        "n_signal": int(len(sig)), "n_all": int(len(allv)),
        "mean_fwd1_signal": m_sig, "se_signal": se_sig, "t_signal": t_sig,
        "mean_fwd1_all": m_all,
        "delta_signal_minus_all": delta, "se_delta": se_delta, "t_delta": t_delta,
        "short_expectancy_gross": short_pnl_gross,
        "short_expectancy_net_20bp": short_pnl_net,
    }


def walk_forward(df: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for name, start, end in WF_SPLITS:
        seg = df
        if start is not None:
            seg = seg[seg.index >= start]
        if end is not None:
            seg = seg[seg.index <= end]
        res = eval_target(seg, col)
        rows.append({"split": name, **res})
    return rows


def main() -> None:
    print("KR/US 프록시 로드 및 정렬 중...", flush=True)
    df = build_frame()
    n_total = len(df)
    n_signal = int(df["signal"].sum())
    print(f"전체 {n_total}일 중 신호 {n_signal}일 ({n_signal / n_total:.1%})", flush=True)

    targets = {"KR_069500": "kr_fwd1", "US_SPY": "us_fwd1", "blend_50_50": "blend_fwd1"}
    results = {name: eval_target(df, col) for name, col in targets.items()}
    wf = {name: walk_forward(df, col) for name, col in targets.items()}

    m = len(targets)
    ts = {name: r["t_delta"] for name, r in results.items()}
    from scipy.stats import norm
    pvals = {name: float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else float("nan") for name, t in ts.items()}
    bonferroni_sig = {name: (np.isfinite(p) and p < 0.05 / m) for name, p in pvals.items()}

    judgments = {}
    for name in targets:
        wf_rows = wf[name]
        same_sign = sum(
            1 for row in wf_rows
            if np.isfinite(row["delta_signal_minus_all"])
            and np.sign(row["delta_signal_minus_all"]) == np.sign(results[name]["delta_signal_minus_all"])
        )
        net_positive = results[name]["short_expectancy_net_20bp"] > 0 if np.isfinite(results[name]["short_expectancy_net_20bp"]) else False
        if bonferroni_sig[name] and same_sign >= 2 and net_positive:
            v = "채택"
        elif (abs(ts[name]) > 1.65 if np.isfinite(ts[name]) else False) and same_sign >= 2:
            v = "보류(방향 일치·유의수준 미달 또는 비용후 역전)"
        else:
            v = "기각"
        judgments[name] = {
            "verdict": v, "p_value": pvals[name], "bonferroni_sig(m=3)": bonferroni_sig[name],
            "wf_same_sign": f"{same_sign}/{len(wf_rows)}", "net_positive_after_20bp": net_positive,
        }

    payload = {
        "signal_definition": "kr_ret(d)<0 AND us_ret_prior(d)<0 (backward merge_asof, 당일 US 세션 제외)",
        "n_total_days": n_total, "n_signal_days": n_signal, "signal_freq": n_signal / n_total,
        "fee_roundtrip_bp": FEE_ROUNDTRIP * 10_000,
        "results": results, "walk_forward": wf, "judgments": judgments,
    }
    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_joint_down_short_study.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"wrote {out_json}")

    lines = [
        "# final0706 레짐 파생 스크리닝: 국장·미장 동반 하락 → 1일 숏",
        "",
        f"- 신호 정의: KR 당일 하락 AND 직전 완료 US 세션 하락(인과적 정렬, look-ahead 없음)",
        f"- 신호 빈도: {n_signal}/{n_total}일 ({n_signal / n_total:.1%})",
        f"- 비용 가정: 왕복 {FEE_ROUNDTRIP * 10_000:.0f}bp",
        "",
        "## 숏 기대값(신호일 vs 전체일, NW lag=1)",
        "",
        "| 대상 | n(신호) | 신호일 평균 fwd1 | 전체일 평균 | 델타 t | 숏기대값(총) | 숏기대값(순, -20bp) | Bonferroni유의(m=3) | WF부호일치 | 판정 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name in targets:
        r = results[name]
        j = judgments[name]
        lines.append(
            f"| {name} | {r['n_signal']} | {r['mean_fwd1_signal']*100:.3f}% | {r['mean_fwd1_all']*100:.3f}% | "
            f"{r['t_delta']:.2f} | {r['short_expectancy_gross']*100:.3f}% | {r['short_expectancy_net_20bp']*100:.3f}% | "
            f"{j['bonferroni_sig(m=3)']} | {j['wf_same_sign']} | **{j['verdict']}** |"
        )
    lines += ["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_joint_down_short_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {out_md}")
    print(json.dumps(judgments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
