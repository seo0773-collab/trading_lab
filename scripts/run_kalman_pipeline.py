
#!/usr/bin/env python3
"""
run_kalman_pipeline.py
======================
임의의 실제 자산 -> rolling profile 피처 -> Adaptive Kalman -> 예측/검증 파이프라인.

단계
----
1. 데이터 로드      : yfinance(--symbol) / CSV(--csv) / 합성(--synthetic)
2. 피처 추출        : indicators.flat_chart  -> data/processed/{name}_features.parquet
3. 파라미터 식별    : κ (AR1 피팅), h_s·h_d (OLS), R (잔차 부트스트랩), ℓ̄
4. Walk-forward 필터: 매 봉 update 후 k ∈ {1, 4, 24} 예측 저장
5. 검증            : PIT 보정도, 방향 적중률 vs 베이스라인(RW / AR1)
6. 리포트          : reports/{name}_report.txt, _forecast.csv, _plots.png

사용 예
-------
    python scripts/run_kalman_pipeline.py --symbol BTC-USD --interval 1h --period 720d
    python scripts/run_kalman_pipeline.py --csv data/raw/eth_1h.csv --name ETH
    python scripts/run_kalman_pipeline.py --synthetic        # 드라이런
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flat_chart import FlatChartConfig, compute_features  # noqa: E402
from kalman import AdaptiveKalman, KalmanParams           # noqa: E402

HORIZONS = (1, 4, 24, 48, 72)
WARMUP = 300  # 파라미터 식별 후 필터 burn-in 봉 수


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_yfinance(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # 미설치 시: pip install yfinance

    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna(subset=["close"])


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    tcol = next((c for c in ("timestamp", "time", "date", "datetime") if c in df.columns), None)
    if tcol is None:
        raise ValueError("CSV에 timestamp/time/date/datetime 컬럼이 필요합니다")
    ts = df[tcol]
    df.index = (pd.to_datetime(ts, unit="ms") if np.issubdtype(ts.dtype, np.number) and ts.iloc[0] > 1e11
                else pd.to_datetime(ts))
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 누락된 컬럼: {missing}")
    cols = need + (["volume"] if "volume" in df.columns else [])
    return df[cols].sort_index()


def make_synthetic(n: int = 8000, seed: int = 7) -> pd.DataFrame:
    """드라이런용: 레짐 전환이 있는 합성 시계열 (κ 적응 확인용)."""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) % 3000 < 1500, 0.008, 0.018)
    drift = np.where(np.arange(n) % 4000 < 2000, 0.0002, -0.0001)
    ret = rng.normal(drift, vol)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, vol / 2)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = rng.lognormal(10, 1, n) * (1 + 5 * np.abs(ret) / vol.mean())
    return pd.DataFrame(
        dict(open=open_, high=high, low=low, close=close, volume=volume),
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


# ----------------------------------------------------------------------
# 3. Parameter identification (in-sample 앞 구간만 사용)
# ----------------------------------------------------------------------
def identify_params(feats: pd.DataFrame, ident_frac: float = 0.4) -> tuple[KalmanParams, dict]:
    f = feats.dropna(subset=["m_fast", "w_fast", "m_slow"]).copy()
    n_id = int(len(f) * ident_frac)
    fi = f.iloc[:n_id]
    info: dict = {"n_ident": n_id}

    m, mu = fi["m_fast"].to_numpy(), fi["m_slow"].to_numpy()

    # --- κ: AR(1) on (m - μ) ---
    d = m - mu
    x0, x1 = d[:-1], d[1:]
    phi = float(np.dot(x0, x1) / max(np.dot(x0, x0), 1e-12))
    phi = np.clip(phi, 0.5, 0.9999)
    kappa = float(np.clip(1.0 - phi, 1e-4, 0.05))
    info["ar1_phi"] = phi
    info["kappa"] = kappa
    info["half_life_bars"] = float(np.log(2) / kappa)

    # --- ṁ proxy: m_fast의 평활 1차차분 ---
    mdot_proxy = pd.Series(m, index=fi.index).diff().ewm(span=8).mean().to_numpy()

    # --- h_s, h_d: OLS  z_ch = h · ṁ + ε ---
    def ols_gain(z_ch: np.ndarray) -> tuple[float, float]:
        mask = np.isfinite(z_ch) & np.isfinite(mdot_proxy)
        if mask.sum() < 100:
            return 0.0, np.inf
        zc, md = z_ch[mask], mdot_proxy[mask]
        var_md = float(np.var(md))
        if var_md < 1e-14:
            return 0.0, float(np.var(zc))
        h = float(np.cov(zc, md)[0, 1] / var_md)
        resid_var = float(np.var(zc - h * md))
        # 유의성 약하면 채널 비활성화 (R 무한대 효과)
        corr = np.corrcoef(zc, md)[0, 1]
        if abs(corr) < 0.05:
            h = 0.0
        return h, max(resid_var, 1e-8)

    h_s, r_s = ols_gain(fi["s_fast"].to_numpy())
    h_d, r_d = ols_gain(fi["dc"].to_numpy())
    info["h_s"], info["h_d"] = h_s, h_d

    # --- R: 고주파 잔차 분산 (관측 - 짧은 EMA) ---
    def hf_var(x: np.ndarray) -> float:
        s = pd.Series(x).ewm(span=6).mean().to_numpy()
        r = x - s
        return float(np.nanvar(r))

    r_m = max(hf_var(m), 1e-8)
    r_l = max(hf_var(np.log(fi["w_fast"].to_numpy())), 1e-8)
    info["r_diag_raw"] = (r_m, r_l, r_s, r_d)

    l_bar = float(np.nanmean(np.log(fi["w_fast"].to_numpy())))

    # --- Q0: 상태 변화 스케일 기반 휴리스틱 (adaptive가 미세 조정) ---
    q_m = max(np.nanvar(np.diff(pd.Series(m).ewm(span=24).mean().to_numpy())), 1e-12)
    params = KalmanParams(
        kappa=kappa, theta=0.02,
        kappa_w=kappa, theta_w=0.05, l_bar=l_bar,
        h_s=h_s, h_d=h_d,
        q0_diag=(q_m * 0.1, q_m * 0.02, q_m * 0.1, q_m * 0.02),
        r_diag=(r_m, r_l, r_s if np.isfinite(r_s) else 1.0,
                r_d if np.isfinite(r_d) else 1.0),
    )
    return params, info


# ----------------------------------------------------------------------
# 4. Walk-forward filtering + forecasting
# ----------------------------------------------------------------------
def run_filter(feats: pd.DataFrame, params: KalmanParams) -> pd.DataFrame:
    f = feats.dropna(subset=["m_fast", "w_fast", "m_slow", "cycle"]).copy()
    kf = AdaptiveKalman(params)

    cycle = f["cycle"].to_numpy()
    cycle_slope = pd.Series(cycle).diff().ewm(span=24).mean().to_numpy()

    rows = []
    Z = f[["m_fast", "w_fast", "s_fast", "dc"]].to_numpy()
    MU = f["m_slow"].to_numpy()
    MC = f["mult_close"].to_numpy()

    for t in range(len(f)):
        z = Z[t].copy()
        z[1] = np.log(z[1]) if np.isfinite(z[1]) and z[1] > 0 else np.nan
        r = kf.step(z, MU[t])
        row = dict(
            m_filt=r["x"][0], mdot_filt=r["x"][1],
            q_scale=r["q_scale"], nis=r["nis"],
        )
        if t >= WARMUP:
            for k in HORIZONS:
                fc = kf.forecast(k, MU[t], cycle[t], MC[t],
                                 cycle_slope[t] if np.isfinite(cycle_slope[t]) else 0.0)
                row[f"mhat_{k}"] = fc.m_hat
                row[f"sig_{k}"] = fc.m_sigma
                row[f"pup_{k}"] = fc.p_up
                row[f"price_mid_{k}"] = fc.price_mid
                row[f"price_lo_{k}"] = fc.price_lo
                row[f"price_hi_{k}"] = fc.price_hi
        rows.append(row)

    out = pd.DataFrame(rows, index=f.index)
    return pd.concat([f, out], axis=1)


# ----------------------------------------------------------------------
# 4.5 Sigma calibration (식별 구간에서만 추정 -> OOS에 적용, walk-forward 준수)
# ----------------------------------------------------------------------
def calibrate_sigma(res: pd.DataFrame, ident_frac: float = 0.4) -> dict:
    """상태 m(질량중심)의 불확실성만으로는 개별 봉 배수의 산포를 못 담으므로,
    식별 구간 잔차로 horizon별 σ 스케일 계수를 추정해 전 구간에 적용한다.
    p_up도 보정된 σ로 재계산."""
    from scipy.stats import norm

    n_id = int(len(res) * ident_frac)
    cal = {}
    for k in HORIZONS:
        seg = res.iloc[WARMUP:n_id]
        resid = (seg["mult_close"].shift(-k) - seg[f"mhat_{k}"]).dropna()
        sig_rms = float(np.sqrt(np.nanmean(seg[f"sig_{k}"] ** 2)))
        c = float(resid.std() / sig_rms) if sig_rms > 0 and len(resid) > 100 else 1.0
        c = float(np.clip(c, 1.0, 100.0))
        cal[k] = c

        res[f"sig_{k}"] = res[f"sig_{k}"] * c
        z = (res[f"mhat_{k}"] - res["mult_close"]) / res[f"sig_{k}"]
        res[f"pup_{k}"] = norm.cdf(z)
        half = (res[f"price_hi_{k}"] - res[f"price_lo_{k}"]) / 2.0
        mid = res[f"price_mid_{k}"]
        res[f"price_lo_{k}"] = mid - half * c
        res[f"price_hi_{k}"] = mid + half * c
    return cal


# ----------------------------------------------------------------------
# 5. Validation
# ----------------------------------------------------------------------
def validate(res: pd.DataFrame, ident_frac: float = 0.4) -> str:
    from scipy.stats import norm

    lines = ["=" * 64, "VALIDATION (out-of-sample: 식별 구간 이후만)", "=" * 64]
    n_id = int(len(res) * ident_frac)
    oos = res.iloc[max(n_id, WARMUP):].copy()

    for k in HORIZONS:
        mhat, sig, pup = oos[f"mhat_{k}"], oos[f"sig_{k}"], oos[f"pup_{k}"]
        actual = oos["mult_close"].shift(-k)
        now = oos["mult_close"]
        mask = mhat.notna() & actual.notna() & (sig > 0)
        if mask.sum() < 100:
            lines.append(f"[k={k}] 표본 부족"); continue

        mh, sg, ac, nw, pu = (x[mask].to_numpy() for x in (mhat, sig, actual, now, pup))

        # --- PIT 보정도 ---
        pit = norm.cdf((ac - mh) / sg)
        hist, _ = np.histogram(pit, bins=10, range=(0, 1))
        pit_dev = float(np.abs(hist / hist.sum() - 0.1).max())  # 균등이면 0
        cover_1s = float(((ac > mh - sg) & (ac < mh + sg)).mean())  # 이상적 ≈ 0.683

        # --- 방향 적중률 ---
        dir_pred = np.sign(mh - nw)
        dir_real = np.sign(ac - nw)
        valid = dir_real != 0
        hit = float((dir_pred[valid] == dir_real[valid]).mean())

        # 확신도 상위 30%만 (p_up이 0.5에서 먼 구간) -> trading 후보 구간
        conf = np.abs(pu - 0.5)
        thr = np.quantile(conf, 0.7)
        hi = valid & (conf >= thr)
        hit_conf = float((dir_pred[hi] == dir_real[hi]).mean()) if hi.sum() > 30 else np.nan

        # --- 베이스라인 ---
        # RW: 50%. AR(1)-OU 베이스라인: μ 방향 단순 회귀 예측
        mu_arr = oos["m_slow"][mask].to_numpy()
        dir_ou = np.sign(mu_arr - nw)
        hit_ou = float((dir_ou[valid] == dir_real[valid]).mean())

        lines += [
            f"\n[horizon k={k} bars]  n={int(mask.sum())}",
            f"  방향 적중률          : {hit:.3f}   (RW 0.500 / OU-naive {hit_ou:.3f})",
            f"  확신 상위30% 적중률  : {hit_conf:.3f}" if np.isfinite(hit_conf) else "  확신 상위30% 적중률  : n/a",
            f"  1σ 커버리지          : {cover_1s:.3f}  (목표 0.683)",
            f"  PIT 최대 편차        : {pit_dev:.3f}  (0에 가까울수록 보정 양호, >0.05면 σ 스케일 조정 필요)",
        ]

    lines += [
        "",
        "해석 가이드:",
        "  - 방향 적중률이 RW와 OU-naive 둘 다 못 이기면 엣지 없음 -> 모델 재검토",
        "  - 1σ 커버리지 < 0.683 이면 σ 과소 (r_inflation 또는 q_clip_hi 상향)",
        "  - 확신 상위30% 적중률이 전체보다 높아야 p_up 기반 trading이 성립",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 6. Plots
# ----------------------------------------------------------------------
def make_plots(res: pd.DataFrame, name: str, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = res.iloc[-min(len(res), 2000):]
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    ax = axes[0]
    ax.plot(tail.index, tail["close"], lw=0.8, color="k", label="close")
    ax.plot(tail.index, tail["price_mid_24"], lw=0.9, color="tab:blue", label="24h forecast mid")
    ax.fill_between(tail.index, tail["price_lo_24"], tail["price_hi_24"],
                    alpha=0.2, color="tab:blue", label="±1σ")
    ax.plot(tail.index, tail["lower_price"], lw=0.7, color="tab:red", alpha=0.7, label="lower pct price")
    ax.plot(tail.index, tail["upper_price"], lw=0.7, color="tab:green", alpha=0.7, label="upper pct price")
    ax.set_title(f"{name} — price & 24-bar forecast band"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(tail.index, tail["mult_close"], lw=0.6, color="gray", alpha=0.7, label="mult close")
    ax.plot(tail.index, tail["m_fast"], lw=0.9, color="tab:orange", label="m_fast (obs)")
    ax.plot(tail.index, tail["m_filt"], lw=1.1, color="tab:blue", label="m (kalman)")
    ax.plot(tail.index, tail["m_slow"], lw=0.9, color="tab:purple", ls="--", label="μ = m_slow")
    ax.set_title("multiple space: observation vs filtered state"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    for k, c in zip(HORIZONS, ("tab:gray", "tab:orange", "tab:red",
                               "tab:purple", "tab:green")):
        ax.plot(tail.index, tail[f"pup_{k}"], lw=0.8, color=c, label=f"P(up) k={k}")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_ylim(0, 1); ax.set_title("direction probability"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    ax.plot(tail.index, tail["q_scale"], lw=0.8, color="tab:brown")
    ax.set_title("adaptive Q scale (NIS covariance matching)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="yfinance 심볼 (예: BTC-USD, AAPL)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="720d")
    ap.add_argument("--csv", help="OHLCV CSV 경로")
    ap.add_argument("--name", help="출력 파일 이름 prefix")
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터 드라이런")
    ap.add_argument("--fast-window", type=int, default=120)
    ap.add_argument("--slow-window", type=int, default=720)
    ap.add_argument("--cycle-len", type=int, default=200)
    args = ap.parse_args()

    if args.synthetic:
        df, name = make_synthetic(), "SYNTH"
    elif args.csv:
        df, name = load_csv(args.csv), args.name or Path(args.csv).stem
    elif args.symbol:
        df, name = load_yfinance(args.symbol, args.interval, args.period), \
                   args.name or args.symbol.replace("-", "")
    else:
        ap.error("--symbol / --csv / --synthetic 중 하나가 필요합니다")

    print(f"[1/6] data: {name}  bars={len(df)}  {df.index[0]} ~ {df.index[-1]}")

    cfg = FlatChartConfig(cycle_len=args.cycle_len,
                          fast_window=args.fast_window,
                          slow_window=args.slow_window)
    feats = compute_features(df, cfg)
    proc = ROOT / "data" / "processed" / f"{name}_features.parquet"
    proc.parent.mkdir(parents=True, exist_ok=True)
    try:
        feats.to_parquet(proc)
        print(f"[2/6] features -> {proc}")
    except ImportError:
        proc = proc.with_suffix(".csv")
        feats.to_csv(proc)
        print(f"[2/6] features -> {proc} (pyarrow 미설치, CSV 저장)")

    params, info = identify_params(feats)
    print(f"[3/6] κ={info['kappa']:.5f} (반감기 {info['half_life_bars']:.0f}봉), "
          f"h_s={info['h_s']:.4f}, h_d={info['h_d']:.4f}")
    if info["h_d"] == 0.0:
        print("      ⚠ dc 채널 유의성 없음 -> 비활성화됨 (Δc 선행성 미확인)")

    res = run_filter(feats, params)
    print(f"[4/6] walk-forward filter done, rows={len(res)}")

    cal = calibrate_sigma(res)
    print(f"[4.5] σ calibration factors: " +
          ", ".join(f"k={k}: ×{c:.1f}" for k, c in cal.items()))

    report = validate(res)
    print("[5/6] validation\n" + report)

    rep_dir = ROOT / "reports"; rep_dir.mkdir(exist_ok=True)
    (rep_dir / f"{name}_report.txt").write_text(
        f"params: {params}\nident: {info}\n\n{report}\n", encoding="utf-8")
    fc_cols = ["open", "close", "mult_close", "m_fast", "m_filt", "m_slow", "q_scale"] + \
              [f"{p}_{k}" for k in HORIZONS for p in ("mhat", "sig", "pup", "price_mid", "price_lo", "price_hi")]
    res[fc_cols].to_csv(rep_dir / f"{name}_forecast.csv")
    make_plots(res, name, rep_dir / f"{name}_plots.png")
    print(f"[6/6] reports -> {rep_dir}/{name}_report.txt, _forecast.csv, _plots.png")


if __name__ == "__main__":
    main()
