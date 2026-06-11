
"""
kalman.py
=========
배수(cycle multiple) 공간 Adaptive Kalman Filter.

상태:  x = [m, ṁ, ℓ, ℓ̇]      (ℓ = log w)
관측:  z = [m_fast, log w_fast, s_fast, dc]

상태방정식 (OU 평균회귀 결합):
    m'  = m + ṁ
    ṁ'  = -κ·(m - μ(t)) + (1-θ)·ṁ           μ(t) = m_slow (외생 입력)
    ℓ'  = ℓ + ℓ̇
    ℓ̇'  = -κw·(ℓ - ℓ̄) + (1-θw)·ℓ̇

관측방정식:
    m_fast     = m            + v1
    log w_fast = ℓ            + v2
    s_fast     = h_s · ṁ      + v3    (비대칭도 -> 속도의 간접 관측)
    dc         = h_d · ṁ      + v4    (질량 유입 방향 -> 속도의 선행 관측)

적응 (innovation 기반 공분산 매칭):
    NIS(t) = ν' S⁻¹ ν,  ratio = EWMA(NIS)/dim(z)
    Q(t)   = Q0 · clip(ratio, q_clip_lo, q_clip_hi)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanParams:
    # OU / damping  (1h봉 기준 출발값; 파이프라인에서 AR(1) 피팅으로 갱신)
    kappa: float = 0.005       # m 평균회귀 강도 (반감기 ≈ ln2/κ 봉)
    theta: float = 0.02        # ṁ 감쇠
    kappa_w: float = 0.01      # log w 평균회귀
    theta_w: float = 0.05
    l_bar: float = np.log(0.2)  # log w 장기 평균 (데이터로 갱신)

    # 관측 게인 (s, dc -> ṁ). 파이프라인에서 OLS로 추정.
    h_s: float = 0.0
    h_d: float = 0.0

    # 노이즈
    q0_diag: tuple = (1e-8, 1e-8, 1e-8, 1e-8)   # 프로세스 노이즈 기본
    r_diag: tuple = (1e-4, 1e-3, 1e-2, 1e-2)    # 측정 노이즈 (오프라인 추정 × 인플레이션)
    r_inflation: float = 4.0   # rolling window 중첩 자기상관 보정

    # adaptive
    nis_lambda: float = 0.98   # EWMA forgetting
    q_clip_lo: float = 0.3
    q_clip_hi: float = 10.0


@dataclass
class ForecastResult:
    horizon: int
    m_hat: float       # 예측 배수 평균
    m_sigma: float     # 예측 배수 표준편차
    p_up: float        # P( mult(t+k) > 현재 close 배수 )
    price_mid: float   # 예측 가격 중앙값
    price_lo: float    # 1σ 하단
    price_hi: float    # 1σ 상단


class AdaptiveKalman:
    NX = 4  # [m, mdot, l, ldot]

    def __init__(self, params: KalmanParams | None = None):
        self.p = params or KalmanParams()
        self.x = np.zeros(self.NX)
        self.P = np.eye(self.NX) * 1e-2
        self.Q0 = np.diag(self.p.q0_diag)
        self.R = np.diag(self.p.r_diag) * self.p.r_inflation
        self.nis_ewma = float(self._nz())
        self.q_scale = 1.0
        self._initialized = False

    def _nz(self) -> int:
        return 4

    # ---------------- model matrices ----------------
    def F(self) -> np.ndarray:
        p = self.p
        return np.array([
            [1.0,          1.0,        0.0,          0.0],
            [-p.kappa,     1 - p.theta, 0.0,          0.0],
            [0.0,          0.0,        1.0,          1.0],
            [0.0,          0.0,        -p.kappa_w,   1 - p.theta_w],
        ])

    def u(self, mu: float) -> np.ndarray:
        """외생 입력: κ·μ(t), κw·ℓ̄"""
        p = self.p
        return np.array([0.0, p.kappa * mu, 0.0, p.kappa_w * p.l_bar])

    def H(self) -> np.ndarray:
        p = self.p
        return np.array([
            [1.0, 0.0,   0.0, 0.0],   # m_fast
            [0.0, 0.0,   1.0, 0.0],   # log w_fast
            [0.0, p.h_s, 0.0, 0.0],   # s_fast  ~ h_s·ṁ
            [0.0, p.h_d, 0.0, 0.0],   # dc      ~ h_d·ṁ
        ])

    # ---------------- filter ----------------
    def init_state(self, m0: float, l0: float):
        self.x = np.array([m0, 0.0, l0, 0.0])
        self.P = np.diag([1e-3, 1e-6, 1e-2, 1e-6])
        self._initialized = True

    def step(self, z: np.ndarray, mu: float) -> dict:
        """1봉 predict + update. z = [m_fast, log w_fast, s_fast, dc] (NaN 채널 자동 제외)."""
        if not self._initialized:
            self.init_state(z[0], z[1] if np.isfinite(z[1]) else self.p.l_bar)

        F, H_full = self.F(), self.H()
        Q = self.Q0 * self.q_scale

        # predict
        x_pred = F @ self.x + self.u(mu)
        P_pred = F @ self.P @ F.T + Q

        # NaN 채널 마스킹
        mask = np.isfinite(z)
        if not mask.any():
            self.x, self.P = x_pred, P_pred
            return dict(x=self.x.copy(), P=self.P.copy(), nis=np.nan, q_scale=self.q_scale)

        H = H_full[mask]
        R = self.R[np.ix_(mask, mask)]
        zm = z[mask]

        # update
        nu = zm - H @ x_pred
        S = H @ P_pred @ H.T + R
        S_inv = np.linalg.inv(S)
        K = P_pred @ H.T @ S_inv
        self.x = x_pred + K @ nu
        I_KH = np.eye(self.NX) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # Joseph form

        # adaptive Q via NIS covariance matching
        nis = float(nu @ S_inv @ nu)
        lam = self.p.nis_lambda
        self.nis_ewma = lam * self.nis_ewma + (1 - lam) * nis
        ratio = self.nis_ewma / mask.sum()
        self.q_scale = float(np.clip(ratio, self.p.q_clip_lo, self.p.q_clip_hi))

        return dict(x=self.x.copy(), P=self.P.copy(), nis=nis, q_scale=self.q_scale)

    # ---------------- k-step forecast ----------------
    def forecast(
        self, k: int, mu: float, cycle_now: float,
        mult_now: float, cycle_slope: float = 0.0,
    ) -> ForecastResult:
        """k봉 앞 배수의 가우시안 예측분포 -> 방향 확률 + 가격 밴드.

        μ(t)는 k스텝 동안 현재 값 고정 가정 (slow window라 k≤24에서 오차 미미).
        cycle은 현재 기울기 유지 선형 외삽.
        """
        F = self.F()
        Q = self.Q0 * self.q_scale
        x, P = self.x.copy(), self.P.copy()
        u = self.u(mu)
        for _ in range(k):
            x = F @ x + u
            P = F @ P @ F.T + Q

        m_hat = float(x[0])
        m_var = float(P[0, 0])
        # 예측분포에 분포폭(w) 일부 반영: 측정 자체의 산포 하한
        w_floor = float(np.exp(x[2])) * 0.0  # 보수적으로 0; 보정 단계에서 조정
        m_sigma = float(np.sqrt(max(m_var, 1e-12)) + w_floor)

        from math import erf, sqrt
        p_up = 0.5 * (1.0 + erf((m_hat - mult_now) / (m_sigma * sqrt(2.0))))

        cycle_k = cycle_now + cycle_slope * k
        return ForecastResult(
            horizon=k,
            m_hat=m_hat,
            m_sigma=m_sigma,
            p_up=float(p_up),
            price_mid=m_hat * cycle_k,
            price_lo=(m_hat - m_sigma) * cycle_k,
            price_hi=(m_hat + m_sigma) * cycle_k,
        )


# ----------------------------------------------------------------------
# 합성 OU 데이터로 필터 정합성 자가 테스트
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n, kappa_true, mu_true = 5000, 0.01, 1.0
    m = np.zeros(n); mdot = np.zeros(n)
    for t in range(1, n):
        mdot[t] = -kappa_true * (m[t-1] - mu_true) + 0.98 * mdot[t-1] + rng.normal(0, 1e-4)
        m[t] = m[t-1] + mdot[t]
    m += mu_true - m.mean() + 0  # center

    z = np.column_stack([
        m + rng.normal(0, 0.01, n),
        np.full(n, np.log(0.2)) + rng.normal(0, 0.03, n),
        np.full(n, np.nan),
        np.full(n, np.nan),
    ])

    kf = AdaptiveKalman(KalmanParams(kappa=kappa_true, theta=0.02))
    est = np.zeros(n)
    for t in range(n):
        r = kf.step(z[t], mu_true)
        est[t] = r["x"][0]

    err_raw = np.std(z[500:, 0] - m[500:])
    err_kf = np.std(est[500:] - m[500:])
    print(f"raw obs err std = {err_raw:.5f}")
    print(f"kalman  err std = {err_kf:.5f}  (개선율 {100*(1-err_kf/err_raw):.1f}%)")
    assert err_kf < err_raw, "필터가 관측 노이즈를 줄이지 못함"
    print("OK: filter reduces observation noise on synthetic OU data")
