"""Rolling Ridge 민감도 모델 (finance_plan.txt §7·§22).

각 발표 이벤트에서, 그 시점까지 **타깃이 이미 실현된** 과거 이벤트만으로 Ridge를
적합해(rolling train_quarters) 20일/60일 예상 수익률과 팩터별 계수(=민감도)를 낸다.

누수 차단 두 가지:
1. 인과 학습 풀: 후보 j는 ``ret{H}_time_j <= available_date_i`` 인 과거 이벤트뿐.
   (예측 시점에 아직 실현 안 된 타깃은 학습에 못 쓴다 — §13 5번.)
2. 윈도우 내 표준화: 평균/표준편차를 학습 윈도우에서만 구해 질의점에 적용.
   전체기간 통계를 미리 쓰지 않는다(§13 4번).

외부 의존 없이 정규방정식으로 Ridge를 푼다(sklearn 불요).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import AVAILABLE_DATE
from .config import FinSensitivityConfig
from .fundamentals import feature_columns


def _ridge_beta(x_std: np.ndarray, y_centered: np.ndarray, alpha: float) -> np.ndarray:
    """표준화 X · 중심화 y 에 대한 ridge 계수 (절편은 분리 처리됨)."""
    p = x_std.shape[1]
    gram = x_std.T @ x_std + alpha * np.eye(p)
    return np.linalg.solve(gram, x_std.T @ y_centered)


def _predict_one(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    """윈도우 표준화 후 1개 질의점 예측 + 표준화공간 계수(민감도) 반환.

    실데이터는 팩터별 결측이 흔하므로(개별 종목·과거 구간) 학습행을 버리지 않고
    윈도우 평균으로 대치한다 — 표본 수를 지키면서 인과성을 유지한다(통계는 모두
    학습 윈도우 내에서만 계산). 전부 결측인 컬럼은 계수 0으로 비활성.
    """
    mu = np.nanmean(train_x, axis=0)
    sigma = np.nanstd(train_x, axis=0, ddof=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)

    xs = (np.where(np.isnan(train_x), mu, train_x) - mu) / sigma_safe
    y_mean = train_y.mean()
    yc = train_y - y_mean

    beta = _ridge_beta(xs, yc, alpha)
    q = np.where(np.isnan(query_x), mu, query_x)  # 결측 피처 → 윈도우 평균(=0)
    q_std = (q - mu) / sigma_safe
    pred = float(y_mean + q_std @ beta)
    return pred, beta


def rolling_predict(
    table: pd.DataFrame, cfg: FinSensitivityConfig
) -> dict[str, object]:
    """이벤트 테이블에 pred_ret_20d/60d·민감도 컬럼을 채워 반환.

    반환 dict:
      - "table": 입력에 pred_ret_20d, pred_ret_60d, sens_<factor>(20d 기준),
        insufficient(bool) 가 추가된 사본.
      - "coef20"/"coef60": 이벤트×피처 계수 프레임(민감도 변화 그래프용, §14).
      - "n_predicted": 실제 예측이 산출된 이벤트 수.
    """
    feats = feature_columns(cfg)
    out = table.copy().reset_index(drop=True)
    n = len(out)
    avail = pd.to_datetime(out[AVAILABLE_DATE]).to_numpy()

    horizons = {"pred_ret_20d": ("ret_20d", "ret20_time"),
                "pred_ret_60d": ("ret_60d", "ret60_time")}
    for col in horizons:
        out[col] = np.nan
    for f in feats:
        out[f"sens_{f}"] = np.nan
    out["insufficient"] = True

    coef_rows = {"pred_ret_20d": [], "pred_ret_60d": []}
    n_predicted = 0

    X_all = out[feats].to_numpy(dtype=float)
    for i in range(n):
        predicted_any = False
        for pred_col, (target_col, time_col) in horizons.items():
            y = out[target_col].to_numpy(dtype=float)
            t_real = pd.to_datetime(out[time_col]).to_numpy()
            # 인과 풀: 타깃 실현시점이 현재 사용가능일 이전 + 타깃 유효.
            # 피처 결측은 _predict_one이 윈도우 평균으로 대치(행 보존).
            mask = (t_real <= avail[i]) & ~np.isnan(y)
            mask[i:] = False  # 자기 자신·미래 이벤트 제외
            idx = np.flatnonzero(mask)
            if len(idx) < cfg.min_train_quarters:
                continue
            idx = idx[-cfg.train_quarters:]
            pred, beta = _predict_one(
                X_all[idx], y[idx], X_all[i], cfg.ridge_alpha
            )
            out.at[i, pred_col] = pred
            coef_rows[pred_col].append(
                {"available_date": avail[i], **dict(zip(feats, beta))}
            )
            if pred_col == "pred_ret_20d":
                for f, b in zip(feats, beta):
                    out.at[i, f"sens_{f}"] = b
            predicted_any = True
        if predicted_any:
            out.at[i, "insufficient"] = False
            n_predicted += 1

    return {
        "table": out,
        "coef20": pd.DataFrame(coef_rows["pred_ret_20d"]),
        "coef60": pd.DataFrame(coef_rows["pred_ret_60d"]),
        "n_predicted": n_predicted,
    }
