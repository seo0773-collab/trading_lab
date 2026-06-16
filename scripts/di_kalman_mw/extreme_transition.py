"""다음 극점(P5) 값 변위 조건부 확률구조 (연구 모듈; plan 16 standalone).

LV0 = reversal 임계값으로 확정된 단일 레벨 극점(``extremes.extract_extremes``).
한 DI 라인의 극점열에서 M/W 패턴(P1..P4)을 분류하고, 그 다음 교대 극점 P5의
값 변위 ``dv = P5.value - P4.value`` 를 "패턴 평균 레그 진폭"으로 정규화한
분포를 (패턴 라벨 + 기하 특징) 버킷별로 추정한다. 통계는 train 인스턴스로만
적합하고 동결하며, validation/test는 절대 재적합하지 않는다(stats.py와 동일 규율).

인과성(plan 16): 각 인스턴스는 P4의 ``confirmation_idx``(결정 시점)로 split을
가르고, 그 시점까지 확정된 정보(P1..P4)만으로 버킷팅한다. P5는 이후에 확정되며
분포 적합과 평가에만 쓰인다 — 버킷팅 특징에는 절대 들어가지 않는다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .dmi import dmi, kalman_di
from .events import classify_pattern, mw_shape
from .extremes import Extreme, extract_extremes
from .splits import split_labels

EPS = 1e-9

# 버킷팅에 쓰는 기하 특징(스케일 무관). P5는 포함하지 않는다(인과성).
BUCKET_FEATURES = ("leg3_ratio", "p3_vs_p1_norm")

DEFAULT_MIN_BUCKET = 30  # 정밀 버킷 최소 표본 (stats.StatsConfig와 동일 철학)
DEFAULT_MIN_GLOBAL = 10


@dataclass(frozen=True)
class TransitionInstance:
    """P1..P4 패턴 한 건과 (있다면) 그 다음 극점 P5의 관측 결과."""

    line: str  # "plus" | "minus"
    pattern: str  # "W" | "M"
    shape: str  # "diverging" | "parallel" | "converging" (plan 17)
    width_ratio: float  # right/left channel-width ratio
    p4_idx: int
    p4_conf_idx: int  # 결정/스플릿 기준 인덱스
    mean_leg: float  # P1..P4 평균 레그 진폭 (정규화 기준)
    features: dict  # 기하 특징 (P1..P4만 사용)
    has_p5: bool
    dv: float  # P5.value - P4.value (raw signed; P5 없으면 nan)
    dv_norm: float  # dv / mean_leg (메인 타깃)
    p5_vs_p3_norm: float  # (P5.value - P3.value) / mean_leg
    continuation: bool  # W: P5>P3(higher-low) / M: P5<P3(lower-high)
    p5_conf_idx: int  # P5 확정 인덱스 (없으면 -1)
    # 파형 시각화용 극점 좌표 (P1..P4, 그리고 있으면 P5).
    window_idx: tuple = ()  # P1..P4 극점 idx
    window_val: tuple = ()  # P1..P4 극점 값
    p5_idx: int = -1
    p5_value: float = float("nan")


def _geometry(window: tuple[Extreme, ...]) -> tuple[float, dict]:
    """P1..P4 윈도우에서 평균 레그 진폭과 기하 특징을 계산한다 (P5 미사용)."""
    v = [e.value for e in window]
    i = [e.idx for e in window]
    leg1 = abs(v[1] - v[0])
    leg2 = abs(v[2] - v[1])
    leg3 = abs(v[3] - v[2])
    mean_leg = (leg1 + leg2 + leg3) / 3.0
    features = {
        "leg1": leg1,
        "leg2": leg2,
        "leg3": leg3,
        "leg3_ratio": leg3 / max(leg2, EPS),  # 마지막 레그 확장/수축
        "retr_ratio": leg2 / max(leg1, EPS),  # 중간 되돌림 비
        # higher-low(W)/lower-high(M) 강도: W면 +, M면 - 경향
        "p3_vs_p1_norm": (v[2] - v[0]) / max(mean_leg, EPS),
        "span_bars": float(i[3] - i[0]),
    }
    return mean_leg, features


def enumerate_instances(
    extremes: list[Extreme], line: str, strict: bool = False,
    parallel_band: float = 0.20,
) -> list[TransitionInstance]:
    """극점열을 훑어 분류 가능한 P1..P4마다 한 인스턴스를 만든다.

    P5(extremes[j+1])가 아직 없으면 ``has_p5=False``로 남기고(라이브 예측용),
    있으면 정규화 변위/continuation을 기록한다. 각 인스턴스에는 채널 폭 비로
    판정한 shape(확산/수렴/유지) 라벨도 붙는다.
    """
    out: list[TransitionInstance] = []
    n = len(extremes)
    for j in range(3, n):
        window = tuple(extremes[j - 3:j + 1])  # (P1, P2, P3, P4)
        pattern = classify_pattern(window, strict)
        if pattern is None:
            continue
        mean_leg, features = _geometry(window)
        if mean_leg <= EPS:
            continue
        shape, width_ratio = mw_shape(window, parallel_band)
        p3, p4 = window[2], window[3]
        has_p5 = (j + 1) < n
        if has_p5:
            p5 = extremes[j + 1]
            dv = p5.value - p4.value
            dv_norm = dv / mean_leg
            p5_vs_p3_norm = (p5.value - p3.value) / mean_leg
            continuation = (
                p5.value > p3.value if pattern == "W" else p5.value < p3.value
            )
            p5_conf_idx = p5.confirmation_idx
            p5_idx = p5.idx
            p5_value = p5.value
        else:
            dv = dv_norm = p5_vs_p3_norm = float("nan")
            continuation = False
            p5_conf_idx = p5_idx = -1
            p5_value = float("nan")
        out.append(TransitionInstance(
            line=line,
            pattern=pattern,
            shape=shape,
            width_ratio=width_ratio,
            p4_idx=p4.idx,
            p4_conf_idx=p4.confirmation_idx,
            mean_leg=mean_leg,
            features=features,
            has_p5=has_p5,
            dv=dv,
            dv_norm=dv_norm,
            p5_vs_p3_norm=p5_vs_p3_norm,
            continuation=continuation,
            p5_conf_idx=p5_conf_idx,
            window_idx=tuple(e.idx for e in window),
            window_val=tuple(e.value for e in window),
            p5_idx=p5_idx,
            p5_value=p5_value,
        ))
    return out


def _tercile_edges(values: np.ndarray, n_bins: int = 3) -> list[float]:
    if values.size == 0:
        return []
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return [float(q) for q in np.quantile(values, qs)]


def _bin_index(value: float, edges: list[float]) -> int:
    idx = 0
    for edge in edges:
        if value > edge:
            idx += 1
        else:
            break
    return idx


def _bucket_key(
    pattern: str, shape: str, features: dict, edges: dict[str, list[float]]
) -> str:
    parts = [f"{pattern}:{shape}"]
    for k, feature in enumerate(BUCKET_FEATURES):
        b = _bin_index(features[feature], edges.get(feature, []))
        parts.append(f"f{k}={b}")
    return "|".join(parts)


def _summarize(rows: list[tuple[float, bool]]) -> dict | None:
    """(dv_norm, continuation) 목록 → 경험적 분포 요약."""
    arr = np.array([r[0] for r in rows], dtype=float)
    cont = np.array([r[1] for r in rows], dtype=bool)
    finite = np.isfinite(arr)
    arr = arr[finite]
    cont = cont[finite]
    if arr.size == 0:
        return None
    q = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "q10": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "p_continuation": float(cont.mean()) if cont.size else float("nan"),
    }


def build_transition_stats(
    train_instances: list[TransitionInstance],
    *,
    min_bucket: int = DEFAULT_MIN_BUCKET,
    min_global: int = DEFAULT_MIN_GLOBAL,
) -> dict:
    """train 인스턴스로 조건부 분포를 적합한다(동결 산출물).

    반환 구조: ``buckets``(정밀: 패턴·shape·기하), ``by_pattern_shape``(폴백 1단),
    ``by_pattern``(폴백 2단), ``global``(폴백 3단), ``feature_edges``(새 인스턴스
    버킷팅에 필요한 train 경계).
    """
    usable = [
        x for x in train_instances if x.has_p5 and np.isfinite(x.dv_norm)
    ]
    edges = {
        feature: _tercile_edges(
            np.array([x.features[feature] for x in usable], dtype=float)
        )
        for feature in BUCKET_FEATURES
    }

    bucket_rows: dict[str, list] = defaultdict(list)
    pattern_shape_rows: dict[str, list] = defaultdict(list)
    pattern_rows: dict[str, list] = defaultdict(list)
    global_rows: list = []
    for x in usable:
        row = (x.dv_norm, x.continuation)
        bucket_rows[_bucket_key(x.pattern, x.shape, x.features, edges)].append(row)
        pattern_shape_rows[f"{x.pattern}:{x.shape}"].append(row)
        pattern_rows[x.pattern].append(row)
        global_rows.append(row)

    return {
        "buckets": {k: _summarize(v) for k, v in bucket_rows.items()},
        "by_pattern_shape": {
            k: _summarize(v) for k, v in pattern_shape_rows.items()
        },
        "by_pattern": {k: _summarize(v) for k, v in pattern_rows.items()},
        "global": _summarize(global_rows),
        "feature_edges": edges,
        "bucket_features": list(BUCKET_FEATURES),
        "n_train": len(usable),
        "min_bucket": int(min_bucket),
        "min_global": int(min_global),
        "target": "dv_norm = (P5.value - P4.value) / mean_leg(P1..P4)",
    }


def completed_instances_for_split(
    instances: list[TransitionInstance],
    labels: np.ndarray,
    split: str,
) -> list[TransitionInstance]:
    """Return P1..P5 outcomes fully confirmed inside one chronological split."""
    n = len(labels)
    return [
        x
        for x in instances
        if x.has_p5
        and np.isfinite(x.dv_norm)
        and 0 <= x.p4_conf_idx < n
        and 0 <= x.p5_conf_idx < n
        and labels[x.p4_conf_idx] == split
        and labels[x.p5_conf_idx] == split
    ]


def lookup_transition(
    stats: dict, pattern: str, features: dict, shape: str = ""
) -> dict | None:
    """폴백 체인: 정밀 버킷 → 패턴·shape → 패턴 → 전역.

    shape를 비워두면(``""``) 패턴·shape 단계를 건너뛰고 기존 패턴→전역 체인을
    그대로 사용한다(하위 호환).
    """
    if shape:
        key = _bucket_key(pattern, shape, features, stats["feature_edges"])
        b = stats["buckets"].get(key)
        if b and b["n"] >= stats["min_bucket"]:
            return {**b, "bucket": key}
        ps = stats.get("by_pattern_shape", {}).get(f"{pattern}:{shape}")
        if ps and ps["n"] >= stats["min_bucket"]:
            return {**ps, "bucket": f"{pattern}:{shape}"}
    p = stats["by_pattern"].get(pattern)
    if p and p["n"] >= stats["min_bucket"]:
        return {**p, "bucket": pattern}
    g = stats.get("global")
    if g and g["n"] >= stats["min_global"]:
        return {**g, "bucket": "global"}
    return None


def evaluate_transition(
    stats: dict, eval_instances: list[TransitionInstance]
) -> dict:
    """조건부 분포가 무조건부(전역)보다 P5 변위를 잘 맞추는지 평가한다.

    - ``conditional_mae`` / ``global_mae``: |실제 dv_norm − 예측 중앙값| 평균
    - ``mae_reduction_pct``: 조건부가 전역 대비 줄인 오차 비율(클수록 좋음)
    - ``coverage_10_90``: 실제가 버킷 [q10,q90]에 든 비율(보정 목표 ≈ 0.80)
    - ``continuation_brier``: P(continuation) 예측의 Brier 스코어(작을수록 좋음)
    """
    g = stats.get("global")
    cond_err: list[float] = []
    glob_err: list[float] = []
    in_band = 0
    brier: list[float] = []
    realized_cont: list[int] = []
    n = 0
    for x in eval_instances:
        if not (x.has_p5 and np.isfinite(x.dv_norm)):
            continue
        st = lookup_transition(stats, x.pattern, x.features, x.shape)
        if st is None:
            continue
        n += 1
        cond_err.append(abs(x.dv_norm - st["median"]))
        if g is not None:
            glob_err.append(abs(x.dv_norm - g["median"]))
        if st["q10"] <= x.dv_norm <= st["q90"]:
            in_band += 1
        p_cont = st.get("p_continuation")
        if p_cont is not None and np.isfinite(p_cont):
            brier.append((p_cont - float(x.continuation)) ** 2)
        realized_cont.append(int(x.continuation))

    cond_mae = float(np.mean(cond_err)) if cond_err else float("nan")
    glob_mae = float(np.mean(glob_err)) if glob_err else float("nan")
    reduction = (
        float((glob_mae - cond_mae) / glob_mae * 100.0)
        if glob_err and glob_mae > 0
        else float("nan")
    )
    return {
        "n_evaluated": n,
        "conditional_mae": cond_mae,
        "global_mae": glob_mae,
        "mae_reduction_pct": reduction,
        "coverage_10_90": float(in_band / n) if n else float("nan"),
        "continuation_brier": float(np.mean(brier)) if brier else float("nan"),
        "realized_continuation_rate": (
            float(np.mean(realized_cont)) if realized_cont else float("nan")
        ),
    }


@dataclass(frozen=True)
class PatternDataset:
    """OHLCV → DI Kalman 극점 → M/W 인스턴스 한 묶음 (리포트/대시보드 공용)."""

    instances: list  # list[TransitionInstance] (+DI, -DI 합본)
    labels: np.ndarray  # 봉별 split 라벨
    plus_kalman: "pd.Series"  # noqa: F821
    minus_kalman: "pd.Series"  # noqa: F821
    plus_extremes: list  # list[Extreme]
    minus_extremes: list


def build_pattern_dataset(df, cfg, strict: bool = False) -> PatternDataset:
    """전략 config(StrategyConfig)로 +DI/-DI Kalman 극점을 뽑고 M/W 인스턴스를 만든다.

    ``transition_report``와 대시보드 연구 탭이 동일 결과를 쓰도록 하는 단일 진입점.
    """
    ind = cfg.indicators
    plus_di, minus_di = dmi(df, ind.di_len)
    plus_kalman = kalman_di(plus_di, ind.kalman_q, ind.kalman_r)
    minus_kalman = kalman_di(minus_di, ind.kalman_q, ind.kalman_r)
    plus_ext = extract_extremes(plus_kalman, cfg.extremes)
    minus_ext = extract_extremes(minus_kalman, cfg.extremes)
    labels = split_labels(len(df), cfg.split)
    band = cfg.patterns.parallel_band
    instances = (
        enumerate_instances(plus_ext, "plus", strict, band)
        + enumerate_instances(minus_ext, "minus", strict, band)
    )
    return PatternDataset(
        instances=instances,
        labels=labels,
        plus_kalman=plus_kalman,
        minus_kalman=minus_kalman,
        plus_extremes=plus_ext,
        minus_extremes=minus_ext,
    )
