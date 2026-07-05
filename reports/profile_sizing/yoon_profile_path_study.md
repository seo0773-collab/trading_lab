# yoon_profile 경로 틸트(path_tilt) 정식 검증

- 데이터: BTC/USDT 일봉 2017-08~2026-07 (유효 2,527봉, 워밍업+vel 창 제외)
- 스크립트: `scripts/profile_sizing/yoon_profile_path_study.py` → `yoon_profile_path_study.json`
- 출발점: `state_field_trend_analysis.md` §6 — 궤적 실험에서 "같은 셀이라도 필드
  진입 경로(상승/하강)에 따라 기대수익이 다르다"는 가설. 전 기간 스크리닝 후
  **가설·파라미터를 사전 등록**하고 이 스터디로 검정했다.

## 사전 등록 (스윕 금지)

- vel = mult_pct.diff(**10**) (인과적 필드 속도). 강건성 보고용 {5, 20}은 판정 미사용.
- arrival: up = vel≥0, down = vel<0.
- 검정 family: 셀 {(1,1),(2,1)} × 경로 {down,up} × horizon {5,20,60} = 12개
  (Bonferroni + BH-FDR).
- **PA** M1A1 분리: delta(up−down)>0, 채택 = fwd20/60 t>1.96 & WF 3/3.
- **PB** M2A1 하강 방어: down 평균 fwd20<0, 채택 = t<−1.96 & WF 3/3.
- 백테스트 게이트 가중치(고정): M1A1 up→1.0 / down→0.5, M2A1 down→0.0.
  채택 조건 = 스터디 통과 AND test phase Sharpe·MDD 비열위.

## 판정

| 가설 | 판정 | 근거 |
| --- | --- | --- |
| **PA (M1A1 경로 분리)** | **채택** | delta fwd20 +7.6%p (t=2.32), fwd60 +13.3%p (t=2.10), **WF 3/3 양수**(+20.9/+12.0/+8.1%p). up-arm은 단독으로도 fwd20/60 Bonferroni 유의(t=3.66/4.83) |
| **PB (M2A1 하강 방어)** | **기각** | fwd20 −6.3% (t=−1.69, 임계 미달) & **WF 1/3** — 엣지가 단일 사이클(2022) 의존. 백테스트에서는 all +3.5 수익 개선이 있었으나 사전 등록 규칙대로 제외 |

강건성: vel 창 20도 유의(delta t=2.5~2.9), 창 5는 소멸(t<1) — 신호는 ~2주 이상의
필드 이동에 있고 단기 노이즈에는 없다. WF delta가 사이클이 갈수록 감쇠(+20.9→+8.1%p)
하는 점은 유보 사항.

## 백테스트 게이트 (왕복 20bp, 채택분 PA만)

| phase | base 수익/Sharpe/MDD | tilt_PA 수익/Sharpe/MDD |
| --- | --- | --- |
| all | +1916% / 0.873 / −44.3% | **+2169% / 0.922 / −41.7%** |
| validation | +246% / 1.616 / −20.1% | +203% / 1.496 / −20.1% |
| **test (holdout)** | +15.7% / 0.391 / −21.4% | **+18.5% / 0.439 / −20.2%** |

test phase에서 수익·Sharpe·MDD 모두 개선 → 게이트 통과. validation(2024 강세 구간)
은 소폭 열위 — M1A1 하강 진입 축소가 강세장 일부 재진입을 늦춘 비용. 평균 노출은
0.467→0.457로 거의 동일(수익 개선이 레버리지 증가가 아니라 배치 개선에서 나옴).

## 구현

`scripts/profile_sizing/yoon_profile.py`의 `path_tilt` config 블록 (기본 off →
base 동작 불변, `short` 블록과 같은 opt-in 패턴):

```json
"path_tilt": {
  "enabled": true,
  "vel_window": 10,
  "overrides": [
    {"mult_bin": 1, "angle_bin": 1, "arrival": "up", "weight": 1.0},
    {"mult_bin": 1, "angle_bin": 1, "arrival": "down", "weight": 0.5}
  ]
}
```

## 유보 / 무효화

- vel 창 10·가중치 1.0/0.5는 사전 등록 고정값 — 재튜닝하려면 스터디 재실행.
- WF delta 감쇠 추세(+20.9→+8.1%p): 1년 데이터 누적 후 재실행에서 delta 부호
  반전 또는 holdout 연장 구간에서 base 대비 Sharpe·MDD 동시 열위면 폐기.
- PB(M2A1)는 방향은 맞지만 단일 사이클 의존 — 다음 하락 사이클 데이터가 쌓이면
  재검정 후보.

## 이력

- 2026-07-04: 사전 등록 → 스터디 → PA 채택/PB 기각 → 게이트 통과 (이 문서).
