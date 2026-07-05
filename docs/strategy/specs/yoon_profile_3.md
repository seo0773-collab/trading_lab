# 전략 명세: yoon_profile_3

> yoon_profile(3×3 상태 가중치)에 **path_tilt**(경로 조건부 비중)를 활성한 파생.
> 같은 셀이라도 필드 진입 경로(필드 속도 부호)에 따라 목표 비중을 분리한다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon_profile_3 |
| 분류 | single (단일종목, 기본 BTC/USDT 일봉) |
| 핸들러 | `src/trading_lab/strategies/yoon_profile.py` (yoon_profile과 공유) |
| 파이프라인 | `scripts/profile_sizing/yoon_profile.py` (`path_tilt` 블록) |
| config | `configs/strategies/yoon_profile_3.json` |
| 검증 스터디 | `scripts/profile_sizing/yoon_profile_path_study.py` → `reports/profile_sizing/yoon_profile_path_study.{json,md}` |
| 출발점 | `reports/profile_sizing/state_field_trend_analysis.md` §6 (2D 궤적 실험) |

## 2. base와의 차이 (이것만)

```
vel(t) = mult_pct(t) − mult_pct(t−10)        # 인과적 필드 속도
M1A1 셀 목표 비중: 0.8 고정 →  vel ≥ 0 (상승 진입) 1.0
                              vel < 0 (하강 진입) 0.5
```

나머지(지표·bin·state_weights·사이징·비용)는 yoon_profile과 동일.
vel 창 10·가중치 1.0/0.5는 **사전 등록 고정값** — 스윕 산출물이 아니며,
재튜닝하려면 path 스터디 재실행이 필수.

## 3. 근거 (yoon_profile_path_study, 2026-07-04)

- 궤적 실험: 큰 하락 leg가 M1A1을 경유(하락 leg 점유 20%를 w=0.8로 통과),
  큰 상승 leg 초입도 M1A1 통과 — 같은 셀에 상반된 두 모집단 혼재.
- PA 검정(사전 등록): delta(up−down) fwd20 +7.6%p (t=2.32) / fwd60 +13.3%p
  (t=2.10), walk-forward 3/3 양수. up-arm 단독 fwd20/60 Bonferroni 유의.
- PB(M2A1 하강→0)는 WF 1/3로 **기각** — 이 파생에 미포함.
- 강건성: vel 창 20 유지(t=2.5~2.9), 창 5 소멸 — 신호는 2주+ 필드 이동.

## 4. 백테스트 (BTC/USDT 일봉, 왕복 20bp)

| phase | base (yoon_profile) | yoon_profile_3 |
| --- | --- | --- |
| all | +1916% / Sharpe 0.873 / MDD −44.3% | **+2169% / 0.922 / −41.7%** |
| validation | +246% / 1.616 / −20.1% | +203% / 1.496 / −20.1% |
| test (holdout) | +15.7% / 0.391 / −21.4% | **+18.5% / 0.439 / −20.2%** |

평균 노출 0.467→0.457 (거의 동일) — 개선은 노출 확대가 아닌 배치 개선.
validation(2024 강세)은 소폭 열위: 하강 진입 축소가 재진입을 늦춘 비용.

## 5. 주의 / 무효화 조건

- base yoon_profile의 선택 편향 주의사항이 그대로 상속된다(완전 OOS 아님).
- WF delta 감쇠 추세(+20.9→+12.0→+8.1%p): **1년 데이터 누적 후 path 스터디
  재실행에서 delta 부호 반전, 또는 holdout 연장 구간에서 base 대비 Sharpe·MDD
  동시 열위면 폐기**.
- **멀티에셋 검증(2026-07-04, `yoon_profile_3_multiasset_eval.md`)**: path_tilt는
  BTC에서만 +기여 — SOL은 base(tilt off)가 우위(CAGR +32.3%/Sharpe 0.811 vs B&H
  +8.2%/0.513), ETH 판정보류, ETF(SPY/QQQ/GLD) 기각. 단 **타겟 자산을 크립토로
  한정하지 않는다** — 기각된 것은 "BTC 튜닝 가중치의 무수정 이식"이지 구조가
  아니며, 타 자산 적용은 해당 자산 데이터로 state_weights·path 스터디를
  재추정하는 것이 전제다(현 가중치 그대로면 base 사용).
- `live_eligible=False`.

## 6. 이력

- 2026-07-04: path 스터디(PA 채택/PB 기각) → 게이트 통과 → 등록 (이 문서).
