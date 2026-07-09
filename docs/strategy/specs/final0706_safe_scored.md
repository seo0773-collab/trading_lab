# 전략 명세: final0706_safe_scored

> `final0706`과 구조가 완전히 동일한 파생. 유일한 차이는 안전자산 슬리브
> (TLT/IAU, 148070.KS/132030.KS)의 배분 방식이다 — 원본은 추세 게이트
> (종가>MA100) ON/OFF만 판정해 ON인 자산끼리 균등 배분하지만, 이 파생은
> 안전자산도 메인 유니버스와 같은 profile-sizing 엔진으로 스코어링해 ON
> 자산 간 배분을 점수 비례로 바꾼다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | final0706_safe_scored |
| 코어 구조 | [[final0706]]과 동일(US yoon1j_lp + KR yoon1j_kr 50/50, 63일 통화틸트, 신용스프레드 위험게이트) |
| 차이점 | 두 슬리브의 `safe_sleeve.profile_scored: true` 추가 |
| 핸들러 | `Yoon1kHandler`(`src/trading_lab/strategies/yoon1k.py`) — 재사용, 신규 핸들러 없음 |
| config | `configs/strategies/final0706_safe_scored.json` |

## 2. 배경

기존 안전자산 슬리브(yoon1j 계열, `portfolio.py` `simulate_portfolio`)는 다음처럼
동작한다: 각 안전자산이 자기 종가 > MA(기본 100일)이면 ON, 아니면 OFF로 이진
판정하고, ON인 자산들에게 완충분(`min(safe_max, spare, spare*safe_ratio)`)을
**균등 배분**한다. 이 로직은 메인 주식 유니버스가 쓰는 profile-sizing 스코어링
(cycle_multiple의 자기 분포 내 percentile → buckets → regime_cap = 컨트래리언
점수)과 완전히 분리돼 있었다 — 안전자산은 "얼마나 유리한 위치에 있는지"가 아니라
"추세가 살아있는지"만 봤다.

"안전자산도 프로파일 점수화하면 어떨까"라는 질문에서 출발해, 안전자산에도 같은
엔진(`run_pipeline`)을 돌려 `final_target_weight`를 구하고, ON인 자산들 사이의
배분을 그 점수 비례로 바꾸는 opt-in 옵션(`safe_sleeve.profile_scored`)을
구현했다. 추세 게이트(ON/OFF) 자체는 그대로 둬서 기존 방어 효과(2022류 안전자산
동반 하락 시 완전 회피)는 보존했다 — 바뀌는 것은 "ON일 때 TLT와 GLD/IAU 중
어느 쪽에 더 배정할지"뿐이다.

## 3. 구현

- `scripts/profile_sizing/portfolio.py`
  - `simulate_portfolio(..., safe_scores: pd.DataFrame | None = None)` 파라미터 추가.
  - `safe_score_sig = safe_scores.reindex(idx).shift(1).fillna(0.0)`(전봉 기준, 무누수).
  - 리밸런스 시 ON 자산 집합(`safe_on`, 추세 게이트 불변)은 그대로 두고, 배분을
    `safe_w_map = {s: safe_budget * (score_s / sum(scores)) for s in safe_on}`로
    계산(점수 합이 0이면 균등 배분으로 폴백). `safe_scores=None`(기본)이면
    기존 균등 배분과 완전히 동일 — 다른 전략에 회귀 없음.
- `src/trading_lab/strategies/profile_portfolio.py`
  - `_load_panel` 헬퍼 추가(안전자산 전체 OHLCV 로드, `run_pipeline` 입력용).
  - `safe_sleeve.profile_scored`가 true면 각 안전자산에 `run_pipeline(raw, cfg)`를
    돌려 `final_target_weight`를 점수로 추출, `simulate_portfolio(safe_scores=...)`에 전달.
  - 안전자산 스코어링은 **메인 유니버스와 같은 `cfg`(같은 buckets/regime_cap)를
    공유**한다 — 별도 buckets 슬롯은 없다(알려진 한계, 5절 참조).

## 4. 실측 결과

`scripts/profile_sizing/final0706_safe_sleeve_profile_study.py`,
`reports/profile_sizing/final0706_safe_sleeve_profile_study.md`
(공식 `Yoon1kHandler`/`ProfilePortfolioHandler`로 계산, 재구현 없음).

### 전체 파이프라인 (2007-01-29~2026-07-08, 원화 기준)

| candidate | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| final0706(원본, 균등 배분) | 13.66% | -16.40% | 1.223 |
| **final0706_safe_scored(점수 비례)** | **13.74%** | **-16.15%** | **1.227** |

`test`(2022-12~2026-07) 구간: 33.63%→33.95% / Sharpe 1.953→1.968.
`recent_3y`(2023-07~2026-07) 구간: 36.91%→37.20% / Sharpe 2.011→2.023.
세 구간 모두 같은 방향(개선).

### 슬리브 단독 성과

| sleeve | variant | CAGR | MDD | Sharpe | 평균 안전자산 비중 |
| --- | --- | ---: | ---: | ---: | ---: |
| US(yoon1j_lp) | 원본 | 8.33% | -21.02% | 0.749 | 8.08% |
| US(yoon1j_lp) | profile_scored | 8.41% | -21.02% | 0.755 | 8.09% |
| KR(yoon1j_kr) | 원본 | 9.88% | -37.55% | 0.706 | 13.54% |
| KR(yoon1j_kr) | profile_scored | 9.93% | -37.58% | 0.708 | 13.54% |

### 페어드 블록 부트스트랩 (원본 대비, 8000회, block=20)

| 비교 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| safe_scored vs 원본 | 82.9% | 88.5% | 60.1% |

**해석**: 방향은 CAGR·Sharpe·MDD 전부 개선이고 부트스트랩도 82~89%로 꽤 신뢰할
만하지만, 안전자산 평균 비중 자체가 작아(US 8%/KR 13.5%, 대부분 자금은 공격
슬리브에 있음) 절대 개선폭은 작다(CAGR +0.08%p, Sharpe +0.004). "손해는 아니고
방향은 맞다"는 확인이지 게임체인저는 아니다.

## 5. 알려진 한계

- final0706 자체가 아직 페이퍼 전진검증 전(`live_eligible=false`)이므로 이 파생도
  같은 게이트를 상속한다.
- 안전자산 스코어링이 메인 유니버스와 buckets를 공유하는 구조라, 안전자산 전용
  buckets/regime_cap 튜닝은 아직 해보지 않았다(현재는 주식용으로 캘리브레이션된
  곡선을 채권/금에 그대로 재사용). 이 부분을 분리하면 추가 개선 여지가 있을 수
  있으나 코드 확장(안전자산 전용 profile config 슬롯)이 필요해 1차 등록에는
  포함하지 않았다.
- `mid_peak buckets`(프로파일 40~60% 구간에 최댓값을 두는 가설, [[final0706-strategy]]
  하위 실험)와의 결합은 `final0706_mid_peak_plus_safe_scored_study.py`에서
  기각됐다(mid_peak의 손실 효과가 압도, P(Sharpe 우위) 3.2%) — 이 config에는
  mid_peak을 넣지 않았다.
- 개선폭이 작아(연 CAGR +0.08%p) 표본 기간 전체가 아닌 부분 구간에서는 원본이
  더 나을 수도 있다(부트스트랩 P(MDD 더 얕음)이 60.1%로 다른 지표보다 약함).
  롤링 워크포워드 검증은 아직 수행하지 않았다.

## 6. 승격 정책

`final0706`이 승격(페이퍼 3개월 + KIS 실수수료 확정)되기 전까지는
`live_eligible=false` 유지. 개선폭이 작으므로 승격 검토 시 final0706 원본과
함께 비교해 이 파생을 채택할지(또는 원본을 이 구성으로 교체할지) 판단한다.
