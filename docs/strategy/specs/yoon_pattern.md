# 전략 명세: yoon_pattern

> 직전 **L틱(기본 300 일봉) 패턴을 미래에 투영**해 "투영이 가리키는 H틱 예상수익"을
> 점수로 만들고, 점수 버킷별 **무누수 워크포워드 기대값**을 long-only 비중으로
> 매핑하는 단일종목 전략.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon_pattern |
| 분류 | single (단일종목) |
| 핸들러 | `src/trading_lab/strategies/yoon_pattern.py` (`YoonPatternHandler`) |
| 피처/파이프라인 | `scripts/profile_sizing/pattern_projection.py` |
| config | `configs/strategies/yoon_pattern.json` |
| 검증 스터디 | `scripts/profile_sizing/yoon_pattern_study.py` → `reports/profile_sizing/yoon_pattern_study.json` |

## 2. 가설과 투영 방식 선정 (2026-07-02 이벤트 스터디)

가설: "이전 300틱의 패턴을 미래에 투영시켜 매수/매도 기대값을 형성한다."
투영 두 방식을 무누수로 구현해 IC로 비교했다 (SPY 1993~2026 · QQQ 1999~2026):

| 방식 | L=300, H=21 (SPY) | 판정 |
| --- | --- | --- |
| **analog** — 유사 패턴 앙상블: 현재 L틱 창과 유사한 과거 창 top-M의 '직후 H틱' 실현수익 중앙값 | IC +0.056 · 비중복 +0.047 · 1/3구간 전부 양수 · 5분위 단조(31→131bps) | **채택** |
| self — 자기 패턴 연장(주기 L 반복 가정) | IC −0.056 · 비중복 −0.081 · 1/3구간 전부 음수 | **기각** (L=150에서만 간헐 양수, 구간 불안정) |

- H=5는 양쪽 다 IC ~0.03 이하로 약함 → **H=21 채택**.
- QQQ는 analog도 약함(비중복 +0.088이나 1/3구간 혼조) → **일반화 주의**, SPY로 시작.
- 무누수 규약: 후보 창 끝 s는 `s+H ≤ t`(직후 경로가 t 이전에 완결)만 허용, 매치 간
  최소 간격 gap=H(동일 에피소드 중복 방지), 패턴은 z-정규화 로그가격(step=5 서브샘플).

## 3. 체인

```
score(t) = median( logp(s+H) − logp(s) ), s ∈ top-M 유사 창 (무누수)
  → walk_forward_expectancy: 점수 버킷(3개: <−1%, ±1%, >+1%)별 비용차감 H틱
     선행수익 평균, 확장창·min_samples 200·21봉 재적합 (rolling_expectancy 재사용)
  → expectancy_weights: 기대값>10bps만 비중=기대값/100bps, [0,1] 클립, 점진 리밸런스
```

버킷 3개·확장창·min_samples 200은 rolling_expectancy 표본 설계 스윕(2026-07-02)의
교훈을 그대로 적용한 것이다 (가는 버킷 + 짧은 창 = 표본 미달로 신호 사망).

## 4. 주의 / 무효화 조건

- **단일종목 일봉 타이밍 계열의 레포 전례**(구 yoon2 best-day-miss, rolling_expectancy
  B&H 미달)가 불리한 사전확률. IC +0.056은 약한 신호다 — 승격 기준은 B&H 위험조정
  상회가 아니라 "양의 기대값 + 유의한 낙폭 방어"로 낮춰 시작한다.
- 유사도 계산은 z-정규화라 변동성 레짐 차이를 무시한다(형태만 매칭). 개선 여지.
- **무효화**: holdout(test 구간)에서 전략 수익이 음수이거나, IC가 0 이하로 붕괴하면 폐기.
- `live_eligible=False`. step·M 추가 스윕은 후속(스터디 JSON에 L=150 결과 보존).

## 5. 파라미터 확정 이력 (2026-07-02)

- **L=450 채택** (멀티심볼 결판, `yoon_pattern_multisymbol.py`): SPY 단독 스윕은
  val↔test 순위 역전(약신호 민감도)이라 4심볼(SPY·QQQ·IWM·EFA)로 재판정 —
  L=450만 val/test 전 칸 양수(val Sharpe 중앙 0.643·최소 0.302, test 4/4 양수).
  L=300은 EFA val 음수, L=600은 IWM test 음수. 원가설 300틱은 JSON에 보존.
- **risk_overlay(변동성 타게팅 15%) 기본 ON**: 확장창 기대값 테이블의 급락 전환
  지연을 외생 변동성 신호로 보완(전봉 21봉 실현변동성, 무누수). 실측: test MDD
  -23.3→-16.4%·Sharpe 개선, CAGR -1.8%p 비용(`yoon_pattern_sweep.md`).
- **확정 기본값 SPY 성과**: validation 4.98%/0.694/-11.1%, test(holdout)
  7.67%/0.763/-14.6% — 초기 등록값(L300·오버레이 없음: test 9.49%/0.608/-30.7%)
  대비 위험조정·MDD 대폭 개선, CAGR 소폭 양보. 일반화: SPY·QQQ 강함,
  IWM·EFA 약하지만 양수.
- **숏 확장 기각(2026-07-02 실측)**: 전 구간 기대값 테이블에 음수 버킷이 없고(최약
  버킷도 +30bps), 워크포워드 조건부 음수 기대값으로 숏을 켜면(봉의 4.1%) 그 봉들의
  이후 21틱 실현수익이 **+1.90%**(반등 직전에 숏) — all 구간 CAGR/Sharpe/MDD 전부
  악화(4.60→4.11%/0.409→0.370/-30.7→-32.6%), 대여비용 미반영 상태에서도 열세.
  long-only 유지. (구 yoon2 교훈 "강세장에서 숏이 전략을 죽임"과 일치)
