# 전략 명세: yoon2

> yoon1(다종목 포트폴리오)의 **상승장 추종 강화** 변형. yoon1은 전체 노출을
> `mean(top-K 점수)`로만 정해 강세장에서도 평균 노출이 ~76~79%에 머물렀다(CAGR이
> EW지수에 뒤지는 주원인 = 평균 ~20% 현금 보유 비용). yoon2는 **확정 강세장에서
> 현금 버퍼를 소진해 노출을 floor까지 끌어올린다**. 하락 방어 로직(regime cap·DEFENSE·
> SPY 시장필터)은 그대로 유지한다. 근거: [[profile-portfolio-v1]](yoon1) §2 "방안1".

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon2 |
| 설명 | yoon1 + 확정 강세장 노출 floor(현금 버퍼 소진, 하락 방어 유지) |
| 작성일 | 2026-06-16 |
| 상태 | validation (핸들러 공유·계약/단위 테스트 통과, 실데이터 검증 대기) |
| config | configs/strategies/yoon2.json |
| 핸들러 | yoon1과 동일(`ProfilePortfolioHandler`) — 차이는 config의 `exposure_floor` 블록뿐 |

## 2. 알고리즘 (yoon1과의 차이만)

yoon1과 동일하게 종목별 profile-sizing 점수 → 상위 K개 → `exposure = mean(top-K 점수)` →
점수 비례 배분. yoon2는 매 리밸런스에 **노출 floor 단계**를 1개 추가한다:

```
raw_exp = mean(top-K 점수)              # yoon1의 전체 노출
if 시장레짐 정상(market_ok)              # 확정 강세장에서만
   and raw_exp >= breadth               # 강세 종목 breadth가 충분할 때만
   and raw_exp < level:                 # 아직 floor 미만이면
       노출을 level 까지 상향(점수 비례 유지)
```

핵심 설계 — **하락 방어를 깨지 않는다**:

- **시장필터 게이트**: SPY가 200MA 아래(약세장)면 floor를 적용하지 않고 yoon1처럼
  노출을 `off_scale`배로 줄인다. floor와 시장 축소는 상호배타.
- **breadth 게이트**: `raw_exp < breadth`(=top-K에 방어/DEFENSE 종목이 많아 평균 노출이
  낮은 상태)면 손대지 않는다. 즉 "이미 광범위하게 강세일 때만" 현금을 더 쓴다.
- **레버리지 없음**: `level ≤ 1.0`. floor는 보유 현금 버퍼를 소진할 뿐 차입하지 않는다.
  (레버리지는 별도 후속 단계로 분리.)

## 3. 파라미터 (`exposure_floor` 블록)

| 키 | 기본값 | 의미 |
| --- | ---: | --- |
| `enabled` | true | floor 활성 여부. false면 yoon1과 완전 동일 |
| `level` | 0.95 | 끌어올릴 목표 노출 상한(≤1.0, 레버리지 없음) |
| `breadth` | 0.60 | 이 raw 노출 이상일 때만 floor 적용(강세 breadth 게이트) |

나머지 파라미터(universe·top_k=20·monthly·trend floor=1.0·시장필터)는 yoon1과 동일.

## 4. 합격선 (결과 확인 전 고정)

yoon1을 기준선으로, 공정 벤치마크(상시 완전투자 EW지수) 대비:

| 지표 | 합격선 |
| --- | --- |
| CAGR | yoon1보다 높을 것(상승장 추종 강화가 목적) |
| MDD | EW지수보다 여전히 뚜렷이 얕을 것(방어 엣지 유지, yoon1 대비 악화는 소폭 허용) |
| Sharpe | EW지수 대비 −0.05 이내(위험조정 우위 유지) |

요지: **CAGR↑를 얻되 MDD·Sharpe로 본 방어 엣지를 크게 훼손하지 않을 것.**

## 5. 합성 드라이런 결과 (동일 시드 6종목, phase=all)

동일 합성 유니버스에서 floor만 켠 직접 비교(`market_filter`는 합성에서 비활성이라 floor가
전 구간 적용됨 → 실데이터보다 공격적인 상한값):

| | avg 노출 | CAGR | MDD | Sharpe | 총수익 |
| --- | ---: | ---: | ---: | ---: | ---: |
| yoon1 | 0.590 | 2.51% | -13.97% | 0.481 | 21.71% |
| **yoon2** | **0.664** | **2.76%** | -14.53% | 0.466 | **24.09%** |

→ 설계대로 **노출↑ → CAGR·총수익↑, MDD 소폭↑**. 합성은 시장필터가 꺼져 floor가
무차별 적용되므로 trade-off가 다소 과장된다(실데이터에선 확정 강세장에만 적용).

## 6. 산출물과 등록

- [x] `scripts/profile_sizing/portfolio.py` `simulate_portfolio`에 `exposure_floor`/
  `exposure_floor_breadth` 추가(기본 0 = no-op → yoon1 불변)
- [x] `ProfilePortfolioHandler`가 config `exposure_floor` 블록을 읽어 전달
- [x] `configs/strategies/yoon2.json` + registry 등록
- [x] `tests/test_strategy_contract.py` 자동 포함 + 전체 테스트 통과(101개)
- [ ] **실데이터 30종목 yoon1 vs yoon2 비교(이 환경은 yfinance 차단 — 네트워크 가능
  환경에서 실행 필요)**: `portfolio_sweep`/`batch`로 CAGR↑·MDD·Sharpe 합격선 확인
- [ ] level·breadth 스윕(validation) → test(holdout) 개봉으로 live 판정
