# yoon2b 연구 게이트 확장: EWY/SPY 멀티사이클 틸트 (Phase 0-b)

단일 룩백 momentum(yoon2b_core_tilt_compare)이 게이트 실패한 뒤, 신호를 여러 주기로
분해/결합하면 위험조정 우위가 생기는지 검증. 코어 슬리브·동적 결합·phase 슬라이스는
기존 스크립트 재사용. 주기 = 3M/6M/12M(63/126/252봉).

- **T3 추세 블렌드**: 다주기 모멘텀 z-score 평균 → 추세추종
- **T4 주기 평균회귀**: 다주기 detrend 오실레이터 z-score 평균 → 역추세(과열 시 KR 축소)

## 판정: ❌ 게이트 실패 — 여전히 추가하지 않음

| 변형 | train Sharpe / MDD | validation Sharpe / MDD | all Sharpe | 결과 |
| --- | --- | --- | --- | --- |
| **base 50/50** | **0.788** / -12.09% | **1.083** / -14.46% | **1.143** | 기준 |
| T3_mc_trend (k=0.1) | 0.795 / -12.10% | 1.143 / -14.77% | 1.152 | train 미통과(노이즈) |
| T4_mc_revert (k=0.3) | **0.831** / -10.67% | **0.931** / -13.88% | 1.111 | **train 통과 → validation 붕괴(과최적화)** |

### T4 = 과최적화 함정 (게이트가 잡아낸 사례)
- **train에서는 합격**: Sharpe 0.788→0.831, MDD -12.09%→-10.67%(더 얕음). k 전 구간 통과.
- **그러나 validation에서 무너짐**: Sharpe 1.083→**0.931**, CAGR 11.44%→9.96% (둘 다 악화).
- 전구간(all) Sharpe도 base보다 낮음(1.111 < 1.143).
- → train만(또는 전구간만) 보고 채택했으면 합격처럼 보였을 신호가, OOS에서 깨진다.
  multi-cycle이 추가한 자유도가 train 노이즈를 적합한 것. **검증 분리가 이걸 잡았다.**

### T3 = 노이즈 수준
- train Sharpe +0.007, MDD 0.01pp 더 깊어 가드 탈락. 전구간 Sharpe +0.009 — 새 파라미터를 정당화할 크기 아님.

## 결론
- 단일 룩백·다주기 추세·다주기 역추세 **모두** 고정 50/50을 위험조정 기준으로 robust하게 이기지 못함.
- basis 제거(T2)·다주기 분해(T3/T4)로 신호를 정교화해도 우위가 안 나옴 → **한·미 상대강도에 안정적으로 추출 가능한 타이밍 알파가 없다**는 결론을 강화.
- T4 사례는 멀티사이클처럼 자유도 큰 신호일수록 in-sample 착시가 커진다는 점을 보여줌. validation 게이트 유지 정당.

## 파일
- 스크립트: `scripts/profile_sizing/yoon2b_multicycle_compare.py`
- CSV: `reports/profile_sizing/yoon2b_multicycle_compare.csv`
- 선행: `reports/profile_sizing/yoon2b_core_tilt_compare.md`
