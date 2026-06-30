# yoon2b 연구 게이트: EWY/SPY 동적 원/달러 틸트 (Phase 0)

전략/registry/config 미변경 임시 계산. yoon2 코어의 KRW-base 슬리브 NAV
(미국 yoon1j_lp / 한국 yoon1j_kr)를 재사용해, 월간 리밸런싱 목표비중을
고정 50/50 대신 상대모멘텀으로 틸트했을 때 성과가 개선되는지 검증했다.

- 기간: 2007-01-29 ~ 2026-06-26 (4080봉)
- 틸트 규칙: `w_kr = clip(0.5 + k·tanh(z), [0.30, 0.70])`, z=ratio 모멘텀의 252봉 z-score, 1일 시프트(PIT)
- 신호: **T1**=EWY/SPY(둘 다 USD, 단 EWY는 원화효과 내포) · **T2**=자기 슬리브 NAV 비율(basis 제거)
- 선택 규칙: train(0.6)에서 lookback/k 탐색 → **Sharpe↑ 그리고 MDD≤base** 동시 충족 시에만 채택, validation으로 재확인

## 판정: ❌ 게이트 실패 — 전략 추가하지 않음

train 구간에서 **고정 50/50을 위험조정 기준으로 이기는 (lookback,k) 조합이 없다.**

| 변형 | phase | CAGR | MDD | Sharpe |
| --- | --- | ---: | ---: | ---: |
| **base 50/50** | train | 8.95% | -12.09% | **0.788** |
| | validation | 11.44% | -14.46% | 1.083 |
| | test | 33.52% | -10.26% | 2.084 |
| | all | 13.99% | -14.46% | **1.143** |
| T1 (EWY/SPY, L=126 k=0.10) | train | 8.71% | -11.89% | 0.760 |
| | all | 14.46% | -14.71% | 1.132 |
| T2 (sleeve NAV, L=126 k=0.10) | train | 8.95% | -11.56% | 0.783 |
| | all | 14.21% | -15.09% | 1.121 |

### 근거
- 짧은 룩백(126): Sharpe가 base보다 **낮아진다**(0.760~0.783 < 0.788). MDD만 약간 얕아지는 무알파 방어.
- 긴 룩백(252): CAGR은 오르지만 MDD가 깊어지고(-14~-18%) Sharpe는 ~0.80으로 **거의 평탄**(+0.01, 노이즈). MDD 가드에 모두 탈락.
- 즉 CAGR 상승은 전부 **낙폭을 더 받은 대가**일 뿐 위험효율 개선이 아니다.
- 전구간(all) Sharpe도 base(1.143)를 못 넘는다(T1 1.132 / T2 1.121).
- basis를 제거한 T2조차 base를 못 이김 → 신호 프록시 품질 문제가 아니라 **한·미 상대모멘텀에 들 만한 타이밍 알파 자체가 없음**(상관 중간 → 예상과 일치).

## 결론
- 사전 합의 기준(train·validation 동시 Sharpe↑ & MDD 비악화)에 미달 → **yoon2b 추가 보류.**
- 고정 50/50이 단순할 뿐 아니라 위험조정 기준으로 더 낫다. 자유변수(lookback·k)를 늘릴 정당성 없음.
- 직전 yoon2a 점검에서 지적한 SSO 레이어의 Sharpe 열위 문제는 이 코어 틸트로는 해결되지 않는다(직교 레이어).

## 파일
- 스크립트: `scripts/profile_sizing/yoon2b_core_tilt_compare.py`
- 결과 CSV: `reports/profile_sizing/yoon2b_core_tilt_compare.csv`
