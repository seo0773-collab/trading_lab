# EWY/SPY 리스크오프 게이트 한계기여 (Phase 0-c)

EWY/SPY를 비중 틸트가 아니라 **총노출 디그로싱 게이트**(글로벌 위험선호 프록시)로
사용. 판정 = "VIX/200MA 단독 게이트(G0)에 EWY/SPY를 더하면(G1) 더 나아지나".
게이트 파라미터는 yoon4 레짐에서 사전고정(SPY 200MA, VIX>25, off_scale 0.4), 미적합.

## 판정: ❌ 한계기여 없음 — EWY/SPY 추가 가치 없음

| 변형 | all CAGR | all MDD | all Sharpe | risk-off 일수 |
| --- | ---: | ---: | ---: | ---: |
| G_none (50/50) | 13.99% | -14.46% | **1.143** | 0 |
| G0 (VIX/200MA) | 12.32% | -10.26% | 1.116 | 875 |
| G1 (+EWY/SPY) | 9.86% | -10.56% | 1.078 | **2798** |

**ΔSharpe (G1−G0):** train **−0.089**, validation +0.119, test +0.016, all **−0.038**.

### 근거
- **부호가 일관되지 않음**(train·all 음수, val·test 양수) → robust한 우위 아님. validation 개선은 마침 validation이 약세장이라 노출을 줄인 덕일 뿐 타이밍 스킬이 아님.
- **결정적 결함**: EWY/SPY<200MA가 전 구간의 **62%(2535/4080일)** 동안 참. 이건 "위험 에피소드 감지"가 아니라 **"한국이 미국 대비 구조적으로 약했다"** 는 장기 추세를 반영. risk 게이트로 쓰면 거의 상시 디그로싱 → CAGR만 깎임(12.3%→9.9%).
- G0(VIX/200MA) 자체도 yoon2 코어가 이미 방어적(MDD −14%)이라 효익이 작음(Sharpe ~평탄, CAGR 손실). 그 위에 EWY/SPY는 순악화.

## 결론
- EWY/SPY는 risk-off 게이트로도 기존 VIX/200MA에 아무것도 더하지 못함.
- 단일/basis/다주기 틸트(Phase 0, 0-b)에 이어 **게이트 용법까지 실패** → EWY/SPY로 추출 가능한 알파 없음을 사실상 확정.

## 파일
- 스크립트: `scripts/profile_sizing/yoon2_riskoff_gate_compare.py`
- CSV: `reports/profile_sizing/yoon2_riskoff_gate_compare.csv`
