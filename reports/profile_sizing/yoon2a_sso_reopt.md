# yoon2a SSO 비중 재최적화 (Strict 단독 → Strict + Sharpe≥B&H)

직전 점검에서 현행 채택값(코어65/SSO35)이 MDD 가드는 통과하나 **Sharpe가 B&H보다
낮다(1.025 < 1.089)** 는 문제를 확인. 여기서 (1) Sharpe 제약 추가, (2) 1%p 미세격자로
프런티어 경계 정밀화. 코어=yoon2(50/50), 오버레이=SSO(USD→KRW), B&H=50/50 패시브.

기간 2007-01-29~2026-06-26 (4080봉). B&H all: CAGR 16.78% / MDD −31.35% / Sharpe **1.089**.

## 핵심 결과

| 후보 | SSO/코어 | CAGR | MDD | Sharpe | MDD≤BH | Sharpe≥BH |
| --- | --- | ---: | ---: | ---: | :-: | :-: |
| **현행** | 35/65 | 18.16% | −30.38% | 1.025 | ✓ | **✗** |
| **Strict 단독 최적** | **37/63** | **18.36%** | −31.29% | 1.009 | ✓ | ✗ |
| **Strict+Sharpe 최적** | **26/74** | 17.20% | −26.25% | **1.097** | ✓ | ✓ |

- **37%**: 기존 "MDD≤B&H에서 CAGR 최대" 목적함수의 *진짜* 경계점. 5%p 격자가 놓친 +0.20pp CAGR(35%→37%)을 회수. 38%부터 MDD −31.74%로 B&H 초과.
- **26%**: Sharpe 제약을 더하면 답. **B&H를 MDD·Sharpe 양 축에서 동시에 이기는** 유일 영역의 최대 CAGR. B&H 대비 CAGR +0.42pp, MDD 5.1pp 얕음, Sharpe 우위.
- Sharpe가 B&H를 넘는 경계는 SSO 26~27% 사이. 27%부터 Sharpe<1.089.
- **robustness**: train/val Sharpe가 SSO 비중에 단조 감소 → 낮은 비중일수록 OOS 안정. 26%가 35/37%보다 train·val 모두 높음.

## 권고
- **위험조정 정직성 기준 → SSO 26%** 채택 권장. 원래 지적한 "Sharpe 열위" 결함을 해소하고 B&H를 양 축에서 지배.
- 순수 공격형 유지(MDD를 B&H까지 허용)라면 → **37%** 가 35%의 상위호환(같은 제약, 더 높은 CAGR).
- 어느 쪽이든 **현행 35%는 비최적**(37%에 CAGR·제약 모두 열위, 26%에 Sharpe 열위).

## 파일
- 스크립트: `scripts/profile_sizing/yoon2a_sso_reopt.py`
- CSV: `reports/profile_sizing/yoon2a_sso_reopt.csv`
