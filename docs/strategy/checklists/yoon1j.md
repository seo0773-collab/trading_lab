# 게이트 체크리스트: yoon1j (연속형 안전자산 슬리브)

> 게이트는 명세(`specs/yoon1j.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한
> 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 = `ProfilePortfolioHandler` 공유(전용 코드 없음). 슬리브는
      `simulate_portfolio(..., safe_close, safe_ratio, safe_max, safe_ma_len)`
      오버레이로 구현(short_hedge와 동일 자리, additive·기본 off).
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics) — 공유 핸들러 계약 그대로
- [x] registry 등록 + config JSON(`yoon1j.json`, safe_sleeve 블록)
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (121개)

## Gate 1 — 무누수·불변식

- [x] 추세 게이트 = 전봉 종가 > 자기 MA(safe_ma_len, min_periods=ma_len, 과거만)
      .shift(1) — 룩어헤드 없음. 슬리브 목표는 전봉 cash_ratio/점수 기반 spare로 산정.
- [x] `safe_sleeve.enabled=false`(기본) 시 다른 전략 점수·동작 불변 — 121테스트 통과로 확인.
      safe_exposure 컬럼은 슬리브 ON일 때만 forecast에 추가(타 전략 노이즈 없음).
- [x] 레버리지·공매도 없음: 주식+안전+숏헤지 ≤ 1, 슬리브 예산은 spare에서만 차감
      (현금 완충분 회전) → 손실 상한 = 버퍼 크기.
- [x] 안전자산 회전에 yoon1f 거래비용(fee+slippage 10bps/side) 부과 — 정직한 회계.

## Gate 2 — 합성 드라이런

- [x] 합성 유니버스(SYN1~6, synthetic=True)는 safe_close=None → 슬리브 미적용, 기존 경로 동일.
- [x] 계약 run `succeeded` (test_strategy_contract 자동 포함).

## Gate 3 — 실데이터 검증 (max 기간, 주 벤치마크 SPY)

- [x] 사전 반증: 공격섹터↔안전자산 **이진 전환**은 MDD -31.5%/Sharpe 0.551로 현금
      방어(-15.6%/0.753)에 패배 — "버퍼 회전 + 추세 게이트"로 전환한 근거.
- [x] 넓은 격자 스윕(ratio×cap×ma, 월간+거래비용 현실 모델): ma100 단조 우월,
      r1.25/cap0.4/ma100 위험조정 최적 (`scripts/profile_sizing/safe_sleeve_compare.py`
      → `reports/profile_sizing/safe_sleeve_compare.json`).
- [x] 핸들러 경로 실측(엔진 통합본): yoon1j vs yoon1f vs 섹터 현금방어 vs SPY,
      phase=all/test 양쪽.

## Gate 4 — 판정과 다음 단계

- [x] **판정**: 버퍼 회전 + 추세 게이트는 명확한 플러스.
      - all: yoon1j Sharpe **0.868**·CAGR **+8.3%**·MDD -18.9% — 섹터 현금방어
        (0.686/+6.1%/-19.1%) 대비 Sharpe·CAGR↑·MDD 동등.
      - test(holdout, OOS): yoon1j Sharpe **1.180**·CAGR **+11.8%** — 과적합 아님.
      - vs yoon1f: 수익·Sharpe 우위, MDD 양보(-18.9% vs -15.6%) = 역할 분리 트레이드오프.
- [ ] 다음(오픈): 안전자산 추세강도 비례 배분 / 유니버스 확장(IEF·SHY·금광주) /
      TLT·GLD 점수 유니버스 동시 포함 하이브리드 비교.
- [ ] 페이퍼 트레이딩 적립 후 라이브 적격성 재검토(현재 live_eligible=false).
