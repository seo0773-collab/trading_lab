"""profile-sizing-v1 연구 파이프라인 (profile_plan.txt 모듈화).

Cumulative Profile Percentile 기반 **목표 비중(target-weight) 사이징 전략**.
단순 매수/매도 신호가 아니라 매 봉 목표 주식 비중을 계산하고 regime cap·rebalance
규칙으로 실제 비중을 조절한다. 가격 예측 모델이 아니라 "현재 가격 위치 판단 +
하락장 생존 + 단계적 회복" 포지션 관리기다.

모듈 구성(plan §1을 trading_lab 관례로 압축):
- config: 파라미터(plan §14)
- indicators: base_cycle + cycle multiple(plan §3)
- profile: rolling/cumulative profile + percentile(plan §4·§5)
- regime: 4단계 국면 판정(plan §6)
- sizing: weight model + regime cap + target weight + rebalancer(plan §7~§10)
- engine: 비중 경로 → 평가자산 equity + buy&hold 비교 + lot 기반 trades(plan §12)
- synthetic: 결정적 합성 OHLCV(계약 테스트용)
- run: 위를 묶어 아티팩트 dict 생성(핸들러가 소비)
"""
