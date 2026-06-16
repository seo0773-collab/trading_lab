# yoon1 백업 (2026-06-16)

가장 성공적인 전략 yoon1(구 profile-portfolio-v1) 스냅샷.
다종목 포트폴리오: 매월 상위 K=20 상승종목 추종 + 개별 방어 합산(현금화) + SPY 시장필터.

## 성과 (30종목, phase=all, 공정 EW지수 벤치마크)
- 전략: CAGR 13.9% / MDD -29% / Sharpe 1.133
- 벤치마크(EW지수): CAGR 18.0% / MDD -53.5% / Sharpe 1.046
- 요약: 수익은 일부 양보, 위험조정 소폭 우위, 낙폭 절반. holdout·생존편향 점검 통과.
- git commit: 881c55e

## 구성
scripts/ : 연구 파이프라인 전체 + portfolio_sweep + survivorship
strategies/ : profile_portfolio.py(yoon1 핸들러), profile_sizing.py, base.py
configs/yoon1.json : 기본 파라미터(top_k=20, monthly, floor=1.0, 시장필터 on)
docs/, reports/ : 명세 + 스윕/생존편향 리포트
