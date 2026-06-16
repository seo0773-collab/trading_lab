# profile-portfolio-v1 백업 (2026-06-16)

파라미터 스윕(validation→holdout)으로 튜닝한 **검증된 작동 스냅샷**.
2순위(시장 레짐 필터) 작업 전 보존.

## 이 시점 성과 (30종목, phase=all)
- 기본 조합: top_k=20, monthly, trend floor=1.0
- Sharpe 1.072 (B&H 1.045 추월) · MDD -32.5% (B&H -53.5%) · CAGR 14.1% (B&H 18.0%)
- git commit: e9b75c5

## 구성
- scripts/profile_sizing/ : 연구 파이프라인(config·indicators·profile·regime·sizing·
  engine·account·portfolio·run·batch·portfolio_sweep)
- strategies/ : 핸들러(profile_sizing.py, profile_portfolio.py) + base.py
- configs/ : 전략 5종 config
- docs/ : 명세
- test_profile_sizing.py : 단위 테스트

## 복원
파일을 원위치로 복사하면 된다(이 디렉토리는 import 경로 밖이라 실행에 영향 없음):
  scripts/profile_sizing/, src/trading_lab/strategies/, configs/strategies/ 등.
