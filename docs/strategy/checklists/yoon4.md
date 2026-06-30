# 체크리스트: yoon4

- [x] new_layer.txt 설계 → yoon4(글로벌 자금흐름 국가 로테이션)로 1차 구현
- [x] yoon2(50:50)는 베이스로 보존, yoon4 신규 전략으로 추가
- [x] CRI 팩터(모멘텀 3/6/12M·RS·신고가거리·거래량·변동성축소·가속도) Z-score 통합
- [x] 글로벌 Risk-On/Off 레짐 게이트(VIX·DXY·SPY 200MA) + 안전자산 회전
- [x] 리프=국가 ETF 직접 보유(v1), 섹터 계층은 후속 레이어로 명시
- [x] 전 종목 미국 상장 → KIS 해외주식 거래 가능성 확인·문서화(거래소 매핑 표)
- [x] registry 등록(hierarchical, Yoon4Handler) + config + 명세/체크리스트
- [x] 합성 백테스트 통과(forecast/trades/equity/metrics + wave/perf 패널)
- [x] 전체 테스트 통과 (159 tests OK / `PYTHONPATH=src python -m unittest discover tests -q`)
- [x] 실데이터 스모크(phase=all): yoon4 CAGR 9.3%·MDD -26.2%·Sharpe 0.51 vs SPY 원화 B&H CAGR 12.2%·MDD -30.2%·Sharpe 0.63 (방어형, 낙폭↓·수익 양보) — 가중치/top_n 튜닝은 후속
- [ ] (후속) CRI 가중치 최적화 스윕 · yoon2 대비 비교 리포트
- [ ] (후속) 국가→섹터→리프 3계층 풀스택
- [ ] (후속) KIS paper/dryrun 연동(live_eligible 승격)
