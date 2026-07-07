# 체크리스트: final0706_qld

- [x] final0706 코어 + QLD 오버레이 블렌드 구조 정의(기존 yoon1k_levmix 패턴 재사용)
- [x] QLD 비중 스윕(0~100%, `scripts/profile_sizing/final0706_qld_mix_sweep.py`)으로 MDD -30% 교차점 탐색
- [x] 선형보간 지점(QLD 36.5%) 실측 재검증 — 보간 추정치와 실측 CAGR 일치 확인
- [x] SPY/QQQ/QLD 원화환산 buy&hold 대비 페어드 부트스트랩(8000회) 실행
- [x] 기존 `Yoon1kLeverageMixHandler` 재사용 확인(신규 핸들러 코드 없음)
- [x] config/registry 등록
- [x] 명세 작성(docs/strategy/specs/final0706_qld.md)
- [ ] 전체 테스트 통과 확인(`tests/test_strategy_contract.py` 등)
- [ ] 워크포워드/롤링 안정성 검증(1차 등록에는 스윕 point estimate만 반영)
- [ ] final0706 승격(페이퍼 3개월+KIS 실수수료) 이후 재검토
- [ ] 실거래 전 QLD 자체 운용보수·환전 스프레드·세금 반영
