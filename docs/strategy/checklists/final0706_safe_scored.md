# 체크리스트: final0706_safe_scored

- [x] 안전자산 슬리브 profile 스코어링 opt-in 구현(`portfolio.py` `safe_scores`
      파라미터, `profile_portfolio.py` `safe_sleeve.profile_scored`)
- [x] 기존 균등 배분 동작은 `safe_scores=None`(기본)일 때 완전히 불변임을 확인
      (전체 184개 테스트 통과, 회귀 없음)
- [x] 슬리브 단독(US/KR) 성과 비교(`final0706_safe_sleeve_profile_study.py`)
- [x] 전체 파이프라인(`Yoon1kHandler`) 성과 비교 — all/test/recent_3y 전 구간
- [x] 원본 대비 페어드 블록 부트스트랩(8000회) — P(Sharpe 우위) 82.9%, P(CAGR 우위)
      88.5%, P(MDD 더 얕음) 60.1%
- [x] mid_peak buckets와의 결합 실측 및 기각 확인(반영 안 함)
- [x] config/registry 등록(`final0706_safe_scored.json`, `Yoon1kHandler` 재사용)
- [x] 명세 작성(docs/strategy/specs/final0706_safe_scored.md)
- [ ] 전체 테스트 통과 확인(`tests/test_strategy_contract.py` — 신규 전략 자동 포함)
- [ ] 롤링 워크포워드/안정성 검증(1차 등록에는 전체 구간 point estimate만 반영)
- [ ] 안전자산 전용 buckets/regime_cap 분리 튜닝(현재는 주식용 buckets 재사용)
- [ ] final0706 승격(페이퍼 3개월+KIS 실수수료) 이후 재검토·원본과 비교해 채택 여부 판단
