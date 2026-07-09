# 체크리스트: final0709

- [x] 리서치: 신용스프레드 외 후보 4종(HYG/LQD, IWM/SPY, UUP, CPER/GLD) 리드타임+오버레이+부트스트랩 실측(`final0706_gate_new_signal_search.md`)
- [x] 최유망 조합(`credit_or_small_cap`) 선정 — 이전 기각된 `vix_or_credit_gate`보다 우수함 확인
- [x] `yoon1k.py`에 `credit_or_small_cap` 게이트 소스 구현(기존 경로 불변, `_ratio_weak` 범용 헬퍼 신설)
- [x] config/registry 등록(`Yoon1kHandler` 재사용, 신규 핸들러 없음)
- [x] 공식 파이프라인 point-estimate 실측(vs final0706, vs SPY/QQQ 벤치) — MDD -16.15%→-11.15%, Sharpe·CAGR 동시 개선
- [x] 페어드 부트스트랩(vs final0706, vs 벤치) — P(Sharpe>) 89.6%·P(MDD얕음) 88.8%(final0706 원 게이트 채택 근거보다 강함)
- [x] 롤링 워크포워드 검증(과최적화 여부) — 16개 윈도우 중 MDD 15/16·Sharpe 13/16 우위, 위기표본 편중 없음
- [x] 회전율/비용 민감도 검증 — 발동빈도 2배 이상 증가(294→677일)해도 25bp(실비용 가정)까지 우위 유지, 50bp에서만 거의 동률
- [x] 명세 §5~7 채우기(`docs/strategy/specs/final0709.md`)
- [ ] 전체 테스트 스위트 통과 확인
- [ ] registry description 최종 수치 반영
- [ ] final0706 승격(페이퍼 3개월+KIS 실수수료) 판정과 함께 base 교체 여부 재논의(차기 base 후보로 유력)
