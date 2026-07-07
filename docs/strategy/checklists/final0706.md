# 체크리스트: final0706

- [x] 기반 검증 상속 — yoon2_final_fx_credit_gate 등록 게이트(리드타임 진단·240그리드·워크포워드·부트스트랩·비용민감도) 통과
- [x] 노이즈 필터링(min_consecutive=2) 채택 근거 확인
- [x] 2차 개선 4종(IEF 대피/VIX AND결합/연속형노출/off_scale 재그리드) 검증 — IEF 대피만 채택, 나머지 기각
- [x] off_scale 재그리드 1위 후보 워크포워드/회전율/비용 정밀검증 — 기각(회전율 2.7배로 비용 취약)
- [x] 미국 top3 상관계수 진단 + 상관 페널티 선택 실측 — 상관은 줄어도 Sharpe 악화로 기각
- [x] top_k 스윕(1~8) — 미국 top_k=5 유망하나 합성전략 레벨 미검증, 현 구성(top_k=3) 유지
- [x] 코드 재검토 — `risk_gate.continuous` 변수 충돌 발견·수정, 실사용 결과 불변 재확인
- [x] config 동결(`final0706.json`) + registry 등록(핸들러 `Yoon1kHandler` 공유)
- [x] 명세 작성 (승격 게이트·표본 집중 한계 명시)
- [x] 계약/전체 테스트 통과 확인
- [x] CLI 합성 run succeeded 확인 + 실데이터 직접 재현(CAGR/MDD/Sharpe 일치)
- [x] (2026-07-07 비판적 평가) 연환산 버그 수정·공표 수치 정정 — 봉수 연환산+.KS 갭(2008년 0봉)으로 CAGR 과대표기, 권위 정정치=14.39%/-16.40%/1.176(2011-01~, `final0706_benchmark_compare_v2.md`)
- [x] 데이터 갭 진단·평가 표준 시작점 규칙 수립 (`final0706_data_gap_diagnosis.md`)
- [x] 신용게이트 GFC 대리검증 — 발동·-70일 선행·MDD +1.1%p, 위기 표본 n=2 (`final0706_credit_gate_gfc_proxy.md`)
- [x] 양도세(해외 22%) 반영 세후 비교 — 청산 전제면 B&H 대비 불리하지 않음 (`final0706_tax_impact.md`)
- [x] 레짐 의존 분해·기대치 문서화 — 2025~26 주역=KOSPI 랠리, 스펙 §6 (`final0706_regime_attribution.md`)
- [ ] 페이퍼 전진검증 3개월 이상 — **2026-07-07 개시**, 판정 기준 사전 등록(`docs/strategy/plans/final0706_paper_criteria.md`), 판정일 2026-10-07+
- [ ] KIS 실수수료·환전우대 확정 → live_eligible 승격 판단
