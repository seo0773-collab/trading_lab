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
- [ ] 페이퍼 전진검증 3개월 이상
- [ ] KIS 실수수료·환전우대 확정 → live_eligible 승격 판단
