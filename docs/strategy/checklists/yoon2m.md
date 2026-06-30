# 체크리스트: yoon2m

- [x] yoon2 구조 유지(yoon1j_lp USD + yoon1j_kr KRW, 50/50 월간)
- [x] 두 슬리브 top_k 3/3 inline override (핸들러 `_sub_config` overrides 지원)
- [x] top_k 스윕(1/2/3) 비교 → top_k=3 채택(방어 보존 + 정수주 적합)
- [x] `configs/strategies/yoon2m.json` 추가
- [x] registry 등록 (`Yoon1kHandler`, CATEGORY_HIERARCHICAL)
- [x] 명세 작성
- [x] 부트스트랩 검증: P(MDD<벤치)=95.2% (방어 통계 확정)
- [x] 워크포워드 검증: 롤링3년 MDD방어 17/18
- [x] 200만원 정수주 매입 적합성 실측(94% 투자·추적오차 8.08%p)
- [x] 계약/전체 테스트 통과 확인 (`test_strategy_contract` 3 + 회귀 44, 합성 run_succeeded)
- [ ] holdout test 구간 1회 개봉 검증 (방어 무효화 조건 점검)
- [ ] 모의 전진검증(로컬 원장/저널)에 yoon2m 편입
- [ ] live 자동매수 파이프라인 배선(정수주, 월간) + `live_eligible=True` (게이트 통과 후)
