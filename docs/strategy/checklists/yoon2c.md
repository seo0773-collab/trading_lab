# 체크리스트: yoon2c

- [x] yoon2 하위 슬리브 유지(yoon1j_lp + yoon1j_kr)
- [x] World Bank 버핏지수 레벨 비율 테스트
- [x] World Bank 버핏지수 변화율 direct/revert 테스트
- [x] revert 변화율 강도 스윕(`k=0.00~1.50`)
- [x] `k=1.25` 채택: MDD 거의 유지, CAGR/총수익 개선
- [x] `Yoon1kHandler`에 동적 통화 목표비중 경로 추가
- [x] `configs/strategies/yoon2c.json` 추가
- [x] registry 등록
- [x] 명세 작성
- [ ] 정수주/DCA 실행 조건에서 yoon2 대비 우위 유지 확인
- [ ] live 레이어가 동적 통화비중을 계산하도록 확장
- [ ] 실거래 전 환전비용·세금·정수주 제약 반영
- [ ] 실계좌 경로 차단 해제 + `live_eligible=True` (위 게이트 통과 후)
