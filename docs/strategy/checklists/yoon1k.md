# 게이트 체크리스트: yoon1k

계층 포트폴리오(원/달러 × 통화별 yoon1j). [[yoon1k]] 명세 참조.

- [x] **Gate 0 — 등록**: registry `yoon1k`, config `yoon1k.json`, 핸들러 `Yoon1kHandler`,
  명세/체크리스트 작성.
- [x] **Gate 1 — 계약**: `tests/test_strategy_contract.py` 통과(합성 경로:
  forecast.close, equity 1.0기준, metrics 필수키, run_name 규칙). 합성에서는 환율·벤치
  미로드 → 통화 환산/벤치 None으로 폴백, 동작 불변.
- [x] **Gate 2 — 구현 정합**: 하위 yoon1j/yoon1j_kr 위임 호출, USD/KRW 환율 환산,
  50/50 월간 리밸런싱 결합. 엔진 경로 실측 = 사전 프로토타입 수치와 일치(all Sharpe 1.151).
- [x] **Gate 3 — 실데이터 검증**: 원화 기준 2007~2026. 통합 Sharpe 1.151/MDD -14.4%가
  개별 슬리브·벤치·SPY원화 대비 위험조정 최고. 상관 -0.026 확인.
- [ ] **Gate 4 — 견고성**: 통화 비율 스윕(50/50 외)·환전비용 민감도·기간 확장은 오픈.
- [ ] **Gate 5 — 운영**: live_eligible=False. 실거래는 환전·해외/국내 동시 집행이라 별도
  설계 필요(보류).

## 주의/한계
- 절대수익은 시장 대비 양보(방어형). 짧은 강세 holdout(test)에선 벤치가 앞섬.
- 통화 리밸런싱 환전비용 미반영(월 1회라 작지만 보수적으론 약간 깎임).
- trades는 하위 두 전략 체결을 합산(로컬 통화 net_return, 표시용). equity는 환산·결합 기준.
