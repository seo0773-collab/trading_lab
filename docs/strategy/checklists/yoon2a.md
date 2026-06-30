# 체크리스트: yoon2a

- [x] yoon2 파생 구조 정의(코어 74% + 레버리지 26%)
- [x] 생존편향 회피 레버리지 선정: 섹터 손픽(QLD/USD) 대신 광의 시장 SSO(2x S&P500)
- [x] 레버리지 히스토리 확인(SSO 2006년 상장 → 2008 위기 구간 실측 가능)
- [x] 비중 1차 스윕(Strict: MDD≤50/50 B&H) → 65/35
- [x] 비중 재최적화(Strict+Sharpe: MDD≤B&H **그리고** Sharpe≥B&H, 1%p 격자) → **74/26 채택**(CAGR 17.20%/MDD -26.25%/Sharpe 1.097, B&H 양 축 지배). 근거 `reports/profile_sizing/yoon2a_sso_reopt.md`
- [x] 공통 StrategyHandler(`Yoon1kLeverageMixHandler`) 재사용 + config/registry 등록
- [x] 명세/체크리스트 작성
- [x] 합성 백테스트(계약 테스트) 통과
- [x] 전체 테스트 통과

## 모의 전진검증 환경(KIS mock)

- [x] live 평탄화: yoon2a.json `sub_strategies`(yoon1j_lp 0.325/yoon1j_kr 0.325/sso_bh 0.35)
- [x] SSO buy&hold 슬리브 디스크립터(`configs/strategies/sso_bh.json`, kind=buy_hold)
- [x] `paper_trade.py` buy_hold 분기 → `sso_bh_journal.jsonl` 목표 {SSO:1.0} 적립
- [x] 리밸런서 N-슬리브 일반화 확인(`yoon1k_kis_rebalance.py --strategy yoon2a`)
- [x] sub_strategies↔leverage_mix 비중 일관성 테스트
- [ ] KIS 모의 자격증명으로 3슬리브 드라이런/소액 모의 제출 실측(영업일·세션창)
- [ ] `kis_track`로 모의 NAV vs 50/50 벤치 전진 기록 적립

## 실거래 게이트(차단 유지)

- [ ] OOS(validation/test) 구간 성과 재확인
- [ ] 실거래 전 환전비용·세금·정수주 제약 반영
- [ ] 실계좌 경로 차단 해제 + `live_eligible=True` (위 게이트 통과 후)
