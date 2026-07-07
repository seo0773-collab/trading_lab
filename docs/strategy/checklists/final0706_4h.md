# 체크리스트: final0706_4h

- [x] TF 강건성 조사(1m~4h) — 5m/15m/30m은 데이터 한도로 검증 불가, 1h/4h는 SMA가
      몇 주~몇 달 메모리를 담아야 경향성 재현됨을 확인
- [x] US 슬리브 단독 cycle_len×rebalance 2D 그리드서치 — cycle=100/rebal=42에서
      Sharpe 105%(동일 3년 구간 일봉 기준 대비)
- [x] 리밸런싱 주기를 SMA와 같은 비율로 줄이는 실험 — 수수료 복리로 Sharpe -700까지
      붕괴 확인, 리밸런싱은 원본과 비슷한 실빈도로 독립 유지해야 함을 확정
- [x] US+KR 2조 구조 전체(통화틸트+신용게이트) 확장 시도 — 최초 크래시(다른 거래소
      인트라데이 봉 시각 불일치로 슬리브 결합 인덱스가 빔)
- [x] 버그 수정: `Yoon1kHandler` 슬리브 결합을 intersection→union+ffill로 일반화
      (일봉 동작 불변, 전체 176개 테스트 통과 확인)
- [x] 2조 구조 레벨 cycle_len×rebalance 재그리드서치(1h/4h) — 4h 최선 cycle=100/
      rebal=31(Sharpe 52%), 1h가 근소하게 더 나음(60%)이나 4h로 config 커밋
- [x] config 등록(`final0706_4h.json`) + registry 등록(핸들러 `Yoon1kHandler` 공유)
- [x] 계약 테스트 통과 확인(`tests.test_strategy_contract`, synthetic 경로)
- [x] 전체 테스트 스위트 통과 확인(공유 코드 수정 회귀 없음)
- [x] 실데이터 백테스트 재현(Sharpe 1.121, CAGR 11.48%, MDD -17.89%, 2024-07~2026-07)
- [x] 명세 작성(승격 게이트·표본 짧음·TF 미확정 한계 명시)
- [ ] 1h 버전(`final0706_1h`) 별도 등록·비교 검토
- [ ] out-of-sample 캘린더 누적(향후 데이터 축적) 후 재검증
- [ ] 페이퍼 전진검증 3개월 이상
- [ ] KIS 실수수료·환전우대 확정(특히 4h 빈도 리밸런싱의 실제 슬리피지가 가정한
      왕복 25bp 안에 드는지) → live_eligible 승격 판단
