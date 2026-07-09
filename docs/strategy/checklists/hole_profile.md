# 체크리스트: hole_profile

- [x] 원본 리뷰: `hole_profile.txt`(Pine "Multiscale Market Structure") — 매매 시그널 없는 순수 구조/프로파일 지표임을 확인
- [x] 방향 가설·검증자산 사용자 확인: 돌파/지속 모멘텀(소진/반전 아님) · BTC/USDT 일봉
- [x] 지표 이식 (`hole_profile_features.py`) — ATR 방향전환 L0 지그재그 + `f_next_candidate` 재귀압축 L1/L2 + expanding 백분위 프로파일(무누수)
- [x] 실데이터 sanity check — L0/L1/L2 확정 피벗수 934/287/79, profile_rank 분포 정상
- [x] 검증 스터디 (`hole_profile_study.py`) — decile + 게이트(threshold) Newey-West t / Bonferroni·FDR: **L1·threshold=0.70만 채택**(L0 무의미, L2 불안정)
- [x] 분할 안정성 확인(train/val/test, 전/후반) — 15/15 edge 부호 일관(양+), val/test 개별 유의성은 미확정
- [x] 파이프라인 (`hole_profile.py`, `simulate_hole_profile`) + 핸들러 (`HoleProfileHandler`) + config + registry 등록
- [x] 명세 작성 (`docs/strategy/specs/hole_profile.md`)
- [x] 계약 테스트 통과 (`tests/test_strategy_contract.py`, 3 tests OK)
- [x] 실데이터 백테스트: all/validation/test phase B&H 대비 성과표 (명세 §6.0, 최초 config) — all/validation Sharpe는 B&H 열위, test만 MDD 큰 폭 개선(거래수 적어 강건성 미확인)
- [x] 전체 테스트 스위트(`unittest discover tests`) 통과 확인 (184 tests OK)
- [x] 청산 규칙(손절/percentile_fade/시간청산) 파라미터 스윕 (`hole_profile_exit_sweep.py`, 명세 §6.1) — stop_atr_mult/max_hold_bars는 구조적 확정 청산이 선점해 거의 무의미함을 확인, entry_percentile 0.70→0.60·exit_percentile 0.30→0.50 채택(train Sharpe 기준 선택+val/test 확인+원시신호 재검증으로 뒷받침). config/명세 갱신 완료
- [x] 대시보드 결과창 보강: (1) Pine 원본 "Time Bin"/"Price % Bin" 두 프로파일을 `extras.duration_profile`/`pct_move_profile`(bar 패널, config `dashboard.panels`)로 노출 — 전체 히스토리 누적, 전략별 코드 없이 범용 패널 메커니즘 재사용. (2) 매수/매도 마커는 기존 `build_price_figure`가 이미 범용 제공. (3) 손절/익절선을 `build_price_figure`에 범용 추가 — 거래별 진입~청산 구간에만 그리고 포지션 정리 후 `None`으로 끊어 다음 거래와 섞이지 않게 함(모든 전략의 stop_loss_price/take_profit_price에 적용, hole_profile 전용 아님). 전체 테스트 스위트 재확인 완료
- [x] 프로파일 bin 세분화(20→40) + 리스크 파라미터(stop/take_profit) 스윕 (`hole_profile_risk_sweep.py`, 명세 §6.2, 사용자 요청 "더 보수적으로") — `take_profit_atr_mult` 신설 + `stop_atr_mult` 3.0→2.0으로 조여 train/val/test/all 4구간 동시 개선(all Sharpe 0.640→0.805, CAGR 20.2%→24.9%, 거래수 42→84). 청산 사유 분포 건전화(take_profit_atr 43%·structure_confirmed 37%·stop_atr 13%·percentile_fade 7%). `EXIT_REASON_LABELS` 누락분(stop_atr/take_profit_atr/structure_confirmed/percentile_fade) 뒤늦게 등록. 계약 테스트·전체 테스트 스위트·CLI 백테스트 재확인 완료
- [ ] 타 자산(주식 지수 등) 이식 검증 — 현재 BTC 단일 검증
- [ ] 하락장/횡보장 별도 표본 확인
- [ ] 페이퍼 검증 전 live_eligible 유지 확인
