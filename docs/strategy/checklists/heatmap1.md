# 게이트 체크리스트: heatmap1

> 게이트는 명세(`specs/heatmap1.md`)에 고정한 합격선을 결과 확인 전에 잠그기
> 위한 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 코어 순수 함수(`scripts/volume_profile.py`) — window_histogram·value_area·rolling_profile_levels·build_heatmap·render
- [x] 핸들러 구현(`src/trading_lab/strategies/heatmap1.py`) — 롤링 POC/VAH/VAL + va_reversion/va_breakout 실행
- [x] StrategyArtifacts 계약 충족(forecast[poc/vah/val]·trades·equity·metrics)
- [x] registry 등록 + config JSON(`heatmap1.json`) + dashboard 블록
- [x] 새 exit_reason(poc_target, va_stop) → presentation.py `EXIT_REASON_LABELS` 추가
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [ ] 전체 테스트 통과: `PYTHONPATH=src python -m unittest discover tests -q`

## Gate 1 — 무누수·불변식

- [ ] 롤링 레벨이 t 이하 데이터만 사용(룩어헤드 없음) — rolling_profile_levels 윈도우 점검
- [ ] 체결이 신호 봉의 다음 봉 시가(next_open) 이후
- [ ] 손절가가 진입 방향에 맞게(롱=VAL 아래, 숏=VAH 위), stop 체결가 = 손절가
- [ ] 진입봉 단조 증가(무한루프 없음) · equity > 0
- [ ] window_histogram 합 ≈ 윈도우 총 volume(균등분배 sanity)

## Gate 2~5 — 데이터·검증 (실데이터 게이트)

- [ ] equity(yfinance) 실데이터 1종목 run: forecast 오버레이·trades 정상
- [ ] crypto(ccxt) 실데이터 run(로컬, 네트워크 필요): 페이지네이션·integrity
- [ ] val/test 분할 성과(B&H 대비 위험조정) — 모드별(va_reversion vs va_breakout) 비교
- [ ] 2D 히트맵 PNG 시각 점검(고볼륨 수평 밴드)
