# 게이트 체크리스트: yoon2

> 게이트는 명세(`specs/yoon2.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한
> 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 구현(`src/trading_lab/strategies/yoon2.py`) — 칼만 MACD 라인·델타 트리거·양방향 실행
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics)
- [x] registry 등록 + config JSON(`yoon2.json`) + dashboard 블록
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (118개)

## Gate 0b — 진입 트리거·익절 사다리 확장

- [x] `entry_trigger`(delta_turn|cross) 선택형. cross = `kal_hist` 0교차(라인 크로싱).
      합성 both에서 trigger별 거래수 상이 확인(delta 36 vs cross 31).
- [x] `tp_enabled` 분포 익절 사다리: in-sample `|kal_hist_delta|/ATR` 분위수
      → ATR 배수 → 3단계 부분청산. 합성 검증: mult≈[0.78,1.80,3.02]ATR,
      `take_profit` 청산행 발생, 진입당 `size_frac` 합 = 1.000(분수 정합).
- [x] `tp_enabled=false`(기본) 시 기존 전량 단일청산과 동일 — 전체 118테스트 통과.

## Gate 1 — 무누수·불변식

- [ ] 칼만/EMA/ATR가 모두 과거 데이터만 사용(룩어헤드 없음)
- [ ] 체결이 신호 봉의 다음 봉 시가(next_open) 이후
- [x] 익절 분포는 `identification` 구간에서만 추정해 validation/test에 적용(룩어헤드 없음)
- [ ] 손절가가 진입 방향에 맞게 설정(롱=아래, 숏=위), stop 체결가 = 손절가
- [x] reverse 시 무한루프 없음(진입봉 단조 증가) · equity > 0 (합성 both+TP 확인)

## Gate 2 — 합성 드라이런

- [x] 합성 CLI/계약 run `succeeded`, 롱·숏 체결 모두 등장 (롱 38·숏 37)
- [x] forecast에 4개 칼만 라인 노출 확인 (kal_macd/kal_signal/kal_hist/kal_hist_delta)

## Gate 3 — 실데이터 단일종목

- [x] 5종목(SPY/QQQ/NVDA/AAPL/MSFT) 일봉 10년, phase=all 비교 배치
      (`scripts/yoon2_compare.py` → `reports/yoon2/compare_all.md`)
- [x] 변형 비교: raw/zero × both/long 의 Sharpe·MDD·거래수

## Gate 4 — 판정과 다음 단계

- [x] **판정**: 델타 전환은 양방향 단독으론 실패(raw(both) 평균 Sharpe −0.39,
      281건 whipsaw). **0선 추세필터 + long-only(`zero(long)`)가 우승**:
      평균 Sharpe **0.83**, 거래 8~14건. NVDA(1.32 vs B&H 1.29, MDD −56% vs
      −66%)·MSFT(0.98 vs 0.93, MDD −25% vs −37%)는 **B&H를 위험조정·낙폭 모두
      상회**. 지수(SPY/QQQ)는 Sharpe 열위지만 MDD 대폭 축소.
- [x] 검증 조합을 config 기본값으로 승격(`direction=long`, `macd_zero_filter=true`).
- [ ] 다음: validation/test 분리 OOS 확인, 종목 유니버스 확대, 추세종목 선별 규칙.
