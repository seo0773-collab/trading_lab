# 게이트 체크리스트: yoon2

> 명세(`specs/yoon2.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한 것이다.
> 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 엔진 확장: `simulate_portfolio`에 `exposure_floor`/`exposure_floor_breadth`
  추가, 기본값 0 = no-op(yoon1 동작 불변 — 동일 핸들러 공유)
- [x] 핸들러가 config `exposure_floor` 블록을 읽어 전달, metadata에 `exposure_floor` 노출
- [x] registry 등록(`yoon2`) + config JSON(`exposure_floor` 블록)
- [x] `tests/test_strategy_contract.py` 가 자동으로 yoon2 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src python3 -m unittest discover tests -q` (101개)

## Gate 1 — 무누수·불변식

- [x] floor는 신호(전봉 점수·전봉 시장레짐) 기준으로만 판단 — yoon1의 1봉 지연 체결 경로 재사용
- [x] floor 적용 후에도 비중 ∈ [0,1], `level ≤ 1.0`이라 레버리지 없음(노출 ≤ 1.0)
- [x] 약세 시장(market_flag<1)·breadth 미달 구간에는 미적용 → 하락 방어 보존

## Gate 2 — 합성 드라이런

- [x] 합성 CLI run `succeeded`(`backtest --strategy yoon2 --synthetic`)
- [x] 동일 시드 직접 비교에서 노출↑(0.590→0.664)·CAGR↑(2.51%→2.76%) 확인
  (`specs/yoon2.md` §5)

## Gate 3 — 실데이터 30종목 비교 (대기)

- [ ] **이 환경은 yfinance 차단** — 네트워크 가능 환경에서 실행 필요
- [ ] yoon1 vs yoon2 30종목 phase=all: CAGR↑ & MDD가 EW지수보다 얕음 & Sharpe ≥ EW−0.05
- [ ] 리포트 기록: `reports/profile_sizing/`

## Gate 4 — 판정과 다음 단계

- [ ] 합격선 충족 시: level·breadth 스윕(validation) → test(holdout) 개봉으로 live 판정
- [ ] 부족 시: breadth 게이트/level 조정 또는 조건부 레버리지(별도 변형) 검토
