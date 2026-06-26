# 게이트 체크리스트: yoon1i (HVN 지지/저항 기대값 게이트)

> 명세 `specs/yoon1i.md`의 합격선을 결과 확인 전에 잠근다. 증거와 함께만 체크.

## Gate 0 — 구현·계약

- [x] 핸들러 = `ProfilePortfolioHandler` 공유. 게이트는 `scripts/profile_sizing/sr_gate.py`
      (`sr_gate`, heatmap2 `rolling_sr_levels` 재사용) + `compute_universe(..., sr_gate=...)`
      로 구현(additive·기본 off).
- [x] StrategyArtifacts 계약 충족 — 공유 핸들러 계약 그대로.
- [x] registry 등록 + config JSON(`yoon1i.json`, sr_gate 블록).
- [x] `tests/test_strategy_contract.py` 자동 포함·통과.
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (121개).

## Gate 1 — 무누수·불변식

- [x] EV/지지/저항 = t 이하 rolling_sr_levels + 엔진 shift(1) → 룩어헤드 없음.
- [x] warmup·노드 부재 → gate=1.0(불변).
- [x] `sr_gate.enabled=false`(기본) 시 yoon1b와 total_return 완전 동일(1e-9 이내) 확인.
- [x] gate 식 [g_min,1], 지지근처→1·저항근처→g_min, 한쪽노드 EV 1/0 폴백(sr_gate.py).

## Gate 2 — 합성 드라이런

- [x] 합성 유니버스도 sr_gate 경로 통과 → 계약 run `succeeded`.

## Gate 3 — 실데이터 검증 (megacap30, 주 벤치마크 SPY)

- [x] yoon1b vs yoon1i(g_min 0.3/0.5/0.7) phase val/test/all 비교
      (`scripts/yoon1i_sr_gate_compare.py` → `reports/profile_sizing/yoon1i_sr_gate.md`).
- [x] 방향 확정: **전 g_min에서 yoon1b 하회**. test Sharpe 1.331 → 1.206/1.235/1.261
      (ΔSharpe −0.125/−0.096/−0.070). g_min↑(게이트 약함)일수록 yoon1b에 수렴 =
      게이트가 순수 손해. **MDD·노출은 감소**(test -22.9→-20.3~-22.2%, 노출 91→85~89%)
      = 방어쪽으로만 이동.

## Gate 4 — 판정과 다음 단계

- [x] **판정 = 기각**(yoon1h보다 나쁨). 메커니즘: SR 게이트가 *저항 근처 종목을 억제*
      하는데, 그게 **추세 돌파 직전 종목**(상승 모멘텀)을 깎아 멜트업 수익을 잘라낸다.
      yoon1b는 이미 percentile로 저가권을 보고 있어 SR이 추가 엣지를 못 주고, 멜트업
      약점([[meltup_analysis]] 진단)을 악화시킬 뿐. yoon3 momentum_gate처럼 "방어
      다이얼, base 미추월". registry는 enabled 유지(기본 off=불변, 비교/회귀용).
