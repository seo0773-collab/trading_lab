# 게이트 체크리스트: yoon3 (칼만 히스토그램 모멘텀 게이트)

> 게이트는 명세(`specs/yoon3.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한
> 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 = `ProfilePortfolioHandler` 공유(전용 코드 없음). 게이트는
      `scripts/profile_sizing/momentum.py`(`momentum_gate`) +
      `compute_universe(..., mom_gate=...)`로 구현(additive·기본 off).
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics) — 공유 핸들러 계약 그대로
- [x] registry 등록 + config JSON(`yoon3.json`, mom_gate 블록)
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (118개)

## Gate 1 — 무누수·불변식

- [x] 백분위 = 시작~현재 봉만 누적(과거·현재) + 엔진 shift(1) → 룩어헤드 없음.
- [x] warmup/결측 게이트 = 1.0(불변), z 정규화 std는 min_periods로 과거만.
- [x] `mom_gate.enabled=false`(기본) 시 다른 전략 점수·동작 불변 — 118테스트 통과로 확인
- [x] 게이트 단위 검증: 범위 [g_min,1.0], 상승구간 게이트 > 조정구간 게이트,
      contrarian 반전 확인(합성 시리즈).

## Gate 2 — 합성 드라이런

- [x] 합성 유니버스(SYN1~6)도 게이트 경로 통과(close 기반) → 계약 run `succeeded`
      (test_strategy_contract 자동 포함).

## Gate 3 — 실데이터 검증 (megacap30, 주 벤치마크 SPY)

- [x] yoon1b(게이트 off) vs 모멘텀게이트 g_min∈{0.7,0.5,0.3} vs contrarian 비교
      (`scripts/yoon3_momentum_gate_compare.py`
      → `reports/profile_sizing/yoon3_momentum_gate.md`), phase val/test/all.
- [x] 방향 확정: contrarian은 val Sharpe −0.115로 기각, momentum 3종 모두 통과.

## Gate 4 — 판정과 다음 단계

- [x] **판정**: 모멘텀 게이트는 val Sharpe 소폭↑(g0.5 +0.025)·**MDD 전 phase 개선**
      (test -22.9→-18.6%, all -32→-25.4%)이나, 결정적 holdout Sharpe는 base보다
      소폭 낮다(1.307 < 1.335). → yoon1b를 위험조정으로 **이기지 못하는 방어 다이얼**
      (yoon1c 성격). 운영 1순위는 yoon1b 유지, yoon3는 방어형 변형으로 등록 보존.
- [x] **방향 확정 = 양(+) 성과**: 블렌드는 "저가권 × *상승* 모멘텀"이 옳음을 입증.
- [ ] 다음(선택): g_min·norm_window 민감도, rolling vs 누적 분포, 섹터(yoon1f) 결합.
