# 게이트 체크리스트: yoon1h (POC/VA 매물대 위치 사이징)

> 게이트는 명세(`specs/yoon1h.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한
> 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 = `ProfilePortfolioHandler` 공유(전용 코드 없음). 사이징 입력 교체는
      `scripts/profile_sizing/profile.py`(`_poc_va`/`_va_position`) +
      `run.py`의 `position_source` 분기로 구현(기본 `percentile` → 동작 불변).
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics) — 공유 핸들러 계약 그대로
- [x] registry 등록 + config JSON(`yoon1h.json`, `position_source=poc_va` + `profile.compute_va`)
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (121개)

## Gate 1 — 무누수·불변식

- [x] POC/VA = rolling profile(t 이하 봉만 누적) + 엔진 shift(1) → 룩어헤드 없음.
- [x] warmup/빈 분포 → `va_position`=NaN → 비중 0(불변). 합성 2000봉 중 warmup 199봉만 NaN.
- [x] `position_source=percentile`(기본) + `compute_va=false` 시 다른 전략
      점수·동작·성능 불변 — 전체 테스트 통과로 확인.
- [x] `va_position` 단위 검증: 범위 [0,1], VAL→0·POC→0.5·VAH→1, VA 밖 외삽+클립,
      비대칭 VA(POC 비중심) 상·하 반폭 정규화 확인(합성 시리즈).

## Gate 2 — 합성 드라이런

- [x] 합성 유니버스(SYN1~6)도 poc_va 경로 통과 → 계약 run `succeeded`
      (test_strategy_contract 자동 포함 + CLI `backtest --synthetic` succeeded).

## Gate 3 — 실데이터 검증 (megacap30, 주 벤치마크 SPY)

- [x] yoon1b(percentile) vs yoon1h(poc_va) phase val/test/all 비교
      (`scripts/yoon1h_poc_va_compare.py` → `reports/profile_sizing/yoon1h_poc_va.md`).
      va_pct 민감도(0.6/0.7/0.8) 포함.
- [x] 방향 확정: 전 phase에서 yoon1b 근소 우위, va_pct 둔감. **개선 없음**.
      - val: yoon1b Sharpe 0.932 vs yoon1h ~0.86 (yoon1h MDD만 -19.5→-18.1% 소폭↓)
      - test(holdout): yoon1b 1.331 vs yoon1h ~1.317 (CAGR·MDD 거의 동일)
      - all: yoon1b 1.142 vs yoon1h ~1.130

## Gate 4 — 판정과 다음 단계

- [x] **판정 = 기각**. holdout test ΔSharpe −0.013~−0.015(전 va_pct), MDD 개선도 미미
      → 방어다이얼도 못 됨. 원인: POC(프로파일 최빈)와 percentile(누적분포 위치)은
      같은 볼륨 프로파일에서 나와 상관이 높다 → 같은 정보. VA는 좌우 70%만 보아 오히려
      소폭 정보손실. **메가캡 유니버스에서 사이징 입력 손질로는 base를 못 넘는다는
      기존 패턴 재확인**(역변동성·칼만오버레이·모멘텀게이트와 동일 결론).
- [x] 코드는 무해 보존(코어 `_poc_va`/`_va_position` 재사용 가능, 기본 `percentile`
      이라 다른 전략 불변). registry yoon1h는 **enabled 유지**: 계약 테스트가
      enabled 전략만 돌아 poc_va 경로 회귀 보호 + 비교 재현이 가능하기 때문. 기각이나
      기본 동작 불변이라 무해(운영 후보 아님, 비교용 보존).
