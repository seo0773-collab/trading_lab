# 게이트 체크리스트: yoon1g (회복 레버리지 슬리브)

> 게이트는 명세(`specs/yoon1g.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한
> 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 = `ProfilePortfolioHandler` 공유(전용 코드 없음). 레버리지 게이팅은
      `compute_universe(..., leveraged_symbols, leverage_regimes, market_ok)`로 구현
      (additive·기본 off).
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics) — 공유 핸들러 계약 그대로
- [x] registry 등록 + config JSON(`yoon1g.json`, leverage_sleeve 블록)
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (118개)

## Gate 1 — 무누수·불변식

- [x] 슬리브 게이트 = 자기 RECOVERY 국면(인과적 분류) ∧ 시장 200MA(min_periods=ma_len,
      과거만) — 룩어헤드 없음. warmup NaN은 정상(True) 취급 후 ffill.
- [x] `leverage_sleeve.enabled=false`(기본) 시 다른 전략 점수·동작 불변 — 118테스트 통과로 확인
- [x] 레버리지 비용·낙폭은 합성 마진(이자 모델)이 아니라 **실제 2x ETF 가격에 내장**
      (decay·증폭 자동 반영) — 정직한 비용 회계

## Gate 2 — 합성 드라이런

- [x] 합성 유니버스(SYN1~6)는 슬리브 심볼과 무매칭 → 게이트 미적용, 기존 경로와 동일 동작
- [x] 계약 run `succeeded` (test_strategy_contract 자동 포함)

## Gate 3 — 실데이터 검증 (2007-02~, 주 벤치마크 SPY)

- [x] holdout(test) / full-cycle(all) 비교: 1x(yoon1f) vs 무게이팅 vs RECOVERY게이팅 vs
      RECOVERY+시장ON(yoon1g) vs SPY (`scripts/yoon1g_recovery_leverage_compare.py`
      → `reports/profile_sizing/yoon1g_recovery_leverage.md`)
- [x] 연도별 2x 점유율 진단으로 게이트 의도 작동 확인(위기 바닥·약세장만 0, 회복기 유지)

## Gate 4 — 판정과 다음 단계

- [x] **판정**: 시장필터 보강이 RECOVERY-only의 2008 취약성을 완치.
      - holdout: yoon1g Sharpe **1.065**·CAGR **+10.0%**·MDD -9.1% — 1x(0.972/+8.4%)
        및 RECOVERY-only(1.019/+9.7%) 모두 상회.
      - full-cycle: yoon1g Sharpe **0.851 > 1x 0.812**·MDD **-17.6%(=1x 동일)** —
        RECOVERY-only(0.774/-22.3%, 1x 열위)를 뒤집어 robust.
      - 2x 점유율: 2008 11%→0%, 2022 5%→0%(헛반등 차단), 회복기 09·16·20 유지.
- [x] config 기본값 = 검증 조합(MIX 유니버스·top_k 12·require_market_on=true)으로 등록.
- [ ] 다음: 더 긴 사이클 표본(2x inception 한계) 보강, 슬리브 단계적 비중(회복 후기 가중) 실험.
