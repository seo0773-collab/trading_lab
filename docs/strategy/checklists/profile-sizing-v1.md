# 게이트 체크리스트: profile-sizing-v1

> 게이트는 명세(`specs/profile-sizing-v1.md`)에 고정한 합격선을 결과 확인 전에
> 잠그기 위한 것이다. 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 연구 모듈 구현(`scripts/profile_sizing/`) — 지표/프로파일/국면/사이징/엔진
- [x] 핸들러가 StrategyArtifacts 계약 충족(forecast·trades·equity·metrics)
- [x] registry 등록 + config JSON + dashboard 블록
- [x] `tests/test_strategy_contract.py` 가 자동으로 본 전략 포함·통과
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (90개)

## Gate 1 — 무누수·불변식

- [x] cumulative profile 무누수(미래 봉 절단 시 과거 percentile 불변)
- [x] base_target_weight가 percentile에 대해 단조 감소(저가권↑)
- [x] DEFENSE에서 신규 매수 금지 / 기존 비중 cap까지 축소
- [x] rebalance 봉당 변화 한도(max_trade_weight_per_bar) 준수
- [x] 비중 ∈ [0,1], equity > 0

## Gate 2 — 합성 드라이런

- [x] 합성 CLI/계약 run `succeeded`, 4개 국면 모두 등장
- [x] 합성에서 MDD가 B&H보다 얕음(-10% vs -25% 수준)

## Gate 3 — 실데이터 30종목 B&H 배치

- [x] `scripts/profile_sizing/batch.py --phase all` 30종목 전부 성공
- [x] 리포트 기록: `reports/profile_sizing/perf_batch.md`
- 결과 요약(phase=all): MDD 개선 종목 비율 **100%**(합격선 ≥70% 충족),
  평균 MDD -38% vs B&H -72%, 평균 Sharpe 0.55 vs 0.63(합격선 −0.15 이내 충족),
  총수익률은 B&H 대비 전반적 열위(평균 익스포저 ~36%의 구조적 결과 — 명세 §2 예상).

## Gate 4 — 판정과 다음 단계

- [x] 합격선(MDD 개선·Sharpe) 충족 → 전략의 **낙폭 축소 엣지는 확인**.
- [x] 개선 변형 2종 구현·배치 비교(`reports/profile_sizing/compare.md`, phase=all, 30종목):

  | 전략 | 익스포저 | 평균 총수익 | 중앙값 총수익 | Sharpe | MDD | MDD 개선 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | profile-sizing-v1(baseline) | 36% | 1741% | 1091% | 0.547 | -38.2% | 100% |
  | profile-sizing-trend-v1(①) | 43% | 3955% | 1500% | 0.566 | -40.9% | 100% |
  | profile-sizing-exp-v1(②) | 40% | 2644% | 1626% | **0.572** | -40.9% | 100% |

  두 변형 모두 예상대로 익스포저·수익·Sharpe ↑, MDD 소폭 ↑(-38%→-41%). **exp-v1이
  최고 Sharpe·최고 중앙값 수익**으로 위험조정 기준 최선. 단 셋 다 대형주 B&H 총수익은
  여전히 0% 우위(de-risking의 구조적 한계).
- [ ] live_eligible 전환은 약세장 빈번 자산/지수 재평가 + holdout(test) 개봉 1회로 결정.
