# 전략 테스트 마스터 체크리스트

모든 전략은 Gate 0부터 순서대로 통과해야 하며, 앞 게이트가 실패한 상태에서 뒤 게이트를 진행하지 않습니다. 각 항목에는 통과 증거(run_id, 리포트 경로, 테스트 출력)를 기록합니다. 전략별 사본은 `scripts/new_strategy.py`가 `docs/strategy/checklists/`에 생성합니다.

## Gate 0 — 명세와 재현성

- [ ] 명세 완성: `docs/strategy/specs/<id>.md`에 `미정` 항목 없음, 무효화 조건과 합격선이 결과 확인 전에 기록됨
- [ ] config 유효성: 필수 키 모두 존재, registry 등록 완료
  - `python -m unittest tests.test_strategy_scaffold`
- [ ] 합성 데이터 smoke: `trading-lab backtest --symbol SYNTH --synthetic --phase all` 정상 종료
- [ ] 재현성: 동일 config로 2회 실행 시 metrics 동일 (frozen baseline 저장, 예: `scripts/frozen_h72_price.json` 방식)
- [ ] 플랫폼 회귀: `python -m unittest discover tests` 전체 통과

## Gate 1 — 실행 정확성 (look-ahead / 비용 / 체결)

- [ ] look-ahead 부재: 신호 bar 이후 데이터가 신호 계산에 쓰이지 않음
  - 미래 구간 데이터를 잘라도 과거 신호가 불변인지 확인 (invariance test)
- [ ] 진입 시점: 모든 거래에서 `entry_time > signal_time` (`entry_after_signal == True`)
- [ ] 확신도 임계값: rolling threshold가 signal bar **이전** 데이터만 사용 (`shift(1)` 확인)
- [ ] 수수료 반영: `net_return == gross_return - fee` 검증, fee_bps=0 대비 차이 확인
- [ ] 청산 규칙: horizon 만기 청산과 exit_on_opposite 동작이 명세와 일치
- [ ] 포지션 비중복: 동시에 두 포지션이 열리지 않음
- [ ] 실행 엔진 단위 테스트: `python scripts/test_strategy_execution.py` 통과
- [ ] 분할 무결성: `python scripts/test_generalization_splits.py` 통과 (identification/validation/test 경계 누수 없음)

## Gate 2 — Validation 성과와 데이터 품질

- [ ] 데이터 품질: 자산별 중복 인덱스 0, 결측 처리 기록, 최소 bar 수 충족
- [ ] 개발 자산 validation 실행: `trading-lab backtest --symbol <SYM> --phase validation`
- [ ] 최소 거래 수: 자산당 명세 합격선 이상 (통계적 판단 가능 수준)
- [ ] 성과 합격선: 명세 9절의 validation 합격선 전 항목 충족
- [ ] 거래 경로 점검: `python scripts/test_trade_path_analysis.py` 통과, 소수 거래가 전체 성과를 지배하지 않는지 확인 (상위 1~2개 거래 제외 후에도 부호 유지)

## Gate 3 — 일반화와 강건성

- [ ] 멀티자산: 일반화 자산 세트 전체 실행 (`scripts/generalization_runner.py` 방식), asset class 과반에서 합격선 충족
- [ ] random/placebo 컨트롤: 무작위 방향 신호 대비 성과 우위 (bootstrap 95% CI로 비교)
- [ ] 파라미터 섭동: 핵심 파라미터 ±20%에서 성과 부호 유지 (절벽 형태의 최적점이면 실패)
- [ ] 비용 민감도: 수수료 2배에서 명세 6절 기준 충족
- [ ] leave-one-asset-out: 어느 한 자산을 제외해도 풀링 결과 부호 유지
- [ ] 구간 안정성: validation 구간을 반으로 나눠 양쪽 모두에서 손실이 한쪽에 집중되지 않음

## Gate 4 — Holdout (1회 개봉)

- [ ] 개봉 전 선언: 개봉 일시, 커밋 해시, 고정된 config를 체크리스트에 먼저 기록
- [ ] test 구간 실행: `--phase test` 1회만 실행
- [ ] 합격선 판정: 명세 9절 holdout 합격선과 비교, 결과와 무관하게 보고서 저장
- [ ] 실패 시: 파라미터 수정 금지, 새 버전(`<id>-v2`)으로 재시작

## Gate 5 — Live 적격 (참고: 현재 플랫폼은 live 차단)

- [ ] kill switch 조건이 명세 7절에 정의되고 모니터링 방법 존재
- [ ] 슬리피지/체결 가정의 실거래 타당성 검토 기록
- [ ] registry에서 `live_eligible=True` 전환은 Gate 4 통과 증거 링크와 함께만 수행

## 부록 — 공통 회귀 명령

```bash
python -m unittest discover tests
python scripts/test_strategy_execution.py
python scripts/test_generalization_splits.py
python scripts/test_trade_path_analysis.py
python -m compileall src indicators strategies dashboard scripts tests
```
