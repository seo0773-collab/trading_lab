# 테스트 체크리스트: fin-sensitivity-v1

- 생성일: 2026-06-14
- 명세: specs/fin-sensitivity-v1.md
- 각 항목 통과 시 증거(run_id 또는 리포트 경로)를 같은 줄에 기록

> **현재 상태(2026-06-14): Gate 0 통과(합성).** 데이터/모델/신호 모듈 + 핸들러
> 등록 + 합성 파이프라인 계약 테스트까지 완료. 다음은 Gate 1~2의 실데이터
> (yfinance + `scripts/fetch_fundamentals.py`) 검증이다.

## Gate 0 — 명세와 재현성

- [x] 명세 완성: `미정` 항목 없음, 무효화 조건/합격선이 결과 확인 전 기록됨 — specs/fin-sensitivity-v1.md
- [x] 데이터 계층 단위 테스트: as-of 누수 0 / rolling 표준화 누수 0 / 합성 결정성·인과
  - `PYTHONPATH=src python -m unittest tests.test_fin_sensitivity`
- [x] config 유효성 + registry 등록 — `fin-sensitivity-v1` enabled
- [x] 합성 데이터 smoke: `... backtest --strategy fin-sensitivity-v1 --symbol SYNTH --synthetic` → run `succeeded` (run_name 21_fin-sensitivity-v1_합성_랜덤_*)
- [x] 계약 테스트: `tests.test_strategy_contract` 통과(파이프라인·StrategyArtifacts·run_name)
- [x] 플랫폼 회귀: `PYTHONPATH=src python3 -m unittest discover tests` 전체 65개 통과
- [ ] 재현성: 동일 config 2회 실행 metrics 동일 (frozen baseline 저장) — 실데이터 단계

## Gate 1 — 실행 정확성 (look-ahead / 비용 / 체결)

- [ ] look-ahead 부재: 미래 구간을 잘라도 과거 신호 불변 (invariance test)
- [ ] as-of 무결성: 임의 거래일 t의 피처에 available_date > t 재무 0건
- [ ] 진입 시점: 모든 거래에서 `entry_time > available_date` (발표 다음 봉 체결)
- [ ] rolling 표준화/계수가 윈도우 밖 데이터를 참조하지 않음 (셔플 불변)
- [ ] 타깃 실현 시점 > 피처 시점 단언 (forward window 미완성 분기는 학습 제외)
- [ ] 수수료 반영: `net_return == gross_return - fee` 검증
- [ ] 청산 규칙: rebalance/horizon/signal_flip/stop_loss 동작이 명세와 일치
- [ ] 포지션 비중복: 동시에 두 포지션이 열리지 않음

## Gate 2 — Validation 성과와 데이터 품질

- [ ] 데이터 품질: 중복 인덱스 0, 결측 처리 기록, 최소 표본(train_quarters) 충족
- [x] 데이터 소스 교체: yfinance(7분기) → **SEC EDGAR(~71-75분기)** `scripts/fetch_fundamentals.py`
- [x] 개발 종목 실행(EDGAR, phase=all): MSFT/AAPL/JPM/CAT/KO
- [~] 최소 거래 수: 종목별 0~15 (KO 15, CAT 0) — 종목 의존
- [~] **성과 합격선 부분 충족(재설계 후)**: 타깃을 초과수익(SPY 대비)으로, 피처를
  YoY 변화+서프라이즈(YoY 가속)+밸류 상호작용으로 재설계한 결과 —
  IC60 평균 **+0.047(4/5 양수)** 로 전환(이전 -0.09), IC20은 평균 -0.04로 여전히 약함.
  → **신호는 단기(20일)가 아니라 중기(60일) 효과**(plan §17 "현금흐름 늦게 반영" 일치).
  - 참고: raw 수익률 IC(재설계 전)는 AAPL +0.19였으나 초과수익으로 바꾸니 +0.02로 붕괴
    → 그 신호는 펀더멘털 알파가 아니라 시장 베타였음(초과수익 타깃이 이를 폭로).
  → 다음: §28 배치로 수십 종목 IC60 분포 확인(5종목은 표본 과소). config 기본값=
    feature_set redesign, target_excess true.
- [ ] 거래 경로 점검 / placebo 대조 (배치 단계에서)

## Gate 3 — 일반화와 강건성

- [ ] 멀티종목: §28 배치 러너로 유니버스 실행, 과반 종목에서 합격선 충족
- [ ] placebo 컨트롤: 팩터 셔플 대비 IC 우위 (bootstrap 95% CI)
- [ ] 파라미터 섭동: train_quarters/ridge_alpha ±20%에서 성과 부호 유지
- [ ] 비용 민감도: 수수료 2배에서 명세 6절 기준 충족
- [ ] leave-one-quarter: 특정 분기 제외해도 부호 유지
- [ ] 구간 안정성: validation을 반으로 나눠 손실 한쪽 집중 없음

## Gate 4 — Holdout (1회 개봉)

- [ ] 개봉 전 선언: 일시/커밋 해시/고정 config 기록
- [ ] test 구간 실행: `--phase test` 1회
- [ ] 합격선 판정: 명세 9절 holdout 합격선과 비교, 결과 무관 보고서 저장
- [ ] 실패 시: 파라미터 수정 금지, 새 버전(`-v2`)으로 재시작

## Gate 5 — Live 적격 (현재 플랫폼 live 차단)

- [ ] kill switch / 모니터링 정의
- [ ] 슬리피지·체결 가정 실거래 타당성 검토
- [ ] `live_eligible=True` 전환은 Gate 4 증거 링크와 함께만
