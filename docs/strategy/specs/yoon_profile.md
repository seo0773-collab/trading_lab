# 전략 명세: yoon_profile

> **SMA200 멀티플(가격 위치) × SMA200 기울기 z-각도(레짐)**의 expanding percentile
> 3×3 그리드 상태별 가중치로 long-only 비중을 정하는 단일종목 전략.
> 출발점은 `profile_plan/profile_plan.txt`의 2D 상태공간 조건부 백테스트.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon_profile |
| 분류 | single (단일종목, 기본 BTC/USDT 일봉) |
| 핸들러 | `src/trading_lab/strategies/yoon_profile.py` (`YoonProfileHandler`) |
| 지표/상태 | `scripts/profile_sizing/cycle_state.py` |
| 파이프라인 | `scripts/profile_sizing/yoon_profile.py` |
| config | `configs/strategies/yoon_profile.json` |
| 검증 스터디 | `scripts/profile_sizing/yoon_profile_study.py` → `reports/profile_sizing/yoon_profile_study.json` (+ cell CSV, 히트맵 PNG) |
| 참고 Pine | `profile_plan/sma200p.pine` (기울기 각도 프로파일), `profile_plan/mcvp.pine` (멀티플 프로파일) |

## 2. 지표 (플랜 고정 파라미터 — 스윕 금지)

```
cycle      = SMA(close, 200)
multiple   = close / cycle
slope_pct  = (cycle − cycle.shift(5)) / cycle.shift(5)
slope_z    = slope_pct / rolling_std(slope_pct, 252)
angle_deg  = degrees(arctan(slope_z))          # z=1 → 45°
```

bin 경계는 **expanding window percentile**(시점 t까지 포함, 최소 252 유효표본)로
3분위 산출 — 전 기간 percentile 사용 금지(look-ahead 차단). 상태 미확정 구간은
`WARMUP`(비중 0). 워밍업 총합 ≈ 200+252+252 ≈ 700봉.

## 3. 가설 검증 결과 (2026-07-03, BTC/USDT 일봉 2017-08~2026-07)

forward 5/20/60일, Newey-West(lag=h) t-stat, 셀 최소 50표본, Bonferroni+BH-FDR
(27검정) 병기. 유효표본 2,532봉(워밍업 이후).

| 가설 | 판정 | 근거 |
| --- | --- | --- |
| H1 눌림목 알파 (저멀티플×상승기울기) | **변형 채택** | 엄밀 셀 (0,2)는 표본 7개(구조적 공집합). 알파는 **(0,1) 저멀티플×중립기울기**에서 발생: fwd60 +28.2% (t=4.79, FDR 유의), fwd20 +7.2% (t=2.18), walk-forward 3사이클 전부 양수 |
| H2 떨어지는 칼 회피 (저멀티플×하락기울기) | **보류(방향 일치)** | (0,0) 전 horizon 음수(fwd60 −8.3%, t=−1.64, 60d 최악 셀), WF 일관 음수 — FDR 미달이나 필터(비중 0)로는 정당 |
| H3 감속 익절 (고멀티플×기울기 감속) | **부분 채택(bin 관점)** | 엄밀 감속 정의(극단 도달 후 5봉 감속)는 기각(delta t≈0.4). bin 관점: (2,1) 기울기 상위 이탈 시 fwd20 −3.5% vs (2,2) 유지 시 +6.2%(FDR 유의) |
| 정보 추가성 (멀티플 단독 vs 2D) | **채택** | 멀티플 단독은 약한 분리(최대 t=2.5). 2D는 같은 mult_bin 안에서 큰 분리 — mult0: 60d +28% vs −8%, mult2: 20d +6.2% vs −3.5% |

추가 발견: (2,2) 고멀티플×상승기울기 = 모멘텀 지속(fwd5 t=3.36·fwd20 t=2.67
모두 FDR 유의, WF 3/3 양수) — 고평가라고 팔면 안 되고 기울기가 살아있는 동안은
보유가 옳다.

## 4. 사이징 체인

```
state(t) = (mult_bin, angle_bin) ∈ 3×3   (expanding percentile, 인과)
  → target_weight = state_weights[mult_bin][angle_bin]   (config 행렬)
  → 점진 리밸런스 (threshold 0.05, 봉당 최대 0.25)
  → engine.portfolio_returns: 비중 1봉 지연 적용 + 회전율 비례 비용(왕복 20bp)
```

기본 `state_weights` (스터디 셀 부호·유의성에서 직접 유도, 스윕 아님):

| mult\angle | 0 (하락) | 1 (중립) | 2 (상승) |
| --- | --- | --- | --- |
| 0 (저) | 0.0 (칼 회피) | 1.0 (눌림목) | 1.0 (희소 셀 → 준용) |
| 1 (중) | 0.5 | 0.8 | 0.3 |
| 2 (고) | 0.0 (희소 셀 → 방어) | 0.3 (감속 축소) | 1.0 (모멘텀 지속) |

## 5. 백테스트 결과 (BTC/USDT 일봉, 수수료+슬리피지 왕복 20bp)

| phase | 전략 수익 | Sharpe | MDD | B&H 수익 | B&H Sharpe | B&H MDD | 평균노출 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all (2017-08~) | +1916% | 0.873 | −44.3% | +1347% | 0.654 | −83.2% | 0.47 |
| validation | +246% | 1.616 | −20.1% | +257% | 1.431 | −26.2% | 0.75 |
| test (holdout) | +15.7% | 0.391 | −21.4% | −2.1% | 0.160 | −53.0% | 0.44 |

전 phase에서 B&H 대비 위험조정(Sharpe·MDD) 우위. 절대수익도 all/test에서 상회.

## 6. 주의 / 무효화 조건

- **선택 편향 주의**: 상태 산출은 인과적(expanding)이지만, `state_weights` 셀
  선택은 전 기간 스터디 통계로 정했다 → validation/test가 완전한 OOS는 아니다.
  walk-forward 3사이클 부호 일관성으로 부분 보완. 신규 데이터 1년 누적 후 스터디
  재실행이 1차 재검증 게이트.
- H2·H3는 통계적으로 확정이 아니다(방향 일치 수준). 해당 셀 가중치(0.0/0.3)는
  방어적 선택이지 확증된 알파가 아니다.
- 희소 셀 (0,2)·(2,0)은 표본 <10 — 가중치는 프라이어일 뿐 데이터 근거 없음.
- **무효화**: 스터디 재실행에서 (0,1) 알파 부호 반전, 또는 holdout 연장 구간에서
  B&H 대비 Sharpe·MDD 동시 열위면 폐기.
- 파라미터(200/5/252/3분위)는 플랜의 고정값 — 과최적화 방지를 위해 1차에서 스윕
  금지. `live_eligible=False`.

## 7. 이력

- 2026-07-03: profile_plan.txt 기반 스터디 + 전략 등록 (이 문서).
- 2026-07-04: 2D 상태 필드 궤적 패널 추가 — `mult_pct`/`angle_pct`(연속 expanding
  percentile rank, `cycle_state.py`)로 각 봉의 위치를 점으로 찍고, 종가 zigzag
  (`state_field.move_threshold`, pivot 되돌림 기본 threshold/2)로 라벨한 큰
  상승/하락 leg를 초록/빨강 선으로 잇는다(extras `state_field` → 범용 `field_dots`
  패널). zigzag 라벨은 미래 봉을 쓰므로 시각화 전용(신호 사용 금지).
- 2026-07-04: **path_tilt 검증·구현** — 궤적 실험(§ reports/profile_sizing/
  state_field_trend_analysis.md)에서 나온 "셀 × 진입 경로" 가설을 사전 등록 후
  정식 검정(`yoon_profile_path_study.py`). PA(M1A1을 vel=mult_pct.diff(10) 부호로
  분리, up→1.0/down→0.5) **채택**: delta fwd20 t=2.32/fwd60 t=2.10, WF 3/3,
  holdout 수익 +15.7→+18.5%·Sharpe 0.391→0.439·MDD −21.4→−20.2%. PB(M2A1
  하강→0) **기각**(WF 1/3, 단일 사이클 의존). config `path_tilt` 블록(기본 off,
  base 불변). 무효화: 스터디 재실행 시 delta 부호 반전 또는 holdout 연장 열위.
- 2026-07-04: 필드 궤적 패널 노브 — 궤적 on/off 토글 + 파동 임계값 슬라이더.
  핸들러가 `state_field.knob_thresholds` 그리드(기본 10~50%)별 라벨을
  `move_dir_<pct>`/`move_id_<pct>` 컬럼으로 사전 계산해 담고, 패널은 선택만
  한다(UI 재계산 없음). 기본값은 패널 선언의 `move_threshold`.
