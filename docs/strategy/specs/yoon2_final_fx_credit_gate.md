# 전략 명세: yoon2_final_fx_credit_gate

> `yoon2_final_fx`의 통화 틸트 레이어는 그대로 두고, 위험 게이트의 신호원만
> 추세확인형(`yoon2_final_fx_2`, 기각)에서 신용스프레드 대리(HYG/IEF)로 바꾼
> 검증 후보.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon2_final_fx_credit_gate |
| 기반 전략 | yoon2_final_fx |
| 변경 범위 | `currency_allocation.risk_gate`의 신호원만 변경(틸트 레이어는 동일) |
| 하위 슬리브 | yoon1j_lp(VGT 포함 top3) + yoon1j_kr(top3), 둘 다 hold_buffer=2 |
| 리밸런싱 | 월간 |
| 기준 통화 | KRW, USD 슬리브는 USD/KRW 환산 |
| 핸들러 | `src/trading_lab/strategies/yoon1k.py` (`Yoon1kHandler`) |
| config | `configs/strategies/yoon2_final_fx_credit_gate.json` |

## 2. 배경 — 기존 위험 게이트(`yoon2_final_fx_2`)의 실패

`yoon2_final_fx_2`는 "한/미 동시 하락(63일 모멘텀<0 & 200일선 아래 & 드로다운
10%↓)" 조건으로 노출을 축소하는 추세확인형 게이트다. 22개 과거 드로다운
에피소드로 리드타임을 재보면 이 신호는 **8/22 에피소드에서만 발동하고, 발동해도
중앙값 +19거래일(사후)** — 즉 크래시가 상당히 진행된 뒤에야 켜져 MDD가 거의
개선되지 않았다(`reports/profile_sizing/yoon2_final_fx2_overlay_report.md`).

## 3. 신용스프레드 게이트 — 리드타임 진단

동일 22개 에피소드에 대해 VIX 텀스트럭처(VIX>VIX3M)와 신용스프레드(HYG/IEF)를
비교 진단했다(`scripts/profile_sizing/yoon2_final_fx_leading_signals.py`,
`reports/profile_sizing/yoon2_final_fx_leading_signals.md`).

| 신호 | 발동 에피소드 | peak 이전 선행 발동 | 중앙값 오프셋 |
| --- | ---: | ---: | ---: |
| 기존 추세확인 게이트 | 8/22 | 3건 | +19일(사후) |
| VIX 백워데이션 | 20/22 | 16건 | -37일(선행) |
| **신용스프레드(HYG/IEF)** | 18/22 | 14건 | **-48일(선행)** |

신용스프레드가 가장 이른 신호였고, 단순 오버레이 백테스트(발동 시 노출 0.5배)에서
CAGR은 거의 그대로인데 Sharpe가 개선되고 MDD가 얕아져(`current` 16.27%/
-17.13%/1.189 → 16.23%/-16.40%/1.212) 이 후보로 정식 검증을 진행했다.

## 4. 규칙

```
ratio = HYG.close / IEF.close
mom_10d = ratio.pct_change(10)
below_sma20 = ratio < ratio.rolling(20).mean()
raw_weak = (mom_10d < -0.02) & below_sma20
weak = raw_weak가 최근 2거래일 연속 참일 때만 참    # min_consecutive=2, 노이즈 필터

exposure = 0.5 if weak(직전 봉) else 1.0   # 신호는 1거래일 shift
w_USD_final = w_USD_tilt * exposure
w_KRW_final = w_KRW_tilt * exposure
cash = 1 - exposure
```

`w_USD_tilt`는 기존 `yoon2_final_fx`와 동일한 63일 상대모멘텀 틸트(40~60% 밴드)
이고, 이 게이트는 그 위에 노출 축소만 얹는다. `HYG`/`IEF` 가격이 없으면(합성
데이터 계약 테스트 등) 게이트는 자동으로 꺼지고 정적 비중으로 후퇴한다.

### 4-1. 노이즈 필터링(`min_consecutive`)

원안(`min_consecutive=1`, 즉 필터 없음)은 19년간 397거래일(9.7%)·103개의 별도
에피소드에서 발동하는데, 실질적으로 도움이 된 드로다운은 1~2건뿐이라 대부분이
며칠짜리 노이즈(휩쏘)였다. `min_consecutive=2`(원신호가 2거래일 연속 참이어야
게이트가 켜짐, 해제는 즉시)로 스윕한 결과(`scripts/profile_sizing/
yoon2_final_fx_credit_gate_denoise.py`):

| min_consecutive | CAGR | MDD | Sharpe | 발동일수 | 에피소드 수 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1(원안) | 16.23% | -16.40% | 1.212 | 397 | 103 |
| **2(채택)** | **16.60%** | **-16.40%** | **1.228** | 294 | 71 |

발동일수 -26%·에피소드 -31%로 노이즈를 줄이면서 MDD 방어(-16.40%, 2020-02
코로나 크래시 -8.84% 그대로 유지)는 손해 없이 CAGR·Sharpe가 오히려 개선됐다
(부트스트랩 vs current: P(CAGR>) 43.1%→71.1%, P(Sharpe>) 69.9%→85.2%). 3
이상은 MDD 방어력이 다시 무너지기 시작해(min_consecutive=5부터 MDD가 current와
동일한 -17.13%로 복귀) 2가 최적점으로 채택됐다.

### 4-2. 2차 개선 4종 검증(`scripts/profile_sizing/yoon2_final_fx_credit_gate_v2.py`)

`min_consecutive=2`를 기준선(v1)으로 두고 4가지 추가 개선을 실제
`Yoon1kHandler`로 실측했다.

| variant | CAGR | MDD | Sharpe | 판정 |
| --- | ---: | ---: | ---: | --- |
| v1(min_consecutive=2만) | 16.60% | -16.40% | 1.228 | 기준 |
| **v2 현금→IEF 대피(채택)** | **16.84%** | **-16.40%** | **1.242** | 채택 |
| v3 신용스프레드 AND VIX 백워데이션 | 16.34% | -16.40% | 1.202 | 기각 |
| v4 연속형(그레이디언트) 노출 축소 | 16.33~16.35% | -16.40% | 1.200~1.206 | 기각 |

- **IEF 대피(채택)**: 게이트 발동 시 자금을 현금(0% 수익)이 아니라 `risk_gate.
  safe_asset="IEF"`로 굴린다. 신용스프레드 신호 자체가 "국채가 상대적으로
  강해진다"는 뜻이라, 방어 국면에서 국채 강세분을 그대로 포착한다. MDD는
  -16.40%로 완전히 동일(국채 자체가 저변동이라 방어력 손상 없음)한데 CAGR·
  Sharpe가 뚜렷이 개선(부트스트랩 vs v1: P(Sharpe>) 79.4%, P(CAGR>) 82.0%,
  P(MDD shallower) 63.1%). 최종 채택 파라미터.
- **VIX AND 결합(기각)**: 신용스프레드에 VIX 백워데이션(VIX>VIX3M & VIX>18)
  동시 확인을 추가하면 오히려 악화된다(Sharpe 1.228→1.202, 부트스트랩 vs v1
  P(Sharpe>) 12.5%). 두 신호원의 리드타임이 달라(신용 중앙값 -48일, VIX -37일)
  AND 결합이 더 느린/덜 겹치는 신호로 수렴한 것으로 보인다.
- **연속형 노출(기각)**: 이진 off_scale(1.0 또는 0.5) 대신 신호 강도(threshold
  대비 breach 정도)에 비례해 노출을 연속적으로 깎는 방식을 severity_cap
  1.5/2.0/3.0으로 테스트했으나 전부 이진 방식보다 못하다(Sharpe 1.20~1.21).
  부분적으로만 노출을 줄이는 게 오히려 "애매하게 계속 물려있는" 상태를
  만들어 손해로 작용한 것으로 해석된다.
- **off_scale 재그리드**: `min_consecutive=2` 고정 상태에서 lookback/sma/
  threshold/off_scale을 다시 스윕한 결과 `L10_S30_T-0.01_O0.3`(CAGR 17.08%/
  MDD -14.96%/Sharpe 1.328)가 풀샘플 1위였으나, **아직 워크포워드/부트스트랩
  검증 전**이라 밴드 30/70 사례처럼 과적합 가능성이 있어 채택하지 않았다(후속
  검증 대상으로 남겨둠).

최종 채택 파라미터는 `min_consecutive=2` + `safe_asset="IEF"`(그 외는
lookback=10/sma=20/threshold=-2%/off_scale=0.5 유지).

## 5. 전체 검증 결과 (`scripts/profile_sizing/yoon2_final_fx_credit_gate_validate.py`)

- 그리드(240조합: lookback 5/10/15/20/30 × sma 10/20/30/50 × threshold -1%/-2%/
  -3%/-5% × off_scale 0.3/0.5/0.7): current 대비 **MDD만 개선 220/240(92%)**,
  Sharpe·MDD 동시 개선 120/240(50%).
- 원안(lookback=10, sma=20, threshold=-2%, off_scale=0.5): CAGR 16.23%
  (current 16.27%, 거의 동일), MDD -16.40%(current -17.13%), Sharpe 1.212
  (current 1.189).
- 워크포워드(14폴드, 고정 파라미터): Sharpe 승 8/14, MDD 승 10/14.
- 롤링 3년(16구간): Sharpe·CAGR·MDD 전부 9/16.
- 페어드 부트스트랩(8000회): P(Sharpe 개선) 69.5%, P(MDD 더 얕음) 76.4%,
  P(CAGR 개선) 43.4%(사실상 대등).
- 비용 민감도: top-level 리밸런스 비용이 커질수록(0→50bps) current보다 Sharpe가
  더 빨리 깎인다(게이트로 인한 추가 회전 때문).

## 6. 한계 — 표본 집중

드로다운 에피소드 22건 중 이 게이트의 개선분은 **2020-02 코로나 크래시 한 건에
집중**된다(-17.13% → -8.84%, 거의 절반). 나머지 21건은 거의 변화가 없고,
2022년 완만한 약세장(-9.53% → -9.86%)에서는 오히려 소폭 나빠진다. 즉 "매번
조금씩 방어"가 아니라 "평소엔 거의 공짜로 대기하다가 급성 신용경색형 크래시에서
크게 막아주는" 보험형 오버레이로 해석해야 하며, 19년 표본에 해당 유형 사건이
1건뿐이라 통계적 확신은 제한적이다.

## 7. 의도와 폐기 기준

- 의도: 통화 틸트 레이어의 종목선정·배분 로직은 건드리지 않고, 위험 게이트의
  신호원만 선행성 있는 신호로 교체해 MDD 방어를 실질화한다.
- 폐기 기준: 페이퍼 검증 또는 추가 표본에서 신용경색형 크래시가 아닌 구간에서
  CAGR 드래그가 누적되어 Sharpe가 `yoon2_final_fx` 대비 유의하게 낮아지면 폐기.

## 8. 주의

- 아직 검증 후보이며 `live_eligible=False`다.
- `risk_gate.source="credit_spread"`, `symbols=["HYG","IEF"]`는 `Yoon1kHandler`
  의 `load_data`/`_relative_momentum_fx2_weights`에 일반화된 옵션으로 추가했다
  (다른 `kind`/게이트 미사용 전략에는 영향 없음, `raw["__GATE__"]` 부재 시 자동
  비활성).
