# 전략 명세: yoon4 — Global Capital Flow Regime Engine

> yoon2(미국/한국 50:50 고정 통화 리밸런싱)를 한 단계 위로 일반화한 전략.
> 경제를 예측하지 않고 **글로벌 자금 이동을 추적해 투자 국가를 동적으로 선택**한다.
> 설계 출처: `new_layer.txt`.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon4 |
| 구조 | 글로벌 레짐 게이트 → CRI 국가선택 → (리프) 국가 ETF 직접 보유 |
| config | configs/strategies/yoon4.json |
| 핸들러 | src/trading_lab/strategies/yoon4.py (`Yoon4Handler`) |
| 기준 통화 | KRW (전 종목 USD 상장 → 표시용 1회 환산) |
| category | hierarchical |

## 2. 계층 구조

```
[최상위] 글로벌 Risk-On/Off 레짐 게이트 (총 주식노출 조절)
   └─ 국가 선택: Capital Rotation Index(CRI) 상위 N개 국가 ETF 비중 배분
        └─ 리프(v1): 선택 국가 ETF 직접 보유  (섹터 계층은 후속 레이어)
```

yoon2의 고정 2국가(미국·한국)를 **다국가 동적 로테이션**으로 일반화했다. 국가 ETF가
전부 미국 상장(USD)이라 yoon2의 다통화 리밸런싱 복잡도가 사라지고, 원화 환산은
포트폴리오 NAV에 단일 환율(`KRW=X`)을 1회 적용하는 것으로 끝난다.

## 3. Capital Rotation Index (CRI)

매 리밸런스 시점에 국가 간 횡단면 **Z-score**로 정규화한 뒤 가중합한다
(`cri.weights`로 분리 → 백테스트 최적화 가능).

| 팩터 | 정의 | 의미 |
| --- | --- | --- |
| mom_3m / 6m / 12m | 가격 모멘텀(63/126/252봉) | 추세 강도 |
| rs | SPY 대비 비율(ratio) 모멘텀 | 상대강도(자금 쏠림) |
| hi_dist | 52주 신고가와의 거리 | 추세 지속/신고가 근접 |
| vol_trend | 단기/장기 거래량 비율 | 거래량 유입 |
| vol_contraction | 단기/장기 실현변동성 비율(부호 반전) | 변동성 축소 |
| acceleration | mom_3m의 차분(가속도) | 신규 자금 유입 탐지 |

> 통화 강도(USDJPY 등)는 **USD 표시 ETF 수익률에 이미 반영**(로컬통화↑→ETF가격↑)되므로
> 별도 FX 팩터를 두지 않는다.

CRI 상위 `top_n` 국가를 선택하고, 양(+)의 CRI에 비례해 비중을 준다(전부 음수면 균등).

## 4. 글로벌 레짐 게이트 (Risk-On/Off)

`regime.enabled`일 때 다음 신호의 OFF 비율로 총 주식노출을 `off_scale`까지 축소한다.

- SPY < 200MA (시장 약세)
- VIX > `vix_high` (고변동)
- DXY > 100MA (달러 강세 = 신흥국 자금 이탈)

주식에 안 쓴 spare는 **안전자산 슬리브**(TLT/GLD, 자기 100MA ON일 때만)로 회전하고,
나머지는 현금. 안전자산 총비중은 `safe_sleeve.max`로 제한한다.

## 5. 리밸런싱/비용

월간(`rebalance_freq`) 리밸런스. 비중 변화가 `min_weight_change` 미만이면 거래를 생략해
턴오버를 억제하고, 거래 회전율 × (`fee_bps_per_side`+`slippage_bps`)를 비용으로 차감한다.

## 6. 한국투자증권(KIS) 거래 가능성

전 종목 미국 상장 → KIS 해외주식(`kis_client.py` 거래소 코드 NASD/NYSE/AMEX)으로 매매 가능.
NYSE Arca 상장 ETF는 KIS에서 AMEX(AMS) 코드로 거래된다.

| 심볼 | 국가/역할 | KIS 거래소 |
| --- | --- | --- |
| SPY | 미국 | AMEX |
| EWJ/EWY/EWG/EWT/EWW/EWZ/EWU/EWC | 일본/한국/독일/대만/멕시코/브라질/영국/캐나다 | AMEX |
| INDA | 인도 | AMEX |
| MCHI | 중국 | NASD |
| TLT | 안전(장기채) | NASD |
| GLD | 안전(금) | AMEX |

FX(USD/KRW)는 환산·표시용일 뿐 거래 대상이 아니다(KIS는 USD로 보유). `live_eligible=False`로
시작하며, paper/dryrun 연동은 후속 작업.

## 7. 비목표 (후속 레이어)

- 국가 → 섹터 → 리프(yoon1j) 3계층 풀스택 (new_layer #17~18): 국가별 섹터 ETF 데이터 필요.
- CRI 가중치 자동 최적화 스윕/리포트 (new_layer #13, #20).
