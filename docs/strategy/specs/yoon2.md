# 전략 명세: yoon2

> `yoon1k_lp`를 `yoon2`로 승격한 2단계 계층 전략. 기존 단일종목 Kalman MACD
> `yoon2` 계열은 삭제되었고, 앞으로 `yoon2`는 원/달러 2단계 fund-of-funds 전략을 뜻한다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon2 |
| 구조 | 미국 저가형 슬리브 + 한국 슬리브를 원화 기준 50/50 월간 리밸런싱 |
| config | configs/strategies/yoon2.json |
| 핸들러 | src/trading_lab/strategies/yoon1k.py (`Yoon1kHandler`) |
| 기준 통화 | KRW |

## 2. 2단계 구조

```
[상위] 원/달러 50/50 월간 리밸런싱
   ├─ USD 슬리브 → yoon1j_lp
   └─ KRW 슬리브 → yoon1j_kr
```

- `yoon1j_lp`: 미국 저가 ETF 기반 슬리브. 소액 정수주 모의/실거래 적합도를 높인다.
- `yoon1j_kr`: 한국 ETF 기반 원화 슬리브.
- USD 슬리브는 USD/KRW(`KRW=X`) 환율을 반영해 원화 기준 NAV로 결합한다.

## 3. 판정

`yoon2`는 단일 종목 타이밍 전략이 아니다. 방어형 하위 전략 2개를 통화별로 운용하고,
상위에서 통화 배분을 리밸런싱하는 2단계 전략이다.
