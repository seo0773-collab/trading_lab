# yoon2 한국 산업 ETF 슬리브 임시 테스트

전략 파일/registry는 변경하지 않고, yoon2의 한국 슬리브(`yoon1j_kr`)만 임시 유니버스로 바꿔 계산했다. 미국 슬리브는 별도 표기 없는 경우 현재 `yoon1j_lp` 그대로다. 한미 50/50 월간 리밸런싱과 USD/KRW 환산은 유지했다.

## 가정

- 초기 투자원금: 2,000,000원
- 월 적립금: 200,000원
- 비교 기간: 2015-10-07 ~ 2026-06-26
- SPY B&H는 SPY 달러가격에 USD/KRW 환율을 곱한 원화 기준이다.

## 결과

| 변형 | 미국 종목수 | 한국 종목수 | KR top_k | NAV CAGR | NAV MDD | Sharpe | 적립식 최종 | 적립식 수익률 | 적립식 MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kr_semibattery_core_top5 | 11 | 7 | 5 | 21.58% | -20.56% | 1.56 | 119,369,006원 | 329.38% | -19.55% |
| kr_tiger_sector_plus_top5 | 11 | 18 | 5 | 16.88% | -18.04% | 1.39 | 91,877,322원 | 230.49% | -16.84% |
| SPY_KRW_BH | 1 | 0 | 0 | 19.29% | -30.15% | 0.95 | 91,594,778원 | 229.48% | -29.38% |
| us_industry_plus_kr_industry_plus | 31 | 25 | 5 | 16.27% | -23.40% | 1.23 | 86,691,321원 | 211.84% | -22.15% |
| baseline | 11 | 8 | 5 | 16.20% | -14.46% | 1.40 | 86,295,172원 | 210.41% | -13.23% |
| kr_tiger_sector_plus_top8 | 11 | 18 | 8 | 15.41% | -15.94% | 1.35 | 82,512,677원 | 196.81% | -14.69% |
| kr_industry_theme_recent_top8 | 11 | 28 | 8 | 14.60% | -17.64% | 1.27 | 78,561,058원 | 182.59% | -16.41% |
| kr_industry_theme_plus_top5 | 11 | 25 | 5 | 14.12% | -20.60% | 1.17 | 77,378,176원 | 178.34% | -19.39% |
| kr_industry_theme_plus_top8 | 11 | 25 | 8 | 14.22% | -17.64% | 1.24 | 76,084,333원 | 173.68% | -16.41% |
| kr_theme_core_top8 | 11 | 17 | 8 | 11.21% | -16.19% | 1.04 | 59,973,657원 | 115.73% | -14.95% |
| kr_theme_core_top5 | 11 | 17 | 5 | 10.07% | -18.62% | 0.91 | 56,885,485원 | 104.62% | -17.39% |

## 후보 구성

- `kr_tiger_sector_plus`: 현재 KODEX 업종 ETF에 TIGER 200 업종/저가 대체 후보를 추가
- `kr_industry_theme_plus`: 현재 한국 업종 ETF에 화장품, 여행레저, 미디어컨텐츠, 바이오, 2차전지, 반도체, BBIG/K-뉴딜 후보를 추가
- `kr_industry_theme_recent`: 위 후보에 로봇/AI, 우주방산처럼 히스토리가 짧은 최근 ETF까지 추가
- `kr_semibattery_core`: 반도체+2차전지 중심 집중형
- `kr_theme_core`: 테마/산업 ETF만으로 한국 슬리브를 구성
- `us_industry_plus_kr_industry_plus`: 이전 미국 산업 ETF 확장 후보와 한국 산업 ETF 확장 후보를 동시에 적용

## 판정

- `kr_tiger_sector_plus_top5`: baseline 대비 적립식 최종 5,582,150원, MDD 차이 -3.61%p
- `kr_industry_theme_plus_top5`: baseline 대비 적립식 최종 -8,916,996원, MDD 차이 -6.17%p
- `kr_industry_theme_plus_top8`: baseline 대비 적립식 최종 -10,210,839원, MDD 차이 -3.19%p
- `kr_semibattery_core_top5`: baseline 대비 적립식 최종 33,073,834원, MDD 차이 -6.32%p
- `kr_theme_core_top5`: baseline 대비 적립식 최종 -29,409,687원, MDD 차이 -4.16%p
- `us_industry_plus_kr_industry_plus`: baseline 대비 적립식 최종 396,149원, MDD 차이 -8.92%p

한국 슬리브는 이미 업종 ETF 중심이어서 단순 TIGER 업종 추가보다, 세분화 산업/테마 후보를 더해 랭킹 엔진이 선택하게 하는 쪽이 더 의미 있다. 단, 한국 테마 ETF는 상장 이력이 짧은 종목이 많아 공통기간 편향과 과최적화 위험이 미국보다 크다.

## 파일

- 후보 CSV: `reports/profile_sizing/yoon2_kr_industry_candidates.csv`
- 결과 CSV: `reports/profile_sizing/yoon2_kr_industry_etf_probe.csv`
- 시계열 CSV: `reports/profile_sizing/yoon2_kr_industry_etf_probe_series.csv`
- 그래프 PNG: `reports/profile_sizing/yoon2_kr_industry_etf_probe.png`

## 정수주 실행 레이어 추가 확인

초기 200만원 + 월 20만원 적립, 한미 50/50, 미국 `$50`/한국 `50,000원` 최소 주문금액, 정수주 제약을 적용했다. 변형별 ETF 상장 이력 차이 때문에 시작일과 누적원금이 다를 수 있다.

| 변형 | 시작일 | 누적원금 | 최종평가액 | 수익률 | MDD | 평균 재현율 | 최저 재현율 | 주당가격 제약 상위 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kr_tiger_sector_plus_top5_integer | 2015-02-05 | 29,400,000원 | 101,284,926원 | 244.51% | -15.37% | 97.5% | 76.1% | KR:091170.KS 19, US:FCOM 16, KR:102970.KS 15, US:FENY 15, US:FSTA 14, US:FIDU 14, US:XLB 14 |
| baseline_integer | 2015-02-05 | 29,400,000원 | 95,333,389원 | 224.26% | -12.06% | 97.5% | 76.7% | KR:102970.KS 25, KR:091170.KS 24, US:FREL 22, US:FENY 22, US:FCOM 20, US:FSTA 16, US:XLB 16 |
| kr_semibattery_core_top5_integer | 2021-08-10 | 13,800,000원 | 35,389,473원 | 156.45% | -12.53% | 94.0% | 70.3% | US:FSTA 13, US:XLB 11, KR:305540.KS 8, US:FREL 8, US:XLF 8, US:FCOM 8, US:IYC 7 |
| SPY_KRW_BH_integer_window | 2021-08-10 | 13,800,000원 | 25,842,049원 | 87.26% | -15.61% |  |  |  |

- 정수주 결과 CSV: `reports/profile_sizing/yoon2_kr_industry_etf_integer_probe.csv`

## 장기 히스토리 반도체+2차전지 보수 후보

최근 상장 반도체 ETF(`395160.KS`, `396500.KS`)를 제외하고 2018년부터 데이터가 있는 반도체/2차전지 후보만 추가로 확인했다.

| 변형 | 시작일 | 누적원금 | 최종평가액 | 수익률 | 계좌 MDD | NAV CAGR | NAV MDD | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kr_semibattery_2018_core_top4 | 2007-01-29 | 43,800,000원 | 199,176,686원 | 354.74% | -20.21% | 14.46% | -20.56% | 1.11 |
| kr_current_plus_semibattery_2018_top5 | 2007-01-29 | 43,800,000원 | 194,901,693원 | 344.98% | -19.29% | 14.47% | -19.68% | 1.15 |
| baseline | 2007-01-29 | 43,800,000원 | 184,273,161원 | 320.71% | -14.04% | 13.99% | -14.46% | 1.14 |

- 보수 후보 CSV: `reports/profile_sizing/yoon2_kr_semibattery_2018_probe.csv`
