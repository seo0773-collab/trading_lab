# 볼린저밴드 평균회귀 크로싱 — 횡보종목 파라미터 스윕

- 유니버스: ADX(14) 최근 504봉 평균이 낮은 30종목 (유동성/가격 필터 통과)
- 백테스트: 최근 ~1260봉, train 70% / holdout 30% 시계열 분할, 왕복비용 50.2bp (KIS US 편도수수료+슬리피지 가정)
- offset은 인과적(>=0)만 스윕((0, 1, 2, 3, 5)). offset<0은 미래 밴드값을 과거봉에 대응시키는 것이라 실거래 불가(look-ahead)하여 제외.
- B&H 베이스라인: train Sharpe(median)=0.173, holdout Sharpe(median)=0.488

## Train 기준 상위 5개 조합 (median trades>=5)

| length | mult | offset | train Sharpe | train trades | holdout Sharpe | holdout trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 2.5 | 0 | 0.948 | 8 | 1.079 | 4 |
| 20 | 3.0 | 1 | 0.903 | 10 | 1.033 | 4 |
| 40 | 3.0 | 5 | 0.810 | 6 | 1.037 | 2 |
| 50 | 2.5 | 2 | 0.798 | 7 | 1.253 | 3 |
| 40 | 2.5 | 2 | 0.780 | 9 | 1.121 | 3 |

## 최적 조합 (train Sharpe 기준)

- length=20, mult=2.5, offset=0
- train median Sharpe=0.948 (B&H 0.173 대비 우위)
- holdout median Sharpe=1.079 (B&H 0.488 대비 우위)

## 주의

- 스크리닝 lookback(최근 2년)이 holdout 구간과 일부 겹친다 — 완전한 out-of-sample 종목선정은 아니다(현재 횡보 중인 종목을 고른 뒤 과거로 돌아가 검증하는 구조).
- 진입/청산 모두 당봉 종가 체결로 단순화(익일 시가 체결 지연 미반영).
- maType은 bb.txt 기본값 SMA로 고정(스윕 대상 아님).

## 파일

- 유니버스: `reports/bb_crossing_universe.csv`
- 전체 그리드: `reports/bb_crossing_sweep.json`
