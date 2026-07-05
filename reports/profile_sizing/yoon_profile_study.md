# yoon_profile 사전 검증: 멀티플 × 기울기 2D 상태공간 (2026-07-03)

- 데이터: BTC/USDT 일봉 2017-08-17~2026-07-03 (binance, 3,243봉 · 갭 0)
- 유효표본: 2,532봉 (SMA200 + z252 + expanding bin 252 워밍업 이후 ≈ 2019-07~)
- 파라미터(플랜 고정): SMA 200 / slope lookback 5 / z 252 / 3분위 그리드 / 셀 최소 50표본
- 방법: expanding percentile bin(무누수) · forward 5/20/60d · Newey-West(lag=h) t
  · Bonferroni + BH-FDR(27검정) · walk-forward 3사이클
- 산출물: `yoon_profile_study.json` (전체 수치) · `yoon_profile_cell_stats.csv`
  · `yoon_profile_heatmaps.png` (셀 평균수익/표본밀도)
- 재현: `PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/yoon_profile_study.py`

## 셀 요약 (fwd60, mean %, FDR 유의 굵게)

| mult\angle | 0 (하락) | 1 (중립) | 2 (상승) |
| --- | --- | --- | --- |
| 0 (저) | −8.3 (n=431, t=−1.64) | **+28.2** (n=160, t=4.79) | n=7 판정 제외 |
| 1 (중) | **+21.2** (n=137, t=2.58) | **+13.3** (n=410, t=2.83) | +0.3 (n=260) |
| 2 (고) | n=5 판정 제외 | −3.7 (n=300) | +17.6 (n=767, t=1.76) |

fwd20에서는 (2,2) +6.2%·(1,1) +5.4%가 FDR 유의, fwd5에서는 (2,2) +1.6%(t=3.36)만 유의.

## 가설 판정

| 가설 | 판정 | 근거 |
| --- | --- | --- |
| H1 눌림목 알파 | **변형 채택** | 엄밀 셀 (0,2)=표본 7(저멀티플과 상승기울기 상위분위는 구조적으로 거의 공존 불가). 알파의 실체는 (0,1) "하락이 멈춘 저가권": fwd60 +28.2% FDR 유의, WF 3사이클 전부 양수(+9.7/+0.2/+6.8%) |
| H2 떨어지는 칼 | **보류(방향 일치)** | (0,0) 전 horizon 음수·60d 최악 셀(t=−1.64)·WF 일관 음수(−3.7/−1.2%). FDR 미달 — 필터로는 정당, 확증 알파는 아님 |
| H3 감속 익절 | **부분 채택(bin 관점)** | 엄밀 감속 정의는 기각(t≈0.4). bin 관점: 고멀티플에서 기울기 상위 유지(2,2) +6.2% vs 이탈(2,1) −3.5% (fwd20) |
| 2D 정보 추가성 | **채택** | 멀티플 단독 최대 t=2.5, 셀 간 분리 약함. 2D는 mult0 안에서 60d 36pp, mult2 안에서 20d 9.6pp 분리 |

주의: 다중비교 보정 후 유의 셀은 27검정 중 6개. mid-multiple의 angle 상−하 delta는
음수(60d t=−2.15) — "기울기 높을수록 좋다"가 아니라 셀별 비선형 구조다.

## 결론 → 전략화

셀 부호·유의성을 그대로 `state_weights` 행렬로 옮겨 `yoon_profile` 전략 등록
(스펙: `docs/strategy/specs/yoon_profile.md`). 파라미터 스윕 없음(플랜 §주의사항).
백테스트(왕복 20bp): all Sharpe 0.873/MDD −44% vs B&H 0.654/−83%,
test(holdout) +15.7%/Sharpe 0.391 vs B&H −2.1%/0.160.

한계: 셀 가중치 선정이 전 기간 통계 기반 → val/test 완전 OOS 아님(WF 부호
일관성으로 부분 보완). 1년 데이터 누적 후 스터디 재실행이 재검증 게이트.
