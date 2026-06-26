# 게이트 체크리스트: heatmap2 (HVN 지지/저항 롱숏)

> 게이트는 명세(`specs/heatmap2.md`)에 고정한 합격선을 결과 확인 전에 잠그기 위한 것.
> 각 항목은 증거(테스트/리포트 경로)와 함께만 체크한다.

## Gate 0 — 구현·계약

- [x] 핸들러 `Heatmap2Handler`(Heatmap1Handler 상속, `_levels`/`_entry_reason` override).
      코어 `scripts/volume_profile.py`에 `profile_nodes`·`rolling_sr_levels` 추가.
- [x] heatmap1 레벨 산출을 `_levels()` 메서드로 추출(동작 불변 리팩토링).
- [x] StrategyArtifacts 계약 충족(forecast·trades·equity·metrics) — heatmap1 엔진 상속.
- [x] registry 등록 + config JSON(`heatmap2.json`, price_scale=log·long_only=false·node*).
- [x] `tests/test_strategy_contract.py`가 heatmap2 자동 포함·통과.
- [x] 전체 테스트 통과: `PYTHONPATH=src .venv/bin/python -m unittest discover tests -q` (121개).

## Gate 1 — 무누수·불변식

- [x] HVN/지지/저항 = t 이하 롤링 프로파일 + next_open 체결 → 룩어헤드 없음.
- [x] warmup·노드 없음 → val/vah NaN → 신호 0(불변). _targets는 반대편 노드 부재 시 POC 폴백.
- [x] heatmap1 동작 불변(`_levels` 추출 리팩토링이 기존 결과 안 바꿈) — 전체 테스트로 확인.
- [x] `profile_nodes` 단위 검증: 3봉우리 합성에서 피크 위치(17.5/50.8/84.2)·강도순·
      min_strength 필터(약한 노드 탈락) 확인.

## Gate 2 — 합성 드라이런

- [x] 합성 OHLCV로 계약 run `succeeded`. reversion 106 trades(롱63/숏43)·breakout 87,
      양방향·청산사유 다양(opposite/poc_target/va_stop/horizon).

## Gate 3 — 실데이터 검증

- [~] 1차 단건(AAPL/TSLA/KO/NVDA, phase=test): **부진**. va_reversion 승률 21~28%·
      대부분 큰 손실(PF 0.14~0.31), va_breakout 승률 52~59%이나 총수익 대체로 음수
      (PF~1.1, 비용 잠식). **주요인=거래 과다**(노드 터치마다 진입, KO 466건/test).
      쿨다운·강한노드 한정·min_hold↑·추세필터 없이는 비용에 자멸.
- [ ] node_top_n·min_strength·lookback 민감도, log vs linear, 쿨다운 추가 후 재검.
- [~] **상대가격 프로파일 환산 시도(2026-06-24, 사용자 아이디어)**: 코어
      `rolling_sr_levels(axis="relative")` + `_rolling_sr_relative` 추가(인과 상대위치
      0~1 공간 HVN→절대가격 역환산, `_inv_rel`/`_rel_scalar`/`_relative_histogram`).
      heatmap2 핸들러 `sr_axis` config(기본 absolute=불변). 역함수 정합·121테스트 OK.
      **결과: 거래과다는 극적 해결(KO 466→16)이나 cumulative 상대위치는 추세 상승종목
      에서 현재가가 항상 범위 상단(ref_r≈1)이라 지지/저항 신호 거의 0**(AAPL/NVDA
      0거래), 생긴 거래도 품질 나쁨(PF 0.02~0.03). 통찰: relative는 cumulative일 때만
      absolute와 달라지는데 그게 추세 쏠림을 유발. rolling-relative는 absolute log
      rolling과 거의 동치(새 가치 없음). → cumulative 상대위치 신호는 부적합. 코어
      옵션은 보존(기본 off).

## Gate 4 — 판정과 다음 단계

- [x] **반등 확인 필터 시도(2026-06-24)**: heatmap2 `_signals`에 `confirm_candle`
      (롱=양봉/숏=음봉)·`confirm_volume_mult`(터치봉 거래량≥rolling평균×mult) 옵션
      추가(기본 off=불변, 121테스트 OK). 거래량 필터가 거래과다(KO 467→21~93)와 손실
      폭·MDD를 **일관되게 줄임**(노이즈 제거 효과 실재). **그러나 reversion·breakout
      둘 다 필터 유무 무관하게 양의 엣지 없음**: breakout 평균 Sharpe(4종목) baseline
      −0.217 → +candle+vol1.5 −0.083(완화일 뿐 여전히 음수), reversion도 vol2.0에서
      거래 2~21건으로 통계신뢰 소실. NVDA +candle +102%처럼 종목별 들쭉날쭉 = 강건한
      엣지 아님.
- [x] **판정 = megacap 일봉 부적합**. HVN 지지/저항 신호 자체에 일봉 예측력이 없음
      (필터는 손실을 줄일 뿐 엣지를 만들지 못함). 다음 후보=자산군(크립토 인트라데이,
      heatmap 원래 타겟)·타임프레임(시간/분봉) 변경 또는 heatmap 계열 중단. 코드·필터
      옵션은 보존(기본 off).
