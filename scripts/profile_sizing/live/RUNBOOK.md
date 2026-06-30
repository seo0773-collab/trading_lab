# yoon1k 실거래 운영 런북

검증된 전략은 `yoon1k`(원본). 실매매 프록시 `yoon2`(구 `yoon1k_lp`, 저가 ETF)는 원본과 상관 0.991로
충실(A1 검증). **현실 기대치**: CAGR ~11%, MDD ~-15%, Sharpe **~1.04**(시간대 보정 1.06 −
배당세). 양도세는 계좌유형 의존(ISA/연금·기본공제로 완화). 강세장 절대수익은 시장 하회(방어형).

## 시간대별 실행 (중요)

국장과 미장은 거래시간이 다르므로 **시장별로 분리 실행**한다(주문가가 일봉 종가라,
해당 시장 정규장 시간에 돌려야 stale·미체결을 줄인다). DST는 스크립트가 자동 안내한다.

- 한국 슬리브: **평일 15:00 KST 경**(마감 직전)
- 미국 슬리브: **여름(DST) 22:30 / 겨울 23:30 KST 경**(미국 개장 직후)

리밸런스는 월간이라 매일 돌릴 필요 없다. **매월 첫 영업일** 1회면 충분(월말 신호 반영).
먼저 `paper_trade.py`로 목표를 갱신(신선도 가드 기본 7일)한 뒤 리밸런스를 실행한다.

### crontab 예시 (로컬, 매월 1일)

```cron
# 목표 갱신(두 슬리브) — 매월 1일 08:00 KST
0 8 1 * * cd ~/trading_lab && PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_lp >> var/live/paper.log 2>&1
5 8 1 * * cd ~/trading_lab && PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_kr >> var/live/paper.log 2>&1
# 한국 슬리브 주문 — 매월 1일 15:00 KST (드라이런; --execute 로 제출)
0 15 1 * * cd ~/trading_lab && PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py >> var/live/rebal_kr.log 2>&1
# 미국 슬리브는 KST 밤 — DST 분기는 스크립트 헤더의 '권장 실행창' 참고(여름 22:30/겨울 23:30)
```

> cron은 DST를 모른다. 미국 다리는 여름/겨울 두 줄(22:30·23:30)을 등록하고 한쪽은
> 실행 시 시장 닫힘으로 무해 처리하거나, 스크립트가 출력하는 `권장 실행창`을 보고 월 1회
> 수동 실행해도 된다(월간이라 부담 적음).

## 명령 모음

```bash
# 1) 목표 갱신
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_lp
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_kr

# 2) 드라이런(기본=통합증거금 50/50 복원, ±5%p 밴드, 신선도 가드 7일)
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py

# 3) 모의 제출 + 체결 사후확인
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py --execute --verify

# 잔고 필드 점검(읽기 전용)
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_field_probe.py
```

## yoon2a (공격형 파생) — 모의 전진검증 전용

`yoon2a` = yoon2 코어 74%(내부 yoon1j_lp/yoon1j_kr 50:50) + **SSO(2x S&P500) 26%**.
live 레이어에서는 3개 슬리브로 평탄화한다(yoon2a.json의 `sub_strategies`): `yoon1j_lp`
0.37(USD) / `yoon1j_kr` 0.37(KRW) / `sso_bh` 0.26(USD). SSO는 손픽 섹터(QLD/USD)가 아닌
광의 시장 레버리지라 생존편향을 피한 변형이다. SSO 비중은 Strict+Sharpe 재최적화
(2026-06-29)로 35%→26% 변경(B&H를 MDD·Sharpe 양 축에서 동시 지배).

> **게이트**: `live_eligible=False`. 실계좌(real) 주문은 스크립트에서 차단(모의 전용).
> OOS·실거래 검증 전까지 모의 전진검증만 수행한다. 방어형 yoon2보다 공격적이나, 26%
> 비중에선 백테스트 MDD ~-26%(B&H -31%보다 오히려 얕음).

```bash
# 1) 목표 갱신(3슬리브) — sso_bh 는 buy&hold 라 항상 {SSO:1.0}, data_through 만 갱신
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_lp
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy yoon1j_kr
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_trade.py --strategy sso_bh

# 2) 드라이런(통화 목표비중 복원: USD 63%/KRW 37%, ±5%p 밴드, 신선도 가드 7일)
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py --strategy yoon2a

# 3) 모의 제출 + 체결 사후확인
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py --strategy yoon2a --execute --verify
```

> 세션창: yoon1j_lp·sso_bh(미국)는 KST 밤(여름 22:30/겨울 23:30), yoon1j_kr(한국)는
> 15:00 KST 경. 월간이라 매월 첫 영업일 1회면 충분(SSO 슬리브도 동일 창에서 함께 처리).

### 스케줄러(cron) — 로컬 시뮬 원장 (채택 방식)

**왜 로컬 원장인가**: KIS 모의계좌는 원/달러 예수금 풀을 **고정·불균형**(USD풀이 의도
시드의 ~24배)으로 선지급해(2026-06-29 확인, 사용자 확인=시드 변경 불가), 통합 리밸런싱
NAV가 무의미하다(통화비중 1498% → 허깨비 주문). 그래서 브로커를 거치지 않고 **고정
시드(1천만원)** 로 목표 저널을 시장가에 마크투마켓하는 로컬 원장으로 전진 NAV를 계산한다.
일봉 기준이라 **시장 시간/영업일 무관 → 월 1회 단일 실행**으로 충분.

```cron
# TZ=UTC. 매월 1일 06:00 UTC. 목표갱신 3슬리브 → 원장 update → NAV report.
0 6 1 * * cd ~/trading_lab && scripts/profile_sizing/live/run_yoon2a_paper.sh
```

- 래퍼 `run_yoon2a_paper.sh`: paper_trade ×3(저널 갱신) → `paper_ledger update`(월 목표
  기록, 버퍼 보존 결합비중) → `paper_ledger report`(50/50 벤치 대비 NAV).
- 수동 확인: `PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/paper_ledger.py report`
- 원장: `var/live/yoon2a_ledger.json` · 로그: `var/live/yoon2a_paper.log`

> 시작점 무관(누적 percentile ~4~5년 포화, period=max라 충족) → 매 실행 목표가 백테스트
> 최신과 일치. **한계**: 결정론적 시뮬(슬리피지·체결오차 없음=낙관 상한), 실체결 아님.

### (보존) KIS 실계좌 경로 — 게이트 통과 후

KIS 주문 경로(`yoon1k_kis_rebalance.py`)는 **실계좌용으로 보존**한다. 시장 시간이
겹치지 않아 `--only-currency`로 KR창(02:00 UTC)·US창(15:00 UTC) 분리 제출하고,
`--seed-krw`(래퍼 `SEED_KRW`) 안전가드가 감지 자본>시드×1.5면 주문을 전면 중단한다
(허깨비 풀·예외 방지). `live_eligible=True` 승격 + 실주문 TR_ID 검증 후에만 사용.

## 모의투자 전진(forward) 경향성 검증 셋업

목표: 백테스트 경향(위험조정 우위 + 낙폭 방어)이 **모의 실행**에서도 유지되는지 전진 측정.

흐름: `paper_trade`(목표) → `yoon1k_kis_rebalance --execute`(모의 체결) →
`kis_track snapshot`(주기 적립) → `kis_track report`(모의 NAV vs 50/50 벤치 낙폭/수익).

```bash
# 주기 적립(주 1회 권장) + 리포트
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_track.py snapshot
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_track.py report
```

### ⚠️ 선행 조건 — 모의 주문 검증

2026-06-28 점검: 현재 `KIS_MOCK_ACCOUNT`로 국내/해외 모의 잔고 조회가 가능하고,
해외 모의 주문 엔드포인트도 계좌 불가 오류 없이 응답한다. 일요일 테스트에서는
`msg_cd 40100000 "모의투자 영업일이 아닙니다"`가 반환되었으므로, 실제 접수 여부는
영업일/장 시간에 소액 모의 주문으로 다시 확인한다.

확인할 것:

- **국내주식 모의투자 참가 신청**(상시 모의투자)과 계좌 기간이 활성 상태인지.
- `KIS_MOCK_ACCOUNT` 가 **모의 앱키(`KIS_MOCK_APP_KEY`)에 바인딩된 계좌**와 일치하는지.
- 해외 주문은 `KIS_MOCK_ACCOUNT`를 그대로 사용한다. 별도 `KIS_MOCK_OVS_ACCOUNT` 경로는 쓰지 않는다.

## 게이트 상태 (2026-06-28)

- A1 프록시 검증 ✅ / A2 잔고파싱 안전화 ✅(KR 필드 `dnca_tot_amt` 실측 확인) /
  A3 통화 리밸런싱(50/50 ±5%p) ✅ / 시간대·세후 기대치 보정 ✅ /
  신선도 가드·체결 확인·세션 안내 ✅
- **남은 것**: 해외 주문 TR_ID 소액 모의 검증(영업일/장 시간),
  실계좌 전환(`confirm_real` 배선)은 모의 충분 검증 후.
- ⚠️ 실서버 주문 TR_ID·해외 잔고 필드는 아직 미검증 → **소액 모의로 먼저 확인**.
