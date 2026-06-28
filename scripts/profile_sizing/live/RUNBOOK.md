# yoon1k 실거래 운영 런북

검증된 전략은 `yoon1k`(원본). 실매매 프록시 `yoon1k_lp`(저가 ETF)는 원본과 상관 0.991로
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

## 게이트 상태 (2026-06-28)

- A1 프록시 검증 ✅ / A2 잔고파싱 안전화 ✅(KR 필드 `dnca_tot_amt` 실측 확인) /
  A3 통화 리밸런싱(50/50 ±5%p) ✅ / 시간대·세후 기대치 보정 ✅ /
  신선도 가드·체결 확인·세션 안내 ✅
- **남은 것**: 해외(USD) 잔고 필드 실측(`KIS_MOCK_OVS_ACCOUNT` 설정 후 probe `--overseas`),
  해외 주문 TR_ID 소액 모의 검증, 실계좌 전환(`confirm_real` 배선)은 모의 충분 검증 후.
- ⚠️ 실서버 주문 TR_ID·해외 잔고 필드는 아직 미검증 → **소액 모의로 먼저 확인**.
