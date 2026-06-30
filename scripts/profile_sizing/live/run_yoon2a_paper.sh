#!/usr/bin/env bash
# yoon2a 모의 전진검증 1회 사이클 (로컬 시뮬 원장 방식, KIS 미사용).
# KIS 모의계좌가 원/달러 예수금 풀을 고정·불균형으로 선지급해 통합 NAV가 무의미한
# 문제(2026-06-29) 때문에 브로커 대신 고정 시드 원장으로 전진 NAV를 계산한다.
# 시장 시간/영업일 무관(일봉 종가 기준)이라 월 1회 단일 실행으로 충분.
#
#   scripts/profile_sizing/live/run_yoon2a_paper.sh
#
# (KIS 실계좌 경로는 yoon1k_kis_rebalance.py 로 별도 보존 — 게이트 통과 후 사용.)
# 로그: var/live/yoon2a_paper.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv/bin/python"
mkdir -p var/live
LOG="var/live/yoon2a_paper.log"

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== yoon2a 전진 사이클 시작 (로컬 원장) ==="

# 1) 목표 갱신 (3슬리브, period=max → 시작점 무관)
for s in yoon1j_lp yoon1j_kr sso_bh; do
  log "목표 갱신: $s"
  "$PY" scripts/profile_sizing/paper_trade.py --strategy "$s" >>"$LOG" 2>&1
done

# 2) 원장에 이번 달 목표 기록(월 1건, 버퍼 보존 결합 비중)
log "원장 업데이트: paper_ledger update"
"$PY" scripts/profile_sizing/live/paper_ledger.py update >>"$LOG" 2>&1

# 3) 전진 NAV 리포트(로그에 50/50 벤치 대비 기록)
log "전진 NAV: paper_ledger report"
"$PY" scripts/profile_sizing/live/paper_ledger.py report >>"$LOG" 2>&1 \
  || log "report 건너뜀(데이터 부족 등)"

log "=== 완료 ==="
