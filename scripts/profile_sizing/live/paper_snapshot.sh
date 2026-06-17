#!/usr/bin/env bash
# 페이퍼 트레이딩 주간 스냅샷 — 로컬 실행용(클라우드는 Yahoo 403으로 불가).
# yoon1b·yoon1c 스냅샷을 적립하고 저널이 바뀌면 커밋·푸시한다.
# cron 예: 매주 금 22:00(머신 로컬시간)  0 22 * * 5  /경로/paper_snapshot.sh >> /경로/paper.log 2>&1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# venv 활성화(없으면 에러)
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: .venv 없음: $ROOT/.venv" >&2
  exit 1
fi
source .venv/bin/activate
export PYTHONPATH=src

echo "=== $(date '+%F %T %Z') 페이퍼 스냅샷 ==="
python scripts/profile_sizing/paper_trade.py --strategy yoon1b
python scripts/profile_sizing/paper_trade.py --strategy yoon1c

if git diff --quiet -- var/paper_trading; then
  echo "저널 변화 없음 — 커밋 생략."
  exit 0
fi
git add var/paper_trading/yoon1b_journal.jsonl var/paper_trading/yoon1c_journal.jsonl
git commit -m "paper trading 주간 스냅샷 (로컬 자동)"
git push origin "$(git rev-parse --abbrev-ref HEAD)"
echo "스냅샷 커밋·푸시 완료: $(git rev-parse --short HEAD)"
