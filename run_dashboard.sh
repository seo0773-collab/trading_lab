#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: virtual environment not found: $VENV_ACTIVATE" >&2
    exit 1
fi

cd "$ROOT"
source "$VENV_ACTIVATE"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m trading_lab ui "$@"
