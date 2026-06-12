#!/usr/bin/env bash

TRADING_LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$TRADING_LAB_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: virtual environment not found: $VENV_ACTIVATE" >&2
    return 1 2>/dev/null || exit 1
fi

cd "$TRADING_LAB_ROOT" || return 1 2>/dev/null || exit 1
source "$VENV_ACTIVATE"
echo "Trading Lab activated: $TRADING_LAB_ROOT"
echo "Run dashboard: python -m trading_lab ui"
