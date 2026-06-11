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
exec streamlit run dashboard/app.py "$@"
