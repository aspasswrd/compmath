#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
  echo "Virtualenv .venv не найден. Создаю..."
  python3 -m venv "${SCRIPT_DIR}/.venv"
  source "${SCRIPT_DIR}/.venv/bin/activate"
  python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
else
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

cleanup() {
  if [[ -n "${PY_PID:-}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    kill "${PY_PID}" 2>/dev/null || true
    wait "${PY_PID}" 2>/dev/null || true
  fi
  if declare -F deactivate >/dev/null; then
    deactivate || true
  fi
}

abort_run() {
  echo "Получен сигнал прерывания, завершаю GUI..." >&2
  cleanup
  exit 130
}

trap abort_run INT TERM

python3 "${SCRIPT_DIR}/gui.py" &
PY_PID=$!

wait "${PY_PID}"
STATUS=$?

trap - INT TERM
cleanup

exit "${STATUS}"
