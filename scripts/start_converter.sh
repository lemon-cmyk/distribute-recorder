#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-lerobot/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Converter environment is missing. Run scripts/setup_converter_env.sh first." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
COMMAND=(
    "${PYTHON}"
    -m tiangong_recorder.lerobot_converter
    --config config/converter.yaml
)

if command -v ionice >/dev/null 2>&1; then
    exec ionice -c 2 -n 7 nice -n 10 "${COMMAND[@]}" "$@"
fi
exec nice -n 10 "${COMMAND[@]}" "$@"
