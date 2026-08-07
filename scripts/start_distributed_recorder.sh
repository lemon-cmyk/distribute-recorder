#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="distribute_recorder"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf -v QUOTED_PROJECT_DIR "%q" "${PROJECT_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed." >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[INFO] tmux session '${SESSION_NAME}' already exists. Attaching..."
    exec tmux attach-session -t "${SESSION_NAME}"
fi

converter_command="cd ${QUOTED_PROJECT_DIR}; exec ./scripts/start_converter.sh"
for argument in "$@"; do
    printf -v quoted_argument "%q" "${argument}"
    converter_command+=" ${quoted_argument}"
done

echo "[INFO] Creating tmux session: ${SESSION_NAME}"

# Window 1: Orin distributed recording server.
tmux new-session -d -s "${SESSION_NAME}" -n "recorder" -c "${PROJECT_DIR}"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on >/dev/null
tmux send-keys -t "${SESSION_NAME}:recorder" -l \
    "echo '[recorder] Starting distributed recorder server...'; cd ${QUOTED_PROJECT_DIR}; exec ./scripts/start_recorder.sh"
tmux send-keys -t "${SESSION_NAME}:recorder" Enter

# Window 2: PKL-to-LeRobot converter. All arguments are forwarded unchanged.
tmux new-window -t "${SESSION_NAME}" -n "converter" -c "${PROJECT_DIR}"
tmux send-keys -t "${SESSION_NAME}:converter" -l \
    "echo '[converter] Starting PKL-to-LeRobot converter...'; ${converter_command}"
tmux send-keys -t "${SESSION_NAME}:converter" Enter

tmux select-window -t "${SESSION_NAME}:recorder"

echo "[INFO] Recorder and converter started in '${SESSION_NAME}'."
echo "[INFO] Attaching to recorder window; press Ctrl+B then n to switch windows."

exec tmux attach-session -t "${SESSION_NAME}"
