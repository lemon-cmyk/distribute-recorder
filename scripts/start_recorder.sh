#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="/home/nvidia/codes/tiangong_distributed_recorder"

source /opt/ros/humble/setup.bash
if [ -f /home/nvidia/ws_tianyi_robot/install/setup.bash ]; then
    source /home/nvidia/ws_tianyi_robot/install/setup.bash
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"
exec python3 -m tiangong_recorder.recorder_server --config config/recorder.yaml
