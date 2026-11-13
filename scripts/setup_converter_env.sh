#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv-lerobot"
ARCH="$(uname -m)"
WHEELHOUSE="${WHEELHOUSE:-/media/grg/robot_shared/.cache/thor-distribute-recorder-wheelhouse}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
elif [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    PYTHON="/usr/bin/python3.12"
else
    PYTHON="python3.10"
fi

unset PYTHONPATH
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if "${PYTHON}" -c 'import ensurepip' >/dev/null 2>&1; then
        "${PYTHON}" -m venv "${VENV_DIR}"
    else
        "${PYTHON}" -m venv --without-pip "${VENV_DIR}"
        PIP_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'pip-*.whl' -print -quit)"
        if [[ -z "${PIP_WHEEL}" ]]; then
            echo "ensurepip is unavailable and no pip wheel was found in ${WHEELHOUSE}" >&2
            exit 1
        fi
        PYTHONPATH="${PIP_WHEEL}" "${VENV_DIR}/bin/python" -m pip install \
            --no-index --find-links "${WHEELHOUSE}" pip setuptools wheel
    fi
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
if [[ "${ARCH}" == "x86_64" ]]; then
    "${VENV_DIR}/bin/python" -m pip install "torch==2.7.1+cpu" "torchvision==0.22.1+cpu" --index-url https://download.pytorch.org/whl/cpu
elif [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    "${VENV_DIR}/bin/python" -m pip install "torch==2.7.1" "torchvision==0.22.1"
else
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
fi
"${VENV_DIR}/bin/python" -m pip install "av==17.1.0"
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[lerobot-v21]"

echo "LeRobot converter environment is ready: ${VENV_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import av
import platform
import torch
import torchvision
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

av.codec.Codec("libsvtav1", "w")
print(f"Architecture: {platform.machine()}")
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"PyAV: {av.__version__}")
print(f"LeRobot codebase: {CODEBASE_VERSION}")
print("Software AV1 encoder ready: libsvtav1")
PY
