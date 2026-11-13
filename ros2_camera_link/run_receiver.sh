#!/usr/bin/env bash
# 保留先前使用的入口名称，行为与 receiver_env.sh 完全相同。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo '请使用 source 加载 run_receiver.sh，而不是 bash。' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/receiver_env.sh"
