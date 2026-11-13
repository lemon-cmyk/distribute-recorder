#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo '请使用 source 加载 receiver_env.sh，而不是 bash。' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh" subscriber
