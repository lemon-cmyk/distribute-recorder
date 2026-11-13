#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash "$root/start_d405.sh" left "$@" &
pids+=("$!")
sleep 2
bash "$root/start_d405.sh" right "$@" &
pids+=("$!")
wait -n "${pids[@]}" || true
echo '一个 D405 启动进程退出，由 systemd 重启两台相机。' >&2
exit 1
