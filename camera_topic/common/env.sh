#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo '此文件需要用 source 加载。' >&2
  exit 2
fi
_camera_topic_environment() {
  local role=${1:-subscriber} root result item
  local -a values=()
  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return
  result=$(/usr/bin/python3 "$root/transport.py" "$role") || return
  while IFS= read -r item; do values+=("$item"); done <<< "$result"
  [[ ${#values[@]} -ge 3 && -r "${values[2]}" ]] || return 1
  if [[ -n "${ROS_DISTRO:-}" && "$ROS_DISTRO" != "${values[0]}" ]]; then
    echo '当前终端已加载另一版本 ROS，请在新终端 source 本文件。' >&2
    return 1
  fi
  source "/opt/ros/${values[0]}/setup.bash" || return
  for item in "${values[@]:3}"; do
    [[ -r "$item" ]] || { echo "驱动环境文件不可读：$item" >&2; return 1; }
    source "$item" || return
  done
  # Confirm that startup uses the same package prefixes validated during deployment.
  /usr/bin/python3 "$root/driver_probe.py" --config "$root/config.json" --quick >/dev/null || return
  export ROS_DOMAIN_ID="${values[1]}" RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  unset ROS_LOCALHOST_ONLY FASTDDS_BUILTIN_TRANSPORTS ROS_STATIC_PEERS ROS_DISCOVERY_SERVER
  unset RMW_FASTRTPS_USE_QOS_FROM_XML
  export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  export FASTRTPS_DEFAULT_PROFILES_FILE="${values[2]}"
  export FASTDDS_DEFAULT_PROFILES_FILE="${values[2]}"
  export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
  printf 'Domain %s，%s；已加载当前终端，未启动订阅。\n' "$ROS_DOMAIN_ID" "${values[2]}"
}
if _camera_topic_environment "${1:-subscriber}"; then
  unset -f _camera_topic_environment
else
  unset -f _camera_topic_environment
  return 1
fi
