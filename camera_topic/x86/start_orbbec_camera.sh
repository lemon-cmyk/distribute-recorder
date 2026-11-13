#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $EUID == 0 ]] || { echo '请通过 orbbec-camera.service 启动摄像头。' >&2; exit 1; }
exec 9>/run/lock/ros2-camera-gemini2.lock
flock -n 9 || { echo 'Gemini 2 已有本包启动的进程。' >&2; exit 1; }
source "$root/env.sh" publisher
serial=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cameras"]["gemini2_serial"])' "$root/config.json")
serial_args=()
[[ -z "$serial" ]] || serial_args+=("serial_number:=$serial")
exec ros2 launch orbbec_camera gemini2.launch.py \
  camera_name:=camera "${serial_args[@]}" \
  enable_color:=true enable_depth:=true enable_ir:=false \
  enable_point_cloud:=false enable_colored_point_cloud:=false \
  depth_registration:=true \
  color_width:=640 color_height:=480 color_fps:=30 color_format:=RGB \
  depth_width:=640 depth_height:=400 depth_fps:=30 depth_format:=Y16 \
  color_qos:=SENSOR_DATA depth_qos:=SENSOR_DATA "$@"
