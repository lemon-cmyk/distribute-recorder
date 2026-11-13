#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $EUID == 0 ]] || { echo '请通过 realsense-dual.service 启动摄像头。' >&2; exit 1; }
side=${1:-}
[[ "$side" == left || "$side" == right ]] || { echo '参数应为 left 或 right。' >&2; exit 2; }
shift
serial=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cameras"][sys.argv[2]+"_serial"])' "$root/config.json" "$side")
[[ "$serial" =~ ^[0-9]+$ ]] || { echo 'D405 序列号未配置。' >&2; exit 1; }
exec 9>"/run/lock/ros2-camera-d405-${serial}.lock"
flock -n 9 || { echo "D405 $serial 已有本包启动的进程。" >&2; exit 1; }
source "$root/env.sh" publisher
exec ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=wrist_cameras camera_name:="$side" serial_no:="_$serial" \
  enable_color:=true enable_depth:=true \
  depth_module.depth_profile:=640x480x30 depth_module.color_profile:=640x480x30 \
  depth_module.depth_format:=Z16 depth_module.color_format:=RGB8 \
  enable_infra:=false enable_infra1:=false enable_infra2:=false \
  enable_gyro:=false enable_accel:=false \
  pointcloud.enable:=false align_depth.enable:=false enable_sync:=false \
  diagnostics_period:=1.0 "$@"
