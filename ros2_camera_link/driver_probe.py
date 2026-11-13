#!/usr/bin/env python3
"""Inspect an already sourced ROS environment; never start a camera or a DDS node."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

MARKER = 'CAMERA_DRIVER_PROBE_JSON='
CORE_PACKAGES = ('rmw_fastrtps_cpp', 'ros2cli', 'ros2topic', 'rclpy', 'sensor_msgs')
DRIVERS = {'orbbec_camera': ('gemini2.launch.py', 'start_orbbec_camera.sh'),
           'realsense2_camera': ('rs_launch.py', 'start_d405.sh')}


def check_distro(config):
    expected = config['ros_distro']
    if os.environ.get('ROS_DISTRO') != expected:
        raise ValueError('驱动工作空间加载了不同的 ROS 版本，预期：' + expected)
    for key in ('AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        for entry in os.environ.get(key, '').split(os.pathsep):
            match = re.match(r'^/opt/ros/([^/]+)(?:/|$)', entry)
            if match and match[1] != expected:
                raise ValueError('工作空间混入其他 ROS 版本：' + entry)


def argument_names(output):
    return set(re.findall(r'^\s*[\'\"]([^\'\"]+)[\'\"]:', output, flags=re.MULTILINE))


def required_arguments(script):
    return set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_.]*)\s*:=', Path(script).read_text()))


def inspect(config, quick=False, script_dir=None):
    check_distro(config)
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    packages = list(CORE_PACKAGES)
    if config['device_role'] == 'publisher':
        packages += list(DRIVERS)
    result = {}
    for name in packages:
        prefix = str(Path(get_package_prefix(name)).resolve())
        share = Path(get_package_share_directory(name))
        package_xml = share / 'package.xml'
        version = ET.parse(package_xml).getroot().findtext('version') if package_xml.is_file() else 'unknown'
        expected = config.get('validated_packages', {}).get(name, {}).get('prefix')
        if expected and Path(expected).resolve() != Path(prefix):
            raise ValueError(f'{name} 加载路径发生变化：{prefix}；部署时为 {expected}，请核对 setup_files 后重新部署。')
        result[name] = dict(prefix=prefix, share=str(share), version=version)
    if quick:
        return result
    # Load type support and the RMW library without creating participants/nodes.
    from rclpy.utilities import get_rmw_implementation_identifier
    from sensor_msgs.msg import Image, CameraInfo
    if get_rmw_implementation_identifier() != 'rmw_fastrtps_cpp':
        raise ValueError('没有加载 rmw_fastrtps_cpp。')
    for msg in (Image, CameraInfo):
        msg.__class__.__import_type_support__()
        if msg._TYPE_SUPPORT is None:
            raise ValueError('sensor_msgs 类型支持加载失败。')
    if config['device_role'] != 'publisher':
        return result
    executable = shutil.which('ros2')
    if not executable:
        raise ValueError('未找到 ros2 命令。')
    for name, (launch, script) in DRIVERS.items():
        path = Path(result[name]['share']) / 'launch' / launch
        if not path.is_file():
            raise ValueError('驱动缺少启动文件：' + str(path))
        with tempfile.TemporaryDirectory(prefix='camera-launch-check-') as logs:
            env = dict(os.environ, ROS_LOG_DIR=logs, PYTHONDONTWRITEBYTECODE='1')
            checked = subprocess.run([executable, 'launch', name, launch, '--show-args'],
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, timeout=45)
        if checked.returncode:
            raise ValueError(f'{name} 的启动文件无法加载：\n' + checked.stdout[-5000:])
        needed = required_arguments(Path(script_dir) / script)
        missing = needed - argument_names(checked.stdout)
        if missing:
            raise ValueError(name + ' 缺少本包使用的启动参数：' + ', '.join(sorted(missing))
                             + '。请使用兼容驱动；复用模式不会自动升级。')
        result[name]['launch'] = str(path)
        result[name]['checked_arguments'] = sorted(needed)
    return result


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path)
    source.add_argument('--config-json')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--script-dir', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text() if args.config else args.config_json)
    result = inspect(config, args.quick, args.script_dir)
    print(MARKER + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('驱动检查失败：' + str(error))
