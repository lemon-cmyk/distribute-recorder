#!/usr/bin/env python3
"""Install the bundled pure Topic camera setup on an already ROS-enabled Ubuntu host."""
import argparse
import datetime
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SDK_SHA256 = '6953ee70546e901746f8d051c01d30418ba67d0b2ba1449e6d45a595ca6180fa'
SDK = ROOT / 'x86/sources/OrbbecSDK_ROS2-2.9.3-snapshot.tar.gz'
ROS_ROOT = Path('/opt/ros')
CAMERA_SERVICES = ('orbbec-camera.service', 'realsense-dual.service')


def run(args, **kwargs):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def clean_env(user='root', home='/root'):
    return dict(HOME=home, USER=user, LOGNAME=user, LANG='C.UTF-8',
                PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')


def driver_options(cfg):
    options = cfg.get('drivers', {'mode': 'install', 'setup_files': []})
    if not isinstance(options, dict) or options.get('mode') not in ('install', 'reuse'):
        raise ValueError('drivers.mode 应为 install（安装）或 reuse（复用）。')
    files = options.get('setup_files', [])
    if not isinstance(files, list) or any(not isinstance(p, str) or not Path(p).is_absolute()
                                        or any(ord(c) < 32 for c in p) for p in files):
        raise ValueError('drivers.setup_files 应是已有 setup.bash/local_setup.bash 的绝对路径列表。')
    if len(files) != len(set(files)):
        raise ValueError('drivers.setup_files 中有重复路径。')
    if options['mode'] == 'install' and files:
        raise ValueError('填写已有工作空间时请将 drivers.mode 改为 reuse；install 模式不混用已有驱动。')
    return options['mode'], files


def load_config(path, device, require_serials=True):
    cfg = json.loads(path.read_text())
    driver_options(cfg)
    if type(cfg['domain_id']) is not int or not 0 <= cfg['domain_id'] <= 232:
        raise ValueError('domain_id 应为 0～232 的整数。')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', cfg['user']) or cfg['user'] == 'root':
        raise ValueError('user 应填写普通用户，例如 grg。')
    if type(cfg.get('build_jobs', 2)) is not int or not 1 <= cfg.get('build_jobs', 2) <= 32:
        raise ValueError('build_jobs 应为 1～32。')
    devices = cfg['devices']
    if device not in devices:
        raise ValueError('设备名称不在 robot.json 中：' + device)
    publishers, ips = [], []
    for name, item in devices.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('非法设备名：' + name)
        address = ipaddress.IPv4Address(item['internal_ip'])
        if address.is_loopback or address.is_multicast or address.is_unspecified or address.is_link_local:
            raise ValueError('internal_ip 必须是设备已有的可用有线 IPv4。')
        ips.append(str(address))
        if item['role'] == 'publisher':
            publishers.append(name)
            if item['ros_distro'] != 'humble':
                raise ValueError('本包 x86 驱动固定为 Ubuntu 22.04 / ROS Humble。')
        elif item['role'] != 'receiver':
            raise ValueError('role 只能为 publisher 或 receiver。')
        if item['ros_distro'] not in ('humble', 'jazzy', 'auto'):
            raise ValueError('接收端支持 humble、jazzy 或 auto。')
    if len(publishers) != 1 or len(ips) != len(set(ips)):
        raise ValueError('必须恰有一个 publisher，且各设备有线 IP 不能重复。')
    cameras = cfg['cameras']
    if not isinstance(cameras['gemini2_serial'], str) or not re.fullmatch(r'[A-Za-z0-9_-]*', cameras['gemini2_serial']):
        raise ValueError('gemini2_serial 应为空或设备序列号。')
    if require_serials and devices[device]['role'] == 'publisher':
        serials = [cameras['left_serial'], cameras['right_serial']]
        if any(not isinstance(s, str) or not re.fullmatch(r'[0-9]{6,20}', s) for s in serials) or serials[0] == serials[1]:
            raise ValueError('请在 robot.json 填写两台 D405 不同的设备序列号，必须为字符串；不要填写 ASIC Serial Number。')
    return cfg


def local_config(cfg, device, distro, home):
    item = cfg['devices'][device]
    role = item['role']
    mode, setups = driver_options(cfg)
    if role != 'publisher':
        mode, setups = 'none', []
    elif mode == 'install':
        setups = [str(home / 'orb_ros2/install/local_setup.bash')]
    return dict(device=device, device_role=role, domain_id=cfg['domain_id'],
                ros_distro=distro, internal_ip=item['internal_ip'],
                peers=[v['internal_ip'] for k, v in cfg['devices'].items() if k != device],
                cameras=cfg['cameras'],
                driver_mode=mode, driver_setup_files=setups,
                orbbec_workspace=str(home / 'orb_ros2') if role == 'publisher' and mode == 'install' else '')


def run_in_ros(config, command, account=None, capture=False):
    """Source only the declared environment, using positional arguments rather than shell interpolation."""
    setups = config['driver_setup_files']
    base = ROS_ROOT / config['ros_distro'] / 'setup.bash'
    for path in [base, *map(Path, setups)]:
        if not path.is_file():
            raise ValueError('环境文件不存在：' + str(path))
    script = ('set -eo pipefail; source "$1"; shift; count=$1; shift; '
              'for ((i=0; i<count; i++)); do source "$1"; shift; done; exec "$@"')
    target = account or pwd.getpwuid(os.geteuid())
    env = clean_env(target.pw_name, target.pw_dir)
    env.update(RMW_IMPLEMENTATION='rmw_fastrtps_cpp', PYTHONDONTWRITEBYTECODE='1')
    args = ['bash', '--noprofile', '--norc', '-c', script, 'camera-driver-check',
            str(base), str(len(setups)), *setups, *map(str, command)]
    if account is not None and os.geteuid() == 0 and account.pw_uid != 0:
        args = ['runuser', '-u', account.pw_name, '--', '/usr/bin/env', '-i',
                *[f'{k}={v}' for k, v in env.items()], *args]
    result = subprocess.run(args, env=env, cwd='/tmp', text=True, timeout=120,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if result.returncode:
        detail = (result.stdout or '')[-6000:]
        raise ValueError('加载/检查已有 ROS 驱动失败：\n' + detail)
    return result.stdout or ''


def probe_drivers(config, account=None, quick=False):
    command = ['/usr/bin/python3', ROOT / 'common/driver_probe.py',
               '--config-json', json.dumps(config), '--script-dir', ROOT / 'x86']
    if quick:
        command.append('--quick')
    output = run_in_ros(config, command, account=account, capture=True)
    marker = 'CAMERA_DRIVER_PROBE_JSON='
    records = [line[len(marker):] for line in output.splitlines() if line.startswith(marker)]
    if len(records) != 1:
        raise ValueError('驱动检查未返回有效结果：\n' + output[-3000:])
    return json.loads(records[0])


def camera_process(argv):
    if not argv:
        return False
    names = [Path(a).name for a in argv]
    if names[0] in ('orbbec_camera_node', 'realsense2_camera_node'):
        return True
    if 'ros2' in names and 'launch' in argv and any(p in argv for p in ('orbbec_camera', 'realsense2_camera')):
        return True
    if any(name in names for name in ('start_orbbec_camera.sh', 'start_d405.sh', 'start_dual_d405.sh')):
        return True
    return ('component_container' in names[0]
            and any(a == '__node:=camera_container' or a.startswith('__ns:=/wrist_cameras') for a in argv))


def check_camera_processes(proc_root=Path('/proc')):
    """Reject recognizable camera processes outside the two services that this kit takes over."""
    conflicts = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            if not camera_process(argv):
                continue
            cgroups = directory.joinpath('cgroup').read_text().splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(line.endswith('/' + service) for line in cgroups for service in CAMERA_SERVICES):
            conflicts.append(f'PID {directory.name}（{Path(argv[0]).name}）')
    if conflicts:
        raise ValueError('检测到本包两项服务之外的相机进程：' + '、'.join(conflicts)
                         + '。请先停止原相机进程；若有旧自启动服务，也应先禁用以避免重启后争用。未停止其他服务。')


def prepare_drivers(config, cfg, home, account):
    """Reuse must never execute APT, build, chown or udev operations."""
    ws = None
    if config['driver_mode'] != 'reuse':
        apt_install(dependencies(config['ros_distro'], config['device_role'] == 'publisher'))
        if config['device_role'] == 'publisher':
            ws = build_orbbec(home, account, cfg.get('build_jobs', 2))
    config['validated_packages'] = probe_drivers(config)
    if account.pw_uid != os.geteuid():
        probe_drivers(config, account=account, quick=True)
    for name in ('orbbec_camera', 'realsense2_camera'):
        info = config['validated_packages'].get(name)
        if info:
            print(f'使用 {name} {info["version"]}：{info["prefix"]}', flush=True)
    return ws


def get_transport():
    spec = importlib.util.spec_from_file_location('camera_transport', ROOT / 'common/transport.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_preflight(cfg, device, check_network):
    if platform.system() != 'Linux' or not Path('/run/systemd/system').is_dir():
        raise ValueError('部署需在使用 systemd 的 Ubuntu 目标设备本机执行。')
    account = pwd.getpwnam(cfg['user'])
    home = Path(account.pw_dir)
    if not home.is_dir() or not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(home)):
        raise ValueError('普通用户的 HOME 必须存在，路径中不能有空格。')
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    if release.get('ID', '').strip('"') != 'ubuntu':
        raise ValueError('本包仅支持 Ubuntu。')
    item = cfg['devices'][device]
    available = [p.parent.name for p in Path('/opt/ros').glob('*/setup.bash')]
    distro = item['ros_distro']
    if distro == 'auto':
        if len(available) != 1:
            raise ValueError('检测到多个或没有 ROS，请在 robot.json 明确 ros_distro。')
        distro = available[0]
    if distro not in ('humble', 'jazzy') or distro not in available:
        raise ValueError('未找到配置要求的 ROS 安装；脚本不会代装 ROS。')
    version = release.get('VERSION_ID', '').strip('"')
    if (distro, version) not in (('humble', '22.04'), ('jazzy', '24.04')):
        raise ValueError('本包支持 Ubuntu 22.04/Humble 和 Ubuntu 24.04/Jazzy。')
    if item['role'] == 'publisher' and platform.machine() != 'x86_64':
        raise ValueError('本包摄像头发布端应为 x86_64。')
    config = local_config(cfg, device, distro, home)
    if check_network:
        transport = get_transport()
        links = transport.read_json_command(['ip', '-j', '-4', 'address', 'show'])
        routes = [transport.read_json_command(['ip', '-j', '-4', 'route', 'get', p])[0] for p in config['peers']]
        transport.select_wired(config, links, routes)
    return account, home, distro, config


def apt_install(packages):
    run(['apt-get', 'update'], env=clean_env())
    # Some ROS repositories bundle the utilities in librealsense2 instead of a separate tools package.
    selected = []
    for package in packages:
        if package.endswith('-librealsense2-tools'):
            policy = subprocess.run(['apt-cache', 'policy', package], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=clean_env())
            match = re.search(r'Candidate:\s*(\S+)', policy.stdout)
            if not match or match[1] == '(none)':
                print('仓库无独立工具包，使用主 SDK 提供的工具：' + package, flush=True)
                continue
        selected.append(package)
    # Allow a bounded wait for an existing package manager; never remove lock files.
    run(['apt-get', '-o', 'DPkg::Lock::Timeout=180', 'install', '-y', '--no-remove', *selected], env=clean_env())


def dependencies(distro, publisher):
    base = ['python3', 'iproute2', 'procps', 'util-linux',
            f'ros-{distro}-rmw-fastrtps-cpp', f'ros-{distro}-ros2cli',
            f'ros-{distro}-ros2topic', f'ros-{distro}-rclpy', f'ros-{distro}-sensor-msgs']
    if not publisher:
        return base
    system = ['build-essential', 'cmake', 'pkg-config', 'python3-colcon-common-extensions',
              'libgflags-dev', 'nlohmann-json3-dev', 'libdw-dev', 'libssl-dev',
              'libusb-1.0-0-dev', 'libgoogle-glog-dev', 'libgl1', 'libeigen3-dev',
              'libopencv-dev', 'libyaml-cpp-dev']
    ros = ['ament-cmake', 'ament-index-cpp', 'backward-ros', 'cv-bridge', 'camera-info-manager',
           'camera-calibration-parsers', 'image-transport', 'image-transport-plugins',
           'compressed-image-transport', 'image-publisher', 'rclcpp', 'rclcpp-components',
           'builtin-interfaces', 'std-msgs', 'std-srvs', 'statistics-msgs', 'tf2', 'tf2-msgs',
           'tf2-ros', 'tf2-sensor-msgs', 'diagnostic-updater', 'diagnostic-msgs',
           'rosidl-default-generators', 'rosidl-default-runtime', 'launch-ros', 'ros2launch',
           'xacro', 'librealsense2', 'librealsense2-tools', 'realsense2-camera', 'realsense2-description']
    return base + system + ['ros-humble-' + p for p in ros]


def build_orbbec(home, account, jobs):
    if hashlib.sha256(SDK.read_bytes()).hexdigest() != SDK_SHA256:
        raise ValueError('随包 Orbbec 源码校验失败。')
    ws = home / 'orb_ros2'
    marker = ws / '.topic-kit-sdk.sha256'
    if ws.exists() and any(ws.iterdir()) and (not marker.is_file() or marker.read_text().strip() != SDK_SHA256):
        raise ValueError(str(ws) + ' 已有非本包工作空间；请先另行保留并移走该目录，脚本不会覆盖。')
    ws.mkdir(exist_ok=True)
    marker.write_text(SDK_SHA256 + '\n')
    source = ws / 'src'
    source.mkdir(exist_ok=True)
    if not (source / 'OrbbecSDK_ROS2').exists():
        # The bundled, hash-verified archive is trusted, but also reject escaping paths/links.
        with tarfile.open(SDK) as archive:
            for member in archive.getmembers():
                target = (source / member.name).resolve()
                if source.resolve() not in target.parents:
                    raise ValueError('源码包包含越界路径。')
                if member.isdev() or member.isfifo():
                    raise ValueError('源码包包含不支持的特殊文件。')
                if member.issym() or member.islnk():
                    link = (target.parent / member.linkname if member.issym() else source / member.linkname).resolve()
                    if source.resolve() not in link.parents:
                        raise ValueError('源码包包含越界链接。')
            archive.extractall(source)
    run(['chown', '-R', f'{account.pw_uid}:{account.pw_gid}', ws])
    built = ws / '.topic-kit-build.ok'
    if not built.is_file() or not (ws / 'install/local_setup.bash').is_file():
        env = clean_env(account.pw_name, str(home))
        env.update(CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), MAKEFLAGS='-j' + str(jobs))
        run(['runuser', '-u', account.pw_name, '--', '/usr/bin/env', '-i',
             *[f'{k}={v}' for k, v in env.items()], 'bash', '--noprofile', '--norc', '-c',
             'set -eo pipefail; source /opt/ros/humble/setup.bash; '
             'colcon build --symlink-install --executor sequential --event-handlers console_direct+ '
             '--cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF'], cwd=ws)
        built.write_text(SDK_SHA256 + '\n')
        os.chown(built, account.pw_uid, account.pw_gid)
    if not (ws / 'install/orbbec_camera/share/orbbec_camera/package.xml').is_file():
        raise ValueError('Orbbec 编译产物不完整，未安装自启动服务。')
    return ws


class Writer:
    def __init__(self):
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.backup = Path('/var/backups/ros2-camera-topic') / stamp
        self.changed = []

    def put(self, path, content, mode=0o644):
        path = Path(path)
        data = content.encode() if isinstance(content, str) else content
        if path.is_symlink():
            raise ValueError('拒绝覆盖符号链接：' + str(path))
        if path.exists() and path.read_bytes() != data:
            backup = self.backup / str(path).lstrip('/')
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.chmod(temp, mode)
        os.replace(temp, path)
        self.changed.append(str(path))


def usb_rules(writer, ws, user):
    rule = ws / 'src/OrbbecSDK_ROS2/orbbec_camera/scripts/99-obsensor-libusb.rules'
    writer.put('/etc/udev/rules.d/99-obsensor-libusb.rules', rule.read_bytes())
    writer.put('/etc/udev/rules.d/99-z-topic-d405.rules',
               'SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5b", MODE:="0660", GROUP:="video"\n')
    run(['usermod', '-aG', 'video', user])
    run(['udevadm', 'control', '--reload-rules'])
    for vendor in ('2bc5', '8086'):
        command = ['udevadm', 'trigger', '--action=add', '--subsystem-match=usb', '--attr-match=idVendor=' + vendor]
        if vendor == '8086':
            command.append('--attr-match=idProduct=0b5b')
        run(command)
    run(['udevadm', 'settle', '--timeout=15'])


def main():
    parser = argparse.ArgumentParser(description='三摄像头纯 Topic 部署（ROS 已安装）。')
    parser.add_argument('device', help='robot.json 中的设备名称：x86 / thor1 / thor2')
    parser.add_argument('--config', type=Path, default=ROOT / 'robot.json')
    parser.add_argument('--head-only', action='store_true', help='仅部署头部自启动；左右尚未确认时保留腕部驱动并暂停腕部自启动')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='只检查；复用模式还在目标机检查已有驱动与启动参数，不修改服务')
    mode.add_argument('--list-cameras', action='store_true', help='列出 D405 序列号；复用模式只调用已有枚举工具')
    args = parser.parse_args()
    cfg = load_config(args.config, args.device, require_serials=not (args.list_cameras or args.head_only))
    publisher = cfg['devices'][args.device]['role'] == 'publisher'
    if args.head_only and not publisher:
        raise ValueError('--head-only 仅适用于 x86 发布端。')
    reuse = publisher and driver_options(cfg)[0] == 'reuse'
    if args.check:
        if reuse:
            account, home, distro, config = host_preflight(cfg, args.device, check_network=True)
            config['validated_packages'] = probe_drivers(config)
            if os.geteuid() == 0 and account.pw_uid != 0:
                probe_drivers(config, account=account, quick=True)
            check_camera_processes()
            for name in ('orbbec_camera', 'realsense2_camera'):
                info = config['validated_packages'][name]
                print(f'{name} {info["version"]}：{info["prefix"]}')
            print('复用检查通过；未安装软件、编译源码、修改配置或启停服务，未做相机硬件验证。')
            return
        if publisher and hashlib.sha256(SDK.read_bytes()).hexdigest() != SDK_SHA256:
            raise ValueError('SDK 校验失败。')
        print('配置检查通过；未进行安装、网络连接或硬件验证。')
        return
    if os.geteuid() != 0:
        raise ValueError('请通过 bash deploy.sh 调用，脚本会使用 sudo。')
    account, home, distro, config = host_preflight(cfg, args.device, not args.list_cameras)
    config['wrist_enabled'] = publisher and not args.head_only
    if args.list_cameras:
        if not publisher:
            raise ValueError('--list-cameras 仅用于连接摄像头的 x86。')
        if reuse:
            run_in_ros(config, ['bash', '-c',
                       'command -v rs-enumerate-devices >/dev/null || { '
                       'echo "未找到已有 rs-enumerate-devices；复用模式不会安装 SDK，请确认已有工具路径。" >&2; exit 1; }; '
                       'exec rs-enumerate-devices -s'])
        else:
            apt_install(['ros-humble-librealsense2', 'ros-humble-librealsense2-tools'])
            run(['bash', '--noprofile', '--norc', '-c',
                 'source /opt/ros/humble/setup.bash && rs-enumerate-devices -s'], env=clean_env())
        print('请将 Serial Number 写入 robot.json 的 left_serial / right_serial，再正常部署。')
        return
    if publisher:
        check_camera_processes()
    ws = prepare_drivers(config, cfg, home, account)
    if publisher:
        check_camera_processes()
    runtime = home / 'ros2_camera_link'
    writer = Writer()
    # Finish dependencies and compilation before changing running camera services.
    if publisher:
        for service in CAMERA_SERVICES:
            if subprocess.run(['systemctl', 'cat', service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                run(['systemctl', 'stop', service])
                if args.head_only and service == 'realsense-dual.service':
                    run(['systemctl', 'disable', service])
    for path in sorted((ROOT / 'common').iterdir()):
        if path.suffix in ('.py', '.sh', '.txt'):
            writer.put(runtime / path.name, path.read_bytes(), 0o755 if path.suffix in ('.py', '.sh') else 0o644)
    writer.put(runtime / 'config.json', json.dumps(config, indent=2) + '\n')
    writer.put(runtime / 'robot.json', args.config.read_bytes())
    writer.put('/etc/systemd/system/ros2-camera-buffers.service',
               (ROOT / 'common/buffers.service.in').read_text().format(runtime=runtime))
    services = ['ros2-camera-buffers.service']
    if publisher:
        for script in ('start_orbbec_camera.sh', 'start_d405.sh', 'start_dual_d405.sh'):
            writer.put(runtime / script, (ROOT / 'x86' / script).read_bytes(), 0o755)
        if not reuse:
            usb_rules(writer, ws, cfg['user'])
        template = (ROOT / 'x86/camera.service.in').read_text()
        for name, script, label in (
                ('orbbec-camera.service', 'start_orbbec_camera.sh', 'Gemini 2 head camera'),
                ('realsense-dual.service', 'start_dual_d405.sh', 'Left and right D405 wrist cameras')):
            if args.head_only and name == 'realsense-dual.service':
                continue
            writer.put('/etc/systemd/system/' + name, template.format(description=label, runtime=runtime, script=script))
            services.append(name)
    run(['systemctl', 'daemon-reload'])
    run(['systemctl', 'enable', *services])
    run(['systemctl', 'restart', *services])
    versions = ''.join(f'{name}\t{info["version"]}\t{info["prefix"]}\n'
                       for name, info in config['validated_packages'].items())
    # A source-built SDK need not be registered in dpkg; missing entries are not deployment failures.
    apt_versions = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n',
                                  f'ros-{distro}-rmw-fastrtps-cpp',
                                  *(['ros-humble-librealsense2', 'ros-humble-realsense2-camera'] if publisher else [])],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    versions += '\nAPT entries (when installed):\n' + apt_versions.stdout
    writer.put(runtime / 'installed-versions.txt', versions)
    writer.put(runtime / 'deployment.json', json.dumps(dict(device=args.device,
               installed_at=datetime.datetime.now().isoformat(), driver_mode=config['driver_mode'],
               head_only=args.head_only,
               sdk_sha256=SDK_SHA256 if publisher and not reuse else None,
               backup=str(writer.backup), installed_files=writer.changed), indent=2) + '\n')
    run(['systemctl', '--no-pager', '--full', 'status', *services])
    print('\n配置及服务已安装。服务 active 不代表图像帧率验收通过。')
    print('root 和普通用户均执行：source ' + str(runtime / 'receiver_env.sh'))
    print('图像订阅由你自行启动；本脚本不自动执行 topic hz。')
    if args.head_only:
        print('当前仅启动头部；腕部左右未启用。确认序列号后，重新运行不带 --head-only 的部署命令。')
    if publisher and not reuse:
        print('普通用户新加入 video 组后，重新登录可获得直接访问 USB 的权限。')
    if writer.backup.exists():
        print('被替换文件的备份：' + str(writer.backup))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit('部署停止：' + str(error))
