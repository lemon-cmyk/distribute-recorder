#!/usr/bin/env python3
"""Offline regression checks. ROS, APT and systemd calls are mocked; no robot is contacted."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


install = module('camera_install_test', ROOT / 'scripts/install.py')
probe = module('camera_probe_test', ROOT / 'common/driver_probe.py')
transport = module('camera_transport_test', ROOT / 'common/transport.py')


class ReuseTests(unittest.TestCase):
    def test_publisher_has_unicast_discovery_without_enabling_wifi_or_disabling_shm(self):
        data = transport.build_xml('publisher', self.config, '192.168.3.102',
                                   ['127.0.0.1', '192.168.3.102', '10.1.52.51'], [], True)
        root = ET.fromstring(data)
        ns = {'d': transport.NS}
        rtps = root.find('d:profiles/d:participant/d:rtps', ns)
        addresses = rtps.findall('d:builtin/d:metatrafficUnicastLocatorList/d:locator/d:udpv4/d:address', ns)
        self.assertEqual({a.text for a in addresses}, {'192.168.3.102', '127.0.0.1'})
        self.assertIsNone(rtps.find('d:defaultUnicastLocatorList', ns))
        self.assertNotIn(b'10.1.52.51', data)
        self.assertIn(b'<type>SHM</type>', data)

    def test_optional_tools_package_missing_from_repository(self):
        with patch.object(install, 'run') as commands, patch.object(install.subprocess, 'run',
                return_value=SimpleNamespace(stdout='', returncode=0)):
            install.apt_install(['ros-humble-librealsense2', 'ros-humble-librealsense2-tools'])
        self.assertEqual(commands.call_args_list[-1].args[0],
                         ['apt-get', '-o', 'DPkg::Lock::Timeout=180', 'install', '-y', '--no-remove', 'ros-humble-librealsense2'])

    def test_optional_tools_package_present_in_repository(self):
        with patch.object(install, 'run') as commands, patch.object(install.subprocess, 'run',
                return_value=SimpleNamespace(stdout='  Candidate: 2.58.4\n', returncode=0)):
            install.apt_install(['ros-humble-librealsense2', 'ros-humble-librealsense2-tools'])
        self.assertIn('ros-humble-librealsense2-tools', commands.call_args_list[-1].args[0])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cfg = json.loads((ROOT / 'robot.json').read_text())
        self.cfg['drivers'] = dict(mode='reuse', setup_files=[])
        self.cfg['cameras'].update(left_serial='111111111111', right_serial='222222222222')
        self.home = self.path / 'grg'
        self.home.mkdir()
        self.config = install.local_config(self.cfg, 'x86', 'humble', self.home)
        self.account = SimpleNamespace(pw_name='grg', pw_dir=str(self.home), pw_uid=os.geteuid(), pw_gid=os.getegid())
        self.inventory = {name: dict(prefix='/opt/ros/humble', version='test', share='/opt/ros/humble/share/' + name)
                          for name in (*probe.CORE_PACKAGES, *probe.DRIVERS)}

    def test_apt_reuse_never_installs_builds_or_creates_workspace(self):
        with patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'build_orbbec', side_effect=AssertionError('build called')), \
             patch.object(install, 'probe_drivers', return_value=self.inventory):
            self.assertIsNone(install.prepare_drivers(self.config, self.cfg, self.home, self.account))
        self.assertFalse((self.home / 'orb_ros2').exists())
        self.assertEqual(self.config['validated_packages'], self.inventory)

    def test_foreign_workspace_is_not_touched_in_reuse(self):
        workspace = self.home / 'orb_ros2'
        workspace.mkdir()
        original = workspace / 'my-source.txt'
        original.write_text('existing source')
        self.cfg['drivers']['setup_files'] = [str(workspace / 'install/setup.bash')]
        config = install.local_config(self.cfg, 'x86', 'humble', self.home)
        with patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'build_orbbec', side_effect=AssertionError('build called')), \
             patch.object(install, 'probe_drivers', return_value=self.inventory):
            install.prepare_drivers(config, self.cfg, self.home, self.account)
        self.assertEqual(original.read_text(), 'existing source')
        self.assertEqual(list(workspace.iterdir()), [original])

    def test_fresh_install_still_installs_and_builds(self):
        self.cfg['drivers']['mode'] = 'install'
        config = install.local_config(self.cfg, 'x86', 'humble', self.home)
        with patch.object(install, 'apt_install') as apt, \
             patch.object(install, 'build_orbbec', return_value=self.home / 'orb_ros2') as build, \
             patch.object(install, 'probe_drivers', return_value=self.inventory):
            install.prepare_drivers(config, self.cfg, self.home, self.account)
        apt.assert_called_once()
        build.assert_called_once()

    def test_multiple_overlays_are_loaded_in_order_without_shell_interpolation(self):
        base = self.path / 'ros/humble'
        base.mkdir(parents=True)
        (base / 'setup.bash').write_text('export ROS_DISTRO=humble\nexport ORDER=base\n')
        first = self.path / 'first overlay.bash'
        second = self.path / 'second $(touch MUST_NOT_EXIST).bash'
        first.write_text('export ORDER="$ORDER:first"\n')
        second.write_text('export ORDER="$ORDER:second"\n')
        config = dict(self.config, driver_setup_files=[str(first), str(second)])
        with patch.object(install, 'ROS_ROOT', self.path / 'ros'):
            output = install.run_in_ros(config, ['/usr/bin/python3', '-c',
                         'import os; print(os.environ["ORDER"]); print(os.environ["ROS_DISTRO"])'], capture=True)
        self.assertEqual(output.splitlines()[-2:], ['base:first:second', 'humble'])
        self.assertFalse(Path('/tmp/MUST_NOT_EXIST').exists())

    def test_missing_setup_fails_before_command(self):
        config = dict(self.config, driver_setup_files=[str(self.path / 'missing/setup.bash')])
        with patch.object(install.subprocess, 'run', side_effect=AssertionError('command executed')):
            with self.assertRaisesRegex(ValueError, '环境文件不存在'):
                install.run_in_ros(config, ['true'])

    def test_invalid_or_mixed_configuration_rejected(self):
        for drivers in (dict(mode='wrong', setup_files=[]),
                        dict(mode='reuse', setup_files=['relative/setup.bash']),
                        dict(mode='reuse', setup_files=['/tmp/setup\n.bash']),
                        dict(mode='reuse', setup_files=['/a', '/a']),
                        dict(mode='install', setup_files=['/a'])):
            with self.subTest(drivers=drivers), self.assertRaises(ValueError):
                install.driver_options(dict(self.cfg, drivers=drivers))
        with patch.dict(os.environ, {'ROS_DISTRO': 'jazzy'}, clear=True), self.assertRaises(ValueError):
            probe.check_distro(self.config)
        with patch.dict(os.environ, {'ROS_DISTRO': 'humble', 'AMENT_PREFIX_PATH': '/opt/ros/jazzy'}, clear=True), self.assertRaises(ValueError):
            probe.check_distro(self.config)

    def test_launch_arguments_check_real_parameter_names(self):
        names = probe.argument_names("Arguments:\n    'camera_name':\n        text\n    'depth_module.color_profile':\n")
        self.assertEqual(names, {'camera_name', 'depth_module.color_profile'})
        required = probe.required_arguments(ROOT / 'x86/start_d405.sh')
        self.assertIn('depth_module.color_profile', required)
        self.assertNotIn('rgb_camera.color_profile', required)
        self.assertIn('serial_no', required)

    def test_probe_rejects_missing_launch_argument_and_changed_prefix(self):
        shares = {}
        for name in self.inventory:
            share = self.path / 'share' / name
            share.mkdir(parents=True)
            (share / 'package.xml').write_text('<package><version>1.0</version></package>')
            shares[name] = share
        for name, (launch, _) in probe.DRIVERS.items():
            (shares[name] / 'launch').mkdir()
            (shares[name] / 'launch' / launch).write_text('# fake launch fixture')
        fake_ament = SimpleNamespace(get_package_prefix=lambda n: '/opt/ros/humble',
                                     get_package_share_directory=lambda n: str(shares[n]))
        class Meta(type):
            _TYPE_SUPPORT = object()
            @classmethod
            def __import_type_support__(cls):
                pass
        class Message(metaclass=Meta):
            pass
        modules = {'ament_index_python': SimpleNamespace(), 'ament_index_python.packages': fake_ament,
                   'rclpy': SimpleNamespace(), 'rclpy.utilities': SimpleNamespace(get_rmw_implementation_identifier=lambda: 'rmw_fastrtps_cpp'),
                   'sensor_msgs': SimpleNamespace(), 'sensor_msgs.msg': SimpleNamespace(Image=Message, CameraInfo=Message)}
        def help_output(command, **kwargs):
            script = probe.DRIVERS[command[2]][1]
            names = probe.required_arguments(ROOT / 'x86' / script)
            if command[2] == 'realsense2_camera':
                names.remove('depth_module.color_profile')
            return SimpleNamespace(returncode=0, stdout='\n'.join(f"    '{name}':" for name in names))
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'ROS_DISTRO': 'humble'}, clear=True), \
             patch.object(probe.shutil, 'which', return_value='/fake/ros2'), \
             patch.object(probe.subprocess, 'run', side_effect=help_output):
            with self.assertRaisesRegex(ValueError, 'depth_module.color_profile'):
                probe.inspect(self.config, script_dir=ROOT / 'x86')
            config = dict(self.config, validated_packages={'orbbec_camera': {'prefix': '/different/install'}})
            with self.assertRaisesRegex(ValueError, '加载路径发生变化'):
                probe.inspect(config, quick=True)

    def test_other_camera_processes_rejected_but_known_service_allowed(self):
        proc = self.path / 'proc'
        directory = proc / '999999'
        directory.mkdir(parents=True)
        (directory / 'cmdline').write_bytes(b'/usr/bin/python3\0/opt/ros/humble/bin/ros2\0launch\0orbbec_camera\0gemini2.launch.py\0')
        (directory / 'cgroup').write_text('0::/system.slice/orbbec-camera.service\n')
        install.check_camera_processes(proc)
        (directory / 'cgroup').write_text('0::/system.slice/custom-camera.service\n')
        with self.assertRaisesRegex(ValueError, '999999'):
            install.check_camera_processes(proc)

    def test_gemini_empty_serial_is_omitted_from_launch_command(self):
        executable = self.path / 'ros2'
        executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o755)
        launch = (ROOT / 'x86/start_orbbec_camera.sh').read_text().split('serial_args=()', 1)[1]
        for serial in ('', 'ABC123'):
            output = subprocess.check_output(['bash', '-c', 'serial=' + shlex.quote(serial) + '; serial_args=()' + launch],
                     env=dict(os.environ, PATH=str(self.path) + os.pathsep + os.environ['PATH']), text=True)
            args = output.splitlines()
            self.assertNotIn('serial_number:=', args)
            self.assertEqual('serial_number:=ABC123' in args, bool(serial))

    def test_reuse_main_deploys_services_without_driver_or_usb_changes(self):
        path = self.path / 'robot.json'
        path.write_text(json.dumps(self.cfg))
        writes = {}
        class FakeWriter:
            backup = self.path / 'backup'
            changed = []
            def put(self, dest, data, mode=0o644):
                writes[str(dest)] = data
                self.changed.append(str(dest))
        def fake_command(args, **kwargs):
            if args[:2] == ['systemctl', 'cat']:
                return SimpleNamespace(returncode=0)
            if args[0] == 'dpkg-query':
                return SimpleNamespace(returncode=1, stdout='')  # source installations need not have APT entries
            raise AssertionError(args)
        with patch.object(sys, 'argv', ['install.py', 'x86', '--config', str(path)]), \
             patch.object(install.os, 'geteuid', return_value=0), \
             patch.object(install, 'host_preflight', return_value=(self.account, self.home, 'humble', self.config)), \
             patch.object(install, 'probe_drivers', return_value=self.inventory), \
             patch.object(install, 'check_camera_processes'), \
             patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'build_orbbec', side_effect=AssertionError('build called')), \
             patch.object(install, 'usb_rules', side_effect=AssertionError('udev called')), \
             patch.object(install, 'Writer', FakeWriter), \
             patch.object(install, 'run') as commands, \
             patch.object(install.subprocess, 'run', side_effect=fake_command):
            install.main()
        config = json.loads(writes[str(self.home / 'ros2_camera_link/config.json')])
        self.assertEqual(config['driver_mode'], 'reuse')
        self.assertIn('orbbec_camera', config['validated_packages'])
        self.assertIn('/etc/systemd/system/realsense-dual.service', writes)
        self.assertFalse(any('udev' in name for name in writes))
        self.assertTrue(any(call.args[0][:2] == ['systemctl', 'restart'] for call in commands.call_args_list))

    def test_head_only_allows_pending_serials_and_does_not_start_wrists(self):
        self.cfg['cameras'].update(left_serial='', right_serial='')
        path = self.path / 'robot.json'
        path.write_text(json.dumps(self.cfg))
        writes = {}
        class FakeWriter:
            backup = self.path / 'backup'
            changed = []
            def put(self, dest, data, mode=0o644):
                writes[str(dest)] = data
                self.changed.append(str(dest))
        def prepared(config, *unused):
            config['validated_packages'] = self.inventory
            return None
        with patch.object(sys, 'argv', ['install.py', 'x86', '--head-only', '--config', str(path)]), \
             patch.object(install.os, 'geteuid', return_value=0), \
             patch.object(install, 'host_preflight', return_value=(self.account, self.home, 'humble', self.config)), \
             patch.object(install, 'prepare_drivers', side_effect=prepared), \
             patch.object(install, 'check_camera_processes'), \
             patch.object(install, 'Writer', FakeWriter), \
             patch.object(install, 'run') as commands, \
             patch.object(install.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='')):
            install.main()
        self.assertNotIn('/etc/systemd/system/realsense-dual.service', writes)
        self.assertIn('/etc/systemd/system/orbbec-camera.service', writes)
        config = json.loads(writes[str(self.home / 'ros2_camera_link/config.json')])
        self.assertFalse(config['wrist_enabled'])
        actions = [call.args[0] for call in commands.call_args_list]
        self.assertIn(['systemctl', 'disable', 'realsense-dual.service'], actions)
        self.assertFalse(any('realsense-dual.service' in cmd for cmd in actions if cmd[:2] in (['systemctl', 'enable'], ['systemctl', 'restart'])))
        with self.assertRaises(ValueError):
            install.load_config(path, 'x86')

    def test_check_mode_fails_before_writes_when_driver_invalid(self):
        path = self.path / 'robot.json'
        path.write_text(json.dumps(self.cfg))
        with patch.object(sys, 'argv', ['install.py', 'x86', '--config', str(path), '--check']), \
             patch.object(install, 'host_preflight', return_value=(self.account, self.home, 'humble', self.config)), \
             patch.object(install, 'probe_drivers', side_effect=ValueError('incompatible driver')), \
             patch.object(install, 'Writer', side_effect=AssertionError('write attempted')), \
             patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'run', side_effect=AssertionError('mutation attempted')):
            with self.assertRaisesRegex(ValueError, 'incompatible driver'):
                install.main()

    def test_successful_check_never_mutates_system(self):
        path = self.path / 'robot.json'
        path.write_text(json.dumps(self.cfg))
        with patch.object(sys, 'argv', ['install.py', 'x86', '--config', str(path), '--check']), \
             patch.object(install, 'host_preflight', return_value=(self.account, self.home, 'humble', self.config)), \
             patch.object(install, 'probe_drivers', return_value=self.inventory), \
             patch.object(install, 'check_camera_processes'), \
             patch.object(install, 'Writer', side_effect=AssertionError('write attempted')), \
             patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'run', side_effect=AssertionError('mutation attempted')):
            install.main()

    def test_list_cameras_in_reuse_does_not_install_sdk(self):
        path = self.path / 'robot.json'
        path.write_text(json.dumps(self.cfg))
        with patch.object(sys, 'argv', ['install.py', 'x86', '--config', str(path), '--list-cameras']), \
             patch.object(install.os, 'geteuid', return_value=0), \
             patch.object(install, 'host_preflight', return_value=(self.account, self.home, 'humble', self.config)), \
             patch.object(install, 'run_in_ros') as sourced, \
             patch.object(install, 'apt_install', side_effect=AssertionError('APT called')), \
             patch.object(install, 'Writer', side_effect=AssertionError('write attempted')):
            install.main()
        sourced.assert_called_once()
        self.assertIn('rs-enumerate-devices', sourced.call_args.args[1][-1])

    def test_source_entry_handles_no_overlay_and_multiple_overlays(self):
        base = self.path / 'base.bash'
        base.write_text('export ROS_DISTRO=humble\nexport ORDER=base\n')
        first = self.path / 'first.bash'
        second = self.path / 'second.bash'
        first.write_text('export ORDER="$ORDER:first"\n')
        second.write_text('export ORDER="$ORDER:second"\n')
        (self.path / 'profile.xml').write_text('<dds/>')
        (self.path / 'driver_probe.py').write_text('import os\nassert os.environ["ROS_DISTRO"] == "humble"\n')
        code = (ROOT / 'common/env.sh').read_text().replace('"/opt/ros/${values[0]}/setup.bash"', shlex.quote(str(base)))
        (self.path / 'env.sh').write_text(code)
        for setups, expected in (([], 'base'), ([str(first), str(second)], 'base:first:second')):
            lines = ['humble', '12', str(self.path / 'profile.xml'), *setups]
            (self.path / 'transport.py').write_text('print(' + repr('\n'.join(lines)) + ')\n')
            # The entry must not inherit an unrelated caller array or invoke ros2 itself.
            command = ('values=(wrong stale data); before=$$; ros2() { return 99; }; '
                       'source "$1" subscriber || exit; [[ $$ == "$before" ]] || exit 88; '
                       'printf "RESULT=%s:%s:%s\\n" "$ORDER" "$ROS_DOMAIN_ID" "$RMW_IMPLEMENTATION"')
            completed = subprocess.run(['bash', '-c', command, 'test-source', str(self.path / 'env.sh')],
                                       env=dict(os.environ, ROS_DISTRO='humble'), text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn(f'RESULT={expected}:12:rmw_fastrtps_cpp', completed.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
