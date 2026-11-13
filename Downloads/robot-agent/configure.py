"""Run with sudo. Asks only robot ID, device name, SSH username/password."""
import getpass
import grp
import json
import os
import tempfile
import uuid
from pathlib import Path
from agent import load_config


def ask(label, default):
    return input(f'{label} [{default}]: ').strip() or default


def main():
    if os.geteuid() != 0:
        raise SystemExit('请执行 sudo python3 /opt/robot-agent/configure.py')
    path = Path('/etc/robot-agent.json')
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    print('回车保留默认值。SSH 密码仅保存在本设备，不上传、不用于自动登录。')
    c = dict(server_url=old.get('server_url', 'http://10.1.47.61:8080'),
             device_id=old.get('device_id', 'dev-' + uuid.uuid4().hex),
             ssh_port=old.get('ssh_port', 22), heartbeat_seconds=old.get('heartbeat_seconds', 10))
    c['robot_id'] = ask('机器人编号（英文/数字）', old.get('robot_id', 'robot-001'))
    c['device_name'] = ask('设备名称（支持中文）', old.get('device_name', '一号机器人-Thor'))
    c['ssh_user'] = ask('本设备 SSH 用户名', old.get('ssh_user', os.environ.get('SUDO_USER', 'ubuntu')))
    c['ssh_password'] = getpass.getpass('SSH 密码（隐藏输入，回车保留旧值，首次可留空）: ') or old.get('ssh_password', '')
    fd, temporary = tempfile.mkstemp(prefix='.robot-agent-', dir='/etc')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(c, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        try:
            load_config(temporary)
        except (ValueError, TypeError, AttributeError):
            raise SystemExit('配置无效，原文件未修改。请检查编号、名称和 SSH 用户名格式。') from None
        os.chown(temporary, 0, grp.getgrnam('robot-agent').gr_gid)
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('已保存 /etc/robot-agent.json；改名时自动保留原设备 ID。')


if __name__ == '__main__':
    main()
