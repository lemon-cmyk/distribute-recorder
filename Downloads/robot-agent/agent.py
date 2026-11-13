"""Robot IP heartbeat client. Python 3.10+, Ubuntu, no pip dependencies."""
import argparse
import ipaddress
import json
import logging
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

LOG = logging.getLogger('robot-agent')
STOP = threading.Event()


def load_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('robot_id', 'device_id'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', c.get(key, '')):
            raise ValueError(f'{key}: use 1-64 English letters, numbers, _, . or -')
    name = c.get('device_name', '')
    if not isinstance(name, str) or not 1 <= len(name) <= 32 or any(ord(x) < 32 for x in name):
        raise ValueError('device_name: use 1-32 printable characters')
    c.setdefault('server_url', 'http://10.1.47.61:8080')
    c['server_url'] = c['server_url'].rstrip('/')
    u = urlsplit(c['server_url'])
    if u.scheme not in ('http', 'https') or u.username or u.password or u.path or u.query or u.fragment:
        raise ValueError('server_url: use an HTTP(S) origin with a numeric IPv4')
    c['server_ip'] = str(ipaddress.IPv4Address(u.hostname))
    u.port
    if not re.fullmatch(r'[a-z_][a-z0-9_-]{0,30}\$?', c.get('ssh_user', '')):
        raise ValueError('ssh_user: invalid Linux username')
    if not isinstance(c.get('ssh_password', ''), str):
        raise ValueError('ssh_password must be text')
    for key, default, low, high in (('ssh_port', 22, 1, 65535), ('heartbeat_seconds', 10, 3, 20)):
        c.setdefault(key, default)
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(f'{key}: integer from {low} to {high} required')
    return c


def ip_json(*args):
    p = subprocess.run(['ip', '-j', '-4', *args], capture_output=True,
                       text=True, check=True, timeout=3)
    return json.loads(p.stdout)


def network_info(c):
    route = ip_json('route', 'get', c['server_ip'])[0]
    interface = route['dev']
    source = str(ipaddress.IPv4Address(route.get('prefsrc') or route.get('src')))
    addr = ipaddress.IPv4Address(source)
    if addr.is_loopback or addr.is_unspecified or addr.is_link_local or addr.is_multicast:
        raise ValueError('No usable management IPv4')
    ips = sorted({a['local'] for entry in ip_json('address', 'show', 'dev', interface)
                  for a in entry.get('addr_info', [])
                  if a.get('family') == 'inet' and a.get('scope') == 'global'})
    if source not in ips:
        raise ValueError('Network changed while inspecting IP; retrying')
    return dict(management_ip=source, interface=interface, ips=ips[:32])


def make_payload(c, network):
    # Explicit allowlist: SSH password and any old token never leave the device.
    return dict(robot_id=c['robot_id'], device_id=c['device_id'],
                role=c['device_name'], hostname=socket.gethostname()[:128],
                ssh_user=c['ssh_user'], ssh_port=c['ssh_port'], **network)


def send_report(c, payload):
    request = Request(c['server_url'] + '/api/heartbeat',
                      data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        if not 200 <= response.status < 300 or json.loads(response.read(4096)).get('ok') is not True:
            raise ValueError('Server did not acknowledge heartbeat')


def describe(error):
    if isinstance(error, HTTPError):
        return f'HTTP {error.code}' + ('; update server to token-free version' if error.code == 401 else '')
    if isinstance(error, URLError):
        return 'Cannot reach server (connection, timeout or TLS error)'
    if isinstance(error, subprocess.SubprocessError):
        return 'Cannot inspect network route/IP'
    return type(error).__name__  # Avoid printing configuration or secret values.


def run(c):
    previous = None
    due = retry = 0
    last_success = last_error = float('-inf')
    failures = 0
    LOG.info('Starting: device=%s name=%s', c['device_id'], c['device_name'])
    while not STOP.is_set():
        try:
            network = network_info(c)
            if network != previous:
                LOG.info('Network: %s %s', network['interface'], network['management_ip'])
                previous, due, retry = network, 0, 0
            now = time.monotonic()
            if now >= max(due, retry):
                send_report(c, make_payload(c, network))
                if failures or now - last_success >= 60:
                    LOG.info('Heartbeat OK: %s', network['management_ip'])
                    last_success = now
                failures = 0
                retry = 0
                due = time.monotonic() + c['heartbeat_seconds']
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as error:
            now = time.monotonic()
            if now >= retry:
                failures += 1
                retry = now + min(20, 2 ** min(failures, 5))
            if now - last_error >= 30:
                LOG.warning('%s; will retry', describe(error))
                last_error = now
        STOP.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='/etc/robot-agent.json')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--print-payload', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        c = load_config(args.config)
    except (OSError, ValueError, TypeError, AttributeError):
        LOG.error('Invalid configuration; check /etc/robot-agent.json')
        return 2
    if args.once or args.print_payload:
        try:
            payload = make_payload(c, network_info(c))
            if args.print_payload:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                send_report(c, payload)
                LOG.info('Heartbeat OK: device=%s IP=%s', c['device_id'], payload['management_ip'])
            return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as error:
            LOG.error('%s', describe(error))
            return 1
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    run(c)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
