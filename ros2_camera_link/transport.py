#!/usr/bin/env python3
"""Generate a process-specific plain Topic DDS profile; never alter IPs or routes."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
NS = 'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'


def read_json_command(command):
    return json.loads(subprocess.check_output(command, text=True))


def select_wired(config, links, routes):
    expected = str(ipaddress.IPv4Address(config['internal_ip']))
    selected = []
    for link in links:
        if expected in [a.get('local') for a in link.get('addr_info', [])]:
            selected.append(link)
    if len(selected) != 1:
        raise ValueError('Configured internal IPv4 is not uniquely assigned locally: ' + expected)
    link = selected[0]
    if 'UP' not in link.get('flags', []) or 'LOWER_UP' not in link.get('flags', []):
        raise ValueError('Internal link is not up: ' + link['ifname'])
    if (Path('/sys/class/net') / link['ifname'] / 'wireless').exists():
        raise ValueError('Internal camera interface must not be Wi-Fi')
    for route in routes:
        if route.get('dev') != link['ifname'] or (route.get('prefsrc') or route.get('src')) != expected:
            raise ValueError('Peer route does not use the configured internal interface/address')
    return link['ifname'], expected


def read_topics():
    topics = []
    for line in (ROOT / 'topics-wired.txt').read_text().splitlines():
        topic = line.split('#', 1)[0].strip()
        if not topic:
            continue
        if not topic.startswith('/') or any(c.isspace() for c in topic):
            raise ValueError('Expected a full ROS topic name: ' + topic)
        if topic not in topics:
            topics.append(topic)
    return topics


def build_xml(role, config, internal_ip, addresses, topics, use_shm):
    dds = ET.Element('dds', xmlns=NS)
    p = ET.SubElement(dds, 'profiles')
    descriptors = ET.SubElement(p, 'transport_descriptors')
    udp = ET.SubElement(descriptors, 'transport_descriptor')
    ET.SubElement(udp, 'transport_id').text = 'camera_udp'
    ET.SubElement(udp, 'type').text = 'UDPv4'
    allowed = [internal_ip, '127.0.0.1'] if role == 'publisher' else addresses
    whitelist = ET.SubElement(udp, 'interfaceWhiteList')
    for address in dict.fromkeys(allowed):
        ET.SubElement(whitelist, 'address').text = address
    ET.SubElement(udp, 'maxInitialPeersRange').text = '64'
    for name in ['sendBufferSize', 'receiveBufferSize']:
        ET.SubElement(udp, name).text = '8388608'
    if use_shm:
        shm = ET.SubElement(descriptors, 'transport_descriptor')
        ET.SubElement(shm, 'transport_id').text = 'camera_shm'
        ET.SubElement(shm, 'type').text = 'SHM'
        ET.SubElement(shm, 'segment_size').text = '67108864'
    part = ET.SubElement(p, 'participant', profile_name='plain_topic_' + role, is_default_profile='true')
    rtps = ET.SubElement(part, 'rtps')
    ET.SubElement(rtps, 'useBuiltinTransports').text = 'false'
    transports = ET.SubElement(rtps, 'userTransports')
    ET.SubElement(transports, 'transport_id').text = 'camera_udp'
    if use_shm:
        ET.SubElement(transports, 'transport_id').text = 'camera_shm'
    builtin = ET.SubElement(rtps, 'builtin')
    for name in ['readerHistoryMemoryPolicy', 'writerHistoryMemoryPolicy']:
        ET.SubElement(builtin, name).text = 'PREALLOCATED_WITH_REALLOC'
    peers = ET.SubElement(builtin, 'initialPeersList')
    for address in dict.fromkeys(config['peers'] + ['127.0.0.1']):
        v4 = ET.SubElement(ET.SubElement(peers, 'locator'), 'udpv4')
        ET.SubElement(v4, 'address').text = address
    # Retain standard multicast discovery for unrelated robot processes.
    multicast = ET.SubElement(builtin, 'metatrafficMulticastLocatorList')
    v4 = ET.SubElement(ET.SubElement(multicast, 'locator'), 'udpv4')
    ET.SubElement(v4, 'address').text = '239.255.0.1'
    # An explicit multicast list does not supply the unicast discovery locators.
    # Launch's component-loading service also needs discovery on the publisher.
    locators = ET.SubElement(builtin, 'metatrafficUnicastLocatorList')
    for address in dict.fromkeys(allowed):
        v4 = ET.SubElement(ET.SubElement(locators, 'locator'), 'udpv4')
        ET.SubElement(v4, 'address').text = address
    # Keep automatic data locators when SHM is enabled; UDP-only defaults suppress SHM.
    if role == 'subscriber':
        if not use_shm:
            locators = ET.SubElement(rtps, 'defaultUnicastLocatorList')
            for address in addresses:
                v4 = ET.SubElement(ET.SubElement(locators, 'locator'), 'udpv4')
                ET.SubElement(v4, 'address').text = address
        if config['device_role'] == 'receiver':
            for topic in topics:
                reader = ET.SubElement(p, 'subscriber', profile_name=topic)
                v4 = ET.SubElement(ET.SubElement(ET.SubElement(reader, 'unicastLocatorList'), 'locator'), 'udpv4')
                ET.SubElement(v4, 'address').text = internal_ip
                ET.SubElement(reader, 'historyMemoryPolicy').text = 'PREALLOCATED_WITH_REALLOC'
    writer = ET.SubElement(p, 'publisher', profile_name='plain_topic_writer', is_default_profile='true')
    ET.SubElement(ET.SubElement(ET.SubElement(writer, 'qos'), 'publishMode'), 'kind').text = 'ASYNCHRONOUS'
    ET.SubElement(writer, 'historyMemoryPolicy').text = 'PREALLOCATED_WITH_REALLOC'
    reader = ET.SubElement(p, 'subscriber', profile_name='plain_topic_reader', is_default_profile='true')
    ET.SubElement(reader, 'historyMemoryPolicy').text = 'PREALLOCATED_WITH_REALLOC'
    ET.indent(dds, space='  ')
    return ET.tostring(dds, encoding='utf-8', xml_declaration=True) + b'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['publisher', 'subscriber'])
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    if args.role == 'publisher' and (os.geteuid() != 0 or config['device_role'] != 'publisher'):
        raise ValueError('Camera publisher profile is restricted to root on x86')
    domain = config['domain_id']
    if not isinstance(domain, int) or not 0 <= domain <= 232:
        raise ValueError('Invalid ROS Domain ID')
    for peer in config['peers']:
        ipaddress.IPv4Address(peer)
    links = read_json_command(['ip', '-j', '-4', 'address', 'show'])
    routes = [read_json_command(['ip', '-j', '-4', 'route', 'get', peer])[0] for peer in config['peers']]
    interface, address = select_wired(config, links, routes)
    distro = config.get('ros_distro', 'auto')
    available = [p.parent.name for p in Path('/opt/ros').glob('*/setup.bash')]
    if distro == 'auto':
        if len(available) != 1:
            raise ValueError('Set ros_distro explicitly; found ' + ','.join(available))
        distro = available[0]
    if distro not in available:
        raise ValueError('Configured ROS installation is missing: ' + distro)
    addresses = ['127.0.0.1']
    for link in links:
        if 'UP' in link.get('flags', []):
            for item in link.get('addr_info', []):
                value = item.get('local')
                if value and value not in addresses:
                    addresses.append(value)
    use_shm = os.geteuid() == 0
    data = build_xml(args.role, config, address, addresses, read_topics(), use_shm)
    runtime = Path('/tmp') / ('ros2-camera-topic-' + str(os.geteuid()))
    runtime.mkdir(mode=0o700, exist_ok=True)
    metadata = runtime.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise ValueError('Unsafe profile cache directory: ' + str(runtime))
    target = runtime / (args.role + ('-shm.xml' if use_shm else '-udp.xml'))
    fd, temporary = tempfile.mkstemp(dir=runtime, prefix='profile-')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    os.replace(temporary, target)
    # Header plus ordered ROS overlay setup files; no child shell is opened.
    print(distro)
    print(domain)
    print(target)
    setups = config.get('driver_setup_files')
    if setups is None:  # Compatibility with a runtime generated by the earlier kit.
        workspace = config.get('orbbec_workspace')
        setups = [str(Path(workspace) / 'install/local_setup.bash')] if workspace else []
    for setup in setups:
        print(setup)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('Topic transport configuration failed: ' + str(error))
