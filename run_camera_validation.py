#!/usr/bin/env python3
"""Bounded, read-only ROS subscriptions and host observations."""
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

role = sys.argv[1]
account = pwd.getpwnam('grg')
base = Path('/home/grg/camera_topic_deployment/domain14-20261109/validation')
base.mkdir(exist_ok=True)
os.chown(base, account.pw_uid, account.pw_gid)
all_topics = ['/camera/color/image_raw', '/camera/depth/image_raw',
              '/wrist_cameras/left/color/image_raw', '/wrist_cameras/left/depth/image_rect_raw',
              '/wrist_cameras/right/color/image_raw', '/wrist_cameras/right/depth/image_rect_raw']
for name, topics in [('all', all_topics), ('rgb', all_topics[::2])]:
    (base / (name + '.txt')).write_text('\n'.join(topics) + '\n')

def command(user, script):
    env = dict(HOME='/root' if user == 'root' else account.pw_dir, USER=user, LOGNAME=user,
               LANG='C.UTF-8', PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
    args = ['/usr/bin/env', '-i', *[k+'='+v for k,v in env.items()],
            'bash', '--noprofile', '--norc', '-c',
            'source /home/grg/ros2_camera_link/receiver_env.sh && ' + script]
    return args if user == 'root' else ['runuser', '-u', user, '--', *args]

def counters():
    return {p.name: {f.name: int(f.read_text()) for f in (p/'statistics').iterdir()}
            for p in Path('/sys/class/net').iterdir() if (p/'statistics').is_dir()}

cases = [('root', 'all'), ('grg', 'all')] if role == 'x86' else [('root','rgb'), ('grg','rgb'), ('grg','all')]
summary = []
for user, streams in cases:
    name = role + '-' + user + '-' + streams
    (base/'status.txt').write_text('RUNNING ' + name + '\n')
    before = counters()
    started = time.time()
    args = command(user, 'exec python3 /home/grg/probe_images.py --topics ' + str(base/(streams+'.txt'))
                   + ' --seconds 30 --warmup 5 --output ' + str(base/(name+'.json')))
    with (base/(name+'.log')).open('w') as log:
        try:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=50)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    after = counters()
    elapsed = time.time()-started
    delta = {n: {k: after[n][k]-before[n][k] for k in before[n]} for n in before if n in after}
    summary.append(dict(name=name, rc=rc, elapsed=elapsed, interface_counters=delta))
    (base/(role+'-summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
for user in ['root','grg']:
    r = subprocess.run(command(user, 'ros2 topic list --no-daemon --spin-time 3'),
                       capture_output=True, text=True, timeout=20)
    (base/(role+'-'+user+'-topics.txt')).write_text(r.stdout+r.stderr)
(base/'status.txt').write_text('COMPLETE\n')
