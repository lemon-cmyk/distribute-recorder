#!/usr/bin/env python3
"""Raise only the camera socket buffer ceilings, preserving larger values."""
import os
from pathlib import Path
import json
import subprocess

if os.geteuid() != 0:
    raise SystemExit('Administrator privileges required for buffer ceilings')
before = {}
for key in ['net.core.rmem_max', 'net.core.wmem_max']:
    value = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', key], text=True))
    before[key] = value
    if value < 8388608:
        subprocess.run(['/usr/sbin/sysctl', '-q', '-w', key + '=8388608'], check=True)
snapshot = Path('/var/lib/ros2-camera-link/buffer-before.json')
snapshot.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
if not snapshot.exists():
    snapshot.write_text(json.dumps(before, indent=2) + '\n')
