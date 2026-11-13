#!/usr/bin/env bash
set -euo pipefail
if [[ "$EUID" -ne 0 ]]; then
  echo '请执行 sudo bash install.sh'
  exit 1
fi
for tool in python3 ip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo '先安装依赖：sudo apt update && sudo apt install -y python3 iproute2'
    exit 1
  fi
done
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! id -u robot-agent >/dev/null 2>&1; then
  useradd --system --user-group --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin robot-agent
fi
install -d -m 755 /opt/robot-agent
install -m 644 "$source_dir/agent.py" "$source_dir/configure.py" /opt/robot-agent/
python3 /opt/robot-agent/configure.py
install -m 644 "$source_dir/robot-agent.service" /etc/systemd/system/robot-agent.service
systemctl daemon-reload
systemctl enable robot-agent
systemctl restart robot-agent
systemctl status robot-agent --no-pager
echo '验证上报：sudo -u robot-agent python3 /opt/robot-agent/agent.py --once'
