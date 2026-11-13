# 机器人客户端（无上报令牌版）

支持 Ubuntu 22.04 / 24.04，x86 和 Thor ARM64，Python 3.10+，不需要 pip。
服务器默认 http://10.1.47.61:8080，必须已更新为聊天中提供的无令牌服务端版本。网页登录保留。

## 安装

将压缩包上传到设备主目录：

```bash
tar -xzf ~/robot-agent-no-token.tar.gz -C ~
cd ~/robot-agent
sudo bash install.sh
```

ZIP 包解压后进入 robot-agent 文件夹，同样执行 sudo bash install.sh。
若提示缺少依赖，执行 sudo apt update 和 sudo apt install -y python3 iproute2，然后重试。

配置向导只询问：机器人编号、设备名称、SSH 用户名、SSH 密码。
同一机器人中的 x86 和 Thor 使用相同机器人编号，如 robot-001，名称分别填“一号机器人-x86”“一号机器人-Thor”。
机器人编号仅支持英文、数字、下划线、点和短横线，最长 64 字符；设备名称支持中文，最长 32 字符。
SSH 用户名是当前设备的实际登录用户名；SSH 密码隐藏输入，可留空。

唯一设备 ID 首次自动生成，改名时保持不变。不要跨设备复制已生成的 /etc/robot-agent.json，否则记录会互相覆盖。克隆系统后需为新设备手动更换该文件中的 device_id。

## 验证

```bash
sudo -u robot-agent python3 /opt/robot-agent/agent.py --once
sudo systemctl status robot-agent --no-pager
sudo journalctl -u robot-agent -n 50 --no-pager
```

出现 Heartbeat OK 后在网页查看设备。HTTP 401 表示服务器仍是需要令牌的旧版，需先更新服务器。
客户端每 10 秒上报，约每 2 秒检查 IP，断网后自动重试，开机自启无需登录桌面。
上报以访问服务器所选路由的源 IPv4 为准；有线优先时会显示有线 IP，不修改网络设置。
网页使用原有 role 字段显示中文设备名称。

## 修改名字或 SSH 信息

```bash
sudo python3 /opt/robot-agent/configure.py
sudo systemctl restart robot-agent
```

回车保留旧值。服务器地址、SSH 端口默认值为 10.1.47.61:8080、22，可手动编辑 /etc/robot-agent.json 后重启。
server_url 支持 HTTP/HTTPS 数字 IPv4，不支持域名或 IPv6；HTTPS 使用系统证书验证。

SSH 密码按原样仅保存于本机 /etc/robot-agent.json（root:robot-agent，640），不上传、不打印到日志、不显示在网页，也不自动登录或修改系统密码。
网页只提供 SSH 命令，连接时按提示手动输入密码。SSH 是否可达仍取决于设备 SSH 服务和网络路由。

只查看无密码上报信息：

```bash
sudo -u robot-agent python3 /opt/robot-agent/agent.py --print-payload
```

停用：sudo systemctl disable --now robot-agent。
离线测试：在解压目录执行 python3 -m unittest -v test_agent.py。
