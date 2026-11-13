# 客户端 UDP 版（Ubuntu 22.04 / 24.04，x86 / Thor）

固定目的地址默认 10.1.47.61:18080/UDP；网页仍是 http://10.1.47.61:8080。
请先升级服务器，再升级客户端。无令牌，无 pip 依赖。

## 安装 / 从旧 HTTP 客户端升级

将 robot-agent-udp.tar.gz 上传到设备主目录：

```bash
tar -xzf ~/robot-agent-udp.tar.gz -C ~
cd ~/robot-agent-udp
sudo bash install.sh
```

脚本会替换 /opt/robot-agent 内程序，更新原 robot-agent.service 并重启，因此不会同时启动两份客户端。
使用 ZIP 时先解压进入 robot-agent-udp 文件夹，再运行同样的安装脚本。
缺少依赖时先执行 sudo apt update 和 sudo apt install -y python3 iproute2。

向导仍只询问：机器人编号、设备名称、SSH 用户名、SSH 密码。旧设备 ID 会保留，旧 HTTP 地址自动迁移到默认 UDP 地址，不再使用 server_url 字段。
robot_id 仅支持英文、数字、下划线、点和短横线，如 G200-4（不支持 #）；设备名称支持中文，最长 32 字符。
同一机器人中的计算机填相同机器人编号，分别设置不同名称。
SSH 密码只按原样保存在本机 root:robot-agent、640 的配置文件中，不上传、不显示、不自动登录、不修改系统密码。

配置文件 /etc/robot-agent.json 新增 server_ip 和 server_port，默认 10.1.47.61 和 18080。SSH 端口默认 22，心跳间隔默认 10 秒。
如 IT 指定其他端口，手动编辑此文件并重启，同时修改服务器 UDP_PORT。
设备源端口由操作系统动态分配；目的 IP 和目的端口固定。不发送广播，不使用 HTTP 上报，也不通过公网服务。

## 验证

```bash
sudo -u robot-agent python3 /opt/robot-agent/agent.py --once
sudo systemctl status robot-agent --no-pager
sudo journalctl -u robot-agent -n 50 --no-pager
```

UDP sent (not acknowledged) 仅表示发送调用成功，必须在网页看到最后上报时间更新，才能确认服务器接收成功。UDP 接收端不回包。
--once 和常驻服务共用系统启动标识、使用单调时钟序号，一次性测试不会使常驻服务失效。
正常 10 秒上报一次，约 2 秒检查 IP，地址变化后尽快发送；断网自动重试。重启系统后自动运行，无需登录桌面。
管理 IP 来自访问服务器的路由，有线优先时上报有线 IP，不改变 WiFi、路由、SSH 配置。

```bash
# 修改名字（自动保留唯一设备 ID）
sudo python3 /opt/robot-agent/configure.py
sudo systemctl restart robot-agent
# 查看无密码上报内容
sudo -u robot-agent python3 /opt/robot-agent/agent.py --print-payload
# 停用
sudo systemctl disable --now robot-agent
```

不要复制已生成的配置文件到另一台计算机。克隆系统后为新设备手动设置不同 device_id，否则会覆盖同一条网站记录。
网页超时阈值仍为原配置（默认 45 秒）。在线仅表示收到心跳，SSH 可达性取决于实际网络和 SSH 服务。

报文版本为 1，UTF-8 JSON，最多 1200 字节；名称放在兼容字段 role 中。
boot_id 是本次系统启动的标识，seq 是单调递增序号。若网卡 IP 列表过长则只保留管理 IP，以满足长度限制。
