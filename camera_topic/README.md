# 三摄像头纯 Topic 部署包

适用：一台 x86 连接头部 Gemini 2、左右各一台 RealSense D405；一台或多台 Thor/Orin 通过有线订阅。摄像头开机自启动，root 和普通用户使用同一个订阅入口。

前提：x86 已安装 Ubuntu 22.04 + ROS Humble；接收端已安装 Humble/Ubuntu 22.04 或 Jazzy/Ubuntu 24.04。设备的有线 IP、路由由网络维护人员预先配好。本包不设置 IP、网关或防火墙，也不更换内核；会配置下文注明的 UDP 缓冲上限。

## 首次部署：只做这四步

**1. 修改一份配置。** 打开本目录的 [robot.json](robot.json)：

- `domain_id`：填写本机器人使用的 Domain；当前的 `12` 仅是示例，各设备会自动使用同一值。
- `devices`：填写各设备已有的有线 IP；只有一个接收端时删除 `thor2`。接收端只有一个 ROS 版本可保留 `auto`。
- `left_serial`、`right_serial`：填写对应 D405 的 **Serial Number**，保留双引号。`gemini2_serial` 可留空，适用于只连接一台奥比中光相机。
- `user` 默认 `grg`；不要填写密码。`build_jobs` 默认 2，限制编译并行度。
- `drivers.mode`：新装驱动填 `install`；已有可用驱动填 `reuse`，并填写 `drivers.setup_files`，见下面示例。

不知道 D405 序列号时，先把本包放到 x86，执行 `bash deploy.sh x86 --list-cameras`。安装模式会安装枚举工具；复用模式只调用已有工具，找不到就提示，不自动安装。按实际左右填写配置，不要使用 ASIC Serial Number。左右尚未确认时，可留空并先执行 `bash deploy.sh x86 --head-only`：准备驱动，只启用头部自启动；确认后填入序列号，再运行不带 `--head-only` 的正常部署命令。

**2. 复制整个文件夹。** 将填写好的同一份包复制到 x86 和各接收端，建议目录统一为 `/home/grg/camera_topic`。必须保留整个目录结构。

**3. 在 x86 部署。** 以 grg 登录，执行：

```bash
cd /home/grg/camera_topic
bash deploy.sh x86
```

`install` 模式安装依赖、编译随包 Orbbec 2.9.3、安装 D405 驱动和 USB 规则；`reuse` 模式检查并加载已有驱动，跳过这些安装步骤。两种模式都会部署启动及通信配置、设置 UDP 缓冲上限，并启动及启用三台相机的开机自启动。

**4. 在接收端部署。** Thor No.1 执行：

```bash
cd /home/grg/camera_topic
bash deploy.sh thor1
```

第二台接收端执行 `bash deploy.sh thor2`。名称与 `robot.json` 的设备名称一致；Orin 也使用相同的接收端流程。接收端不需要安装相机驱动。

部署时 sudo 可能要求输入设备密码。**不需要手写 XML、大段 Python、export 或 systemd 文件。** 旧说明中的参数替换、缓冲设置、root/grg 共用入口已全部封装。

## 已有驱动：改这两项即可复用

例如 Orbbec 已编译在 `/home/grg/ros2_ws`，D405 已通过 APT 安装，将 `robot.json` 的 `drivers` 改为：

```json
"drivers": {
  "mode": "reuse",
  "setup_files": ["/home/grg/ros2_ws/install/setup.bash"]
}
```

如果 D405 也在源码工作空间中，再向数组加入它的 `install/setup.bash`。两个驱动在同一个工作空间只填一次；均通过 APT 安装则填空数组 `[]`。基础 `/opt/ros/humble/setup.bash` 自动加载，列表按顺序 source，后面的工作空间优先。

部署命令仍为 `bash deploy.sh x86`。可先执行 `bash deploy.sh x86 --check`，检查环境、实际驱动路径和启动参数；检查不会启停相机服务。复用模式不运行 APT、不编译、不改驱动源码、已有 USB 规则或用户组，缺少依赖或参数不兼容时会停止并说明原因。

复用的是驱动，启动参数和自启动由本包接管，使用下文的服务名。如果以前有其他名称的相机自启动服务，应先停止并禁用它；脚本不会擅自结束未知进程。已有同名相机服务会备份被替换的服务文件，并在部署时停止、更新和重新启动。

## 每次使用

x86 本机与 Thor/Orin、root 与 grg，均在准备运行订阅程序的终端执行同一个命令：

```bash
source /home/grg/ros2_camera_link/receiver_env.sh
```

它只配置当前终端，不创建新 shell、不自动订阅。随后自行运行程序或检查话题：

```bash
ros2 topic list --no-daemon --spin-time 5
ros2 topic hz /camera/color/image_raw
```

摄像头由 x86 的 `orbbec-camera.service` 和 `realsense-dual.service` 自动启动；正常使用不再手动 launch，也不需要另开 Topic/Action 转发服务。

## 目录与补充说明

| 内容 | 用途 |
|---|---|
| [x86/README.md](x86/README.md) | Gemini 2、D405 参数、话题、启停与日志 |
| [receiver/README.md](receiver/README.md) | 本机/跨机订阅路径与新增话题 |
| `x86/` | 三台相机启动脚本、服务模板、Orbbec 源码及许可证 |
| `common/` | 通信入口、动态 XML 生成器、话题清单、缓冲设置 |
| `scripts/install.py` | 部署内部实现，无需手动编辑 |
| [VALIDATION.md](VALIDATION.md) | 本包验证范围与已知限制 |

本目录只保留这一套三摄像头纯 Topic 方案，没有旧 Domain 预设。修改设备参数后，同步 `robot.json` 并在相关设备重新执行部署命令；x86 重新部署会重启相机。Wi-Fi IP 改变不需要改配置；已有订阅进程的环境不会被新的 source 追溯修改。

安装模式需要联网获取 APT 依赖，不是完全离线包；APT 可能随依赖安装更新已有相关包。后台更新占用安装锁时最多等待 180 秒，不删除锁或结束更新进程。复用模式在所需组件齐全时不联网安装。安装模式的 D405 来自 APT，复用模式使用已有版本；实际 ROS 包路径和版本记录在运行目录。安装完成和服务 active 不等于帧率验收通过；六路原始图像同时传输也不保证千兆网络能达到每路 30 Hz。
