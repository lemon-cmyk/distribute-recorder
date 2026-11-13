# x86：三台摄像头

首次配置只需填写包根目录的 `robot.json`，执行 `bash deploy.sh x86`。本页用于查看参数、话题和日常维护。

## 驱动模式

| 模式 | 自动处理 |
|---|---|
| `drivers.mode = install` | 安装 APT 依赖、编译随包 Orbbec、安装 D405、配置 USB 权限 |
| `drivers.mode = reuse` | 加载指定工作空间和基础 ROS，检查已有驱动；跳过 APT、编译、源码与 USB 权限修改 |

复用时在 `drivers.setup_files` 中填写已有环境脚本的绝对路径，例如 `/home/grg/orbbec_ws/install/setup.bash`、`/home/grg/realsense_ws/install/setup.bash`。按依赖顺序填写，后加载的工作空间优先；APT 驱动无需单独填写。不要混用 Humble 与 Jazzy 工作空间，也不要依赖交互式 `.bashrc` 中才有的 SDK 环境变量；必要的 SDK 环境应包含在你填写的环境脚本中。

复用检查用 ROS 包索引解析真实路径，加载 Fast DDS 与图像消息类型支持，并通过 `ros2 launch ... --show-args` 检查本包所用参数，不执行相机 launch。参数缺失、路径不可读或 ROS 版本混用时停止，不会自动升级驱动。root 部署时还会用配置的普通用户检查工作空间可读性。实际 SDK 动态库、USB 打开、流模式和帧率仍需启流验收。

正式部署与 `--check` 都会检查可识别的其他相机进程。同名 `orbbec-camera.service`、`realsense-dual.service` 可由本包接管；其他进程会提示 PID 并停止部署，需要你先停止原进程和禁用冲突的旧自启动服务。进程名检查不是对所有自定义容器或程序的完整占用检测。

两种模式都会替换本包管理的启动/通信配置、启用自启动并启动相机；旧配置文件会备份。复用模式保留已有驱动版本，但采用本包列出的相机参数。

## 安装模式封装的配置

| 项目 | 安装方式 / 设置 |
|---|---|
| Gemini 2 驱动 | 使用 `sources/OrbbecSDK_ROS2-2.9.3-snapshot.tar.gz`，在本机编译 |
| D405 驱动 | 安装 `ros-humble-librealsense2`、`ros-humble-realsense2-camera`、`ros-humble-realsense2-description`；仓库提供独立的 `ros-humble-librealsense2-tools` 时一并安装，否则使用 SDK 主包中的工具 |
| 依赖 | 针对随包固定源码列出完整 APT 依赖；不需要手动 rosdep init/update |
| USB 权限 | 安装 Orbbec 官方规则、D405 的 video 组规则；将配置用户加入 video 组 |
| 左右绑定 | 用 `left_serial`、`right_serial` 绑定设备，启动时添加 RealSense 所需的序列号前缀 |
| 通信 | 同一个 Domain；发布端使用机内有线 UDP + SHM，保留回环 UDP 供普通用户本机订阅；显式配置单播发现地址，支持相机组件加载服务 |
| 开机启动 | `orbbec-camera.service` 与 `realsense-dual.service` 均以 root 运行 |

安装模式不会从网络拉取浮动的 Orbbec 分支。D405 的 APT 版本可能随软件仓库变化；两种模式均检查实际加载的启动文件是否声明所需参数，并保存包版本及路径。复用不限制 Orbbec 必须是 2.9.3，但不保证任意 SDK/驱动版本的运行兼容性。

## 相机参数和话题

| 相机 | 彩色 | 深度 | 其他 |
|---|---|---|---|
| Gemini 2 | 640×480、30 fps、RGB | 取流设为 640×400、30 fps、Y16；保留对齐，预期输出 640×480 | IR、点云关闭 |
| 左/右 D405 | 640×480、30 fps、RGB8 | 640×480、30 fps、Z16 | IR、IMU、点云、深度对齐关闭 |

D405 彩色流使用 `depth_module.color_profile`；没有套用其他型号的 `rgb_camera.color_profile`。参数写在 `start_d405.sh` 中，双相机由 `start_dual_d405.sh` 管理。Gemini 2 参数写在 `start_orbbec_camera.sh` 中。

| 相机 | 彩色话题 | 深度话题 |
|---|---|---|
| 头部 | `/camera/color/image_raw` | `/camera/depth/image_raw` |
| 左腕 | `/wrist_cameras/left/color/image_raw` | `/wrist_cameras/left/depth/image_rect_raw` |
| 右腕 | `/wrist_cameras/right/color/image_raw` | `/wrist_cameras/right/depth/image_rect_raw` |

对应的 `color/camera_info`、`depth/camera_info` 也加入了有线接收配置。帧率是请求值，实际输出需实测；本包没有把 Gemini 2 偶发降到 15 Hz 的原因视为已经解决。

## 日常使用

root 或 grg 在同一终端执行：

```bash
source /home/grg/ros2_camera_link/receiver_env.sh
ros2 topic hz /wrist_cameras/left/color/image_raw
```

换一个话题名即可检查右腕或头部，深度同理。SDK 设备序列号可用以下命令查看（无需运行相机服务）：

```bash
source /opt/ros/humble/setup.bash
rs-enumerate-devices -s
```

查看、停止、恢复三台相机：

```bash
systemctl status orbbec-camera.service realsense-dual.service --no-pager
sudo systemctl stop orbbec-camera.service realsense-dual.service
sudo systemctl start orbbec-camera.service realsense-dual.service
```

需要重新启流时执行 `sudo systemctl restart orbbec-camera.service realsense-dual.service`。查看日志：

```bash
journalctl -u orbbec-camera.service -u realsense-dual.service -n 100 --no-pager
```

## 设备上文件的位置

- `/home/grg/orb_ros2/`：安装模式的 Gemini 2 源码和产物；复用模式继续使用原工作空间，不创建或覆盖此目录。
- `/home/grg/ros2_camera_link/`：启动脚本、订阅入口、设备配置、XML 生成器及安装版本记录。
- `/etc/systemd/system/`：两项相机服务和 `ros2-camera-buffers.service`。
- `/etc/udev/rules.d/`：安装模式写入 Orbbec 与 D405 规则；复用模式保留现有规则。

若更换普通用户名，生成的 HOME 路径随 `robot.json` 中的 `user` 自动变化，但复用的 `setup_files` 绝对路径需自行填写。安装模式遇到非本包的 `orb_ros2` 会停止；复用模式不要求它有本包标记。被替换的配置文件会备份到 `/var/backups/ros2-camera-topic/`。

安装模式加入 video 组后，普通用户需重新登录才能直接访问 USB；订阅图像本身不需要 USB 权限。复用模式以原驱动的 USB 访问已经配置好为前提。新设备必须填写实际序列号，左右不依据 USB 端口或枚举顺序自动猜测。

参数依据：[RealSense 官方启动文件](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_camera/launch/rs_launch.py)；Gemini 2 参数以随包 2.9.3 源码为准。
