# 本机与 Thor/Orin 接收

接收端首次部署：填写同一份 `robot.json`，在对应设备执行 `bash deploy.sh thor1`，第二台执行 `bash deploy.sh thor2`。脚本只安装必要的 ROS 通信组件与本包配置，不安装相机驱动。

## 一个 source 入口

无论 x86、Thor/Orin，还是 root、grg，都执行：

```bash
source /home/grg/ros2_camera_link/receiver_env.sh
```

随后手动启动你自己的程序，也可以检查话题：

```bash
ros2 topic list --no-daemon --spin-time 5
ros2 topic hz /camera/color/image_raw
ros2 topic hz /wrist_cameras/left/color/image_raw
```

两个 hz 命令是分别运行的示例，第一个会一直运行，按 Ctrl+C 后再执行第二个，或另开已 source 的终端。兼容旧用法的 `source /home/grg/ros2_camera_link/run_receiver.sh` 与上述入口等价；不要用 `bash` 加载环境。

| 场景 | 本包配置的路径 |
|---|---|
| x86 root 订阅 root 相机进程 | 允许 SHM，保留 UDP；需运行时验证实际选路 |
| x86 grg 订阅 root 相机进程 | 关闭订阅端 SHM，通过本机 UDP，避开跨用户 SHM 权限问题 |
| Thor/Orin root 或 grg 接收相机 | 指定接收端有线地址，跨机图像通过有线 UDP |

## XML 在哪里

无需编辑 XML。`common/transport.py` 会依据本机 IP、角色和当前 UID 生成 XML，source 时打印文件路径：

- root：`/tmp/ros2-camera-topic-0/subscriber-shm.xml`。
- grg：`/tmp/ros2-camera-topic-<用户UID>/subscriber-udp.xml`。
- 相机服务：`/tmp/ros2-camera-topic-0/publisher-shm.xml`。

这些文件开机后可重新生成。网卡名按配置的有线 IP 自动识别，Wi-Fi 地址在 source 时读取；有线 IP 或 Domain 改变时修改包内 `robot.json` 并重新部署。脚本检查现有路由，但不会修改路由或替你配置网络。

订阅端保留其他活动网卡及标准发现机制，因此同 Domain、网络和 QoS 兼容的其他机器人话题仍可发现和订阅。相机发布进程只使用机内有线网卡和回环，不会因 Wi-Fi 地址变化主动改用 Wi-Fi 发送图像。可发现不等于任意网络、防火墙或自定义消息环境下均可订阅。

## 增加其他话题

若只想在程序中同时接收机械臂 state，source 后直接订阅即可；自定义消息类型还需 source 相应工作空间，订阅 QoS 要与发布端兼容。

若要把一个额外话题的接收地址也限定为有线，将其**完整 ROS 话题名**加入接收端 `/home/grg/ros2_camera_link/topics-wired.txt`，重新 source，再重新启动该订阅进程。也同步修改交接包 `common/topics-wired.txt`，避免下次部署覆盖。发布该话题的进程也必须能在有线网络与接收端发现、通信；本清单不会改动其他发布进程的网络设置。

新 source 只影响当前终端随后启动的进程，不能改变已经运行的 ROS 进程。发现列表异常时先使用 `--no-daemon` 排除旧 daemon 环境的影响。

实现依据：[rmw_fastrtps 官方 XML 话题配置说明](https://github.com/ros2/rmw_fastrtps/blob/humble/README.md)、[Fast DDS locators 文档](https://fast-dds.docs.eprosima.com/en/v2.6.11/fastdds/xml_configuration/common.html)。
