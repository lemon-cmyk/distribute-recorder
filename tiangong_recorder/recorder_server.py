from __future__ import annotations

# ============================================================================
# 当前文件：tiangong_recorder/recorder_server.py
#
# 【这个文件在整个项目中的职责】
#
# 这个文件不是直接负责“采相机、接机器人状态、保存 PKL”的地方，
# 它主要负责 Recorder 的“控制面”和 episode worker 的生命周期管理。
#
# 可以把整个项目粗略理解为：
#
#   scripts/start_recorder.sh
#           │
#           │ 启动
#           ▼
#   recorder_server.py          ← 当前文件
#           │
#           │ START
#           ▼
#   episode_worker.py
#           │
#           ├── ROS 2 三路相机
#           │
#           ├── ZMQ robot state/action
#           │
#           ▼
#   LiveEpisodeSynchronizer
#           │
#           │ 时间戳对齐
#           ▼
#      episode_data
#           │
#           ▼
#   write_episode_pickle()
#           │
#           ▼
#      最终 *.pkl
#
#
# 【当前文件主要处理的两类数据】
#
# 1. 控制请求：
#
#       客户端
#          │
#          │ START / STOP_SAVE / DISCARD / STATUS / PING / SHUTDOWN
#          ▼
#       ZeroMQ REP socket
#          │
#          ▼
#       RecorderServer
#
#
# 2. worker 生命周期消息：
#
#       run_episode_worker
#             │
#             │ ready / saving / saved / discarded / error
#             ▼
#       multiprocessing.Pipe
#             │
#             ▼
#       RecorderServer
#
#
# 【重要区别】
#
# RecorderServer 的 ZeroMQ REP socket：
#     是“控制面”，负责 START / STOP_SAVE / STATUS 等低频命令。
#
# episode_worker 内部还有另一个 ZeroMQ PULL socket：
#     是“数据面”，负责接收 x86 teleoperation process 发来的
#     robot state/action 数据。
#
# 两条链路不要混在一起理解。
# ============================================================================


# ============================================================================
# argparse
#
# 来源：
#   Python 标准库。
#
# 作用：
#   解析程序启动时传入的命令行参数。
#
# 当前文件中主要用于解析：
#
#       --config config/recorder.yaml
#
# 最终生成：
#
#       args.config
#
# 再传给：
#
#       RecorderConfig.from_yaml(args.config)
# ============================================================================
import argparse


# ============================================================================
# multiprocessing
#
# 来源：
#   Python 标准库。
#
# 作用：
#   用独立进程运行每一个 episode 的实际录制任务。
#
# 当前文件使用它创建：
#
#   1. multiprocessing.Process
#       → 真正执行 run_episode_worker()
#
#   2. multiprocessing.Pipe
#       → RecorderServer 和 worker 之间交换控制/状态消息
#
#
# 数据关系：
#
#       RecorderServer 主进程
#               │
#        parent_connection
#               │
#            Pipe
#               │
#         child_connection
#               │
#       episode worker 子进程
#
# 使用别名 mp，只是为了后面写 mp.Process / mp.get_context() 更简洁。
# ============================================================================
import multiprocessing as mp


# ============================================================================
# signal
#
# 来源：
#   Python 标准库。
#
# 作用：
#   接收操作系统发送给当前进程的信号。
#
# 当前文件处理：
#
#   SIGINT
#       常见来源：Ctrl+C
#
#   SIGTERM
#       常见来源：kill、systemd、Docker 等关闭操作
#
# 收到后不会立刻在 signal handler 中做复杂清理，
# 而只是：
#
#       server.running = False
#
# 随后 serve_forever() 自然退出，
# 最终由 finally -> server.close() 统一清理资源。
# ============================================================================
import signal


# ============================================================================
# time
#
# 来源：
#   Python 标准库。
#
# 当前文件主要使用两类时间：
#
#   time.monotonic()
#       用于 timeout / deadline。
#       它不会因为系统墙上时间被修改而跳变，适合计算超时。
#
#   time.time_ns()
#       返回当前墙上时间的纳秒表示。
#       当前主要用于 PING 回复中的 server_time_ns。
# ============================================================================
import time


# ============================================================================
# deque
#
# 来源：
#   Python 标准库 collections。
#
# 作用：
#   双端队列。
#
# 当前文件中用于：
#
#       self.recent = deque(maxlen=20)
#
# 只保存最近 20 个已经结束的 episode 结果，
# 避免历史状态无限增长。
# ============================================================================
from collections import deque


# ============================================================================
# dataclass
#
# 来源：
#   Python 标准库 dataclasses。
#
# 作用：
#   自动生成 __init__ 等样板代码，
#   适合把几个彼此相关的变量封装成一个数据对象。
#
# 当前文件中用于定义：
#
#       WorkerHandle
#
# WorkerHandle 用于保存一个 worker 在 RecorderServer 主进程侧的：
#
#       episode_id
#       Process
#       Pipe connection
#       status
#       state_addr
#       last_message
# ============================================================================
from dataclasses import dataclass


# ============================================================================
# Connection
#
# 来源：
#   Python 标准库：
#
#       multiprocessing.connection.Connection
#
# 它是 multiprocessing.Pipe() 返回的通信端点类型。
#
# 当前项目中：
#
#       parent_connection, child_connection = Pipe(...)
#
# parent_connection：
#       留在 RecorderServer 主进程。
#
# child_connection：
#       传给 run_episode_worker 子进程。
#
# Server → Worker：
#       STOP_SAVE
#       DISCARD
#
# Worker → Server：
#       ready
#       saving
#       saved
#       discarded
#       error
# ============================================================================
from multiprocessing.connection import Connection

from typing import Deque, Dict


# ============================================================================
# zmq / PyZMQ
#
# 来源：
#   第三方 Python 库 pyzmq，是 ZeroMQ 的 Python 接口。
#
# ZeroMQ：
#   是一种消息通信库。
#
# 当前项目里至少存在两种 ZeroMQ 通信：
#
#   ① 当前 RecorderServer：
#
#       REP socket
#
#       负责控制命令：
#           START
#           STOP_SAVE
#           STATUS
#           ...
#
#
#   ② episode_worker：
#
#       PULL socket
#
#       负责实际 robot state/action 数据。
#
#
# 因此：
#
#       recorder_server.py
#           ≈ 控制通道
#
#       episode_worker.py 的 state_socket
#           ≈ 高频数据通道
# ============================================================================
import zmq


# ============================================================================
# RecorderConfig
#
# 定义来源：
#
#       tiangong_recorder/config.py
#
#
# 【config.py 的职责】
#
# RecorderConfig 是一个 frozen dataclass，
# 用来集中保存 Recorder 服务运行需要的配置。
#
# 其中包括：
#
#   control_bind_addr
#       RecorderServer 控制 socket 的 bind 地址。
#
#   advertise_host
#       episode worker 向 x86 客户端公布数据端口时使用的主机地址。
#
#   state_port_min / state_port_max
#       episode worker 动态选择 state PULL socket 端口的范围。
#
#   output_dir
#       episode PKL 的输出目录。
#
#   camera_topics
#       head / left_wrist / right_wrist 三路 ROS 2 相机 topic。
#
#   image_width / image_height
#       预期相机图像尺寸。
#
#   startup_timeout_s
#   camera_stale_timeout_s
#   state_drain_timeout_s
#   tail_wait_timeout_s
#       worker 各阶段的 timeout。
#
#   state_receive_hwm
#       worker ZMQ PULL socket 的接收 High Water Mark。
#
#
# 【配置最外层来源】
#
# 项目默认启动脚本：
#
#       scripts/start_recorder.sh
#
# 最终执行：
#
#       python3 -m tiangong_recorder.recorder_server \
#           --config config/recorder.yaml
#
# 因此通常数据链是：
#
#       config/recorder.yaml
#              │
#              ▼
#       RecorderConfig.from_yaml()
#              │
#              ▼
#         RecorderConfig
#              │
#              ▼
#       RecorderServer(config)
#              │
#              ▼
#       run_episode_worker(..., config, ...)
#
# Server 和 Worker 因此共享同一份运行配置。
# ============================================================================
from .config import RecorderConfig


# ============================================================================
# run_episode_worker
#
# 定义来源：
#
#       tiangong_recorder/episode_worker.py
#
#
# 【episode_worker.py 的职责】
#
# RecorderServer 本身不直接录制 episode。
#
# 每次客户端发送 START 后，
# RecorderServer 会启动一个新的 multiprocessing.Process：
#
#       target=run_episode_worker
#
# 真正的数据采集逻辑发生在这个 worker 中。
#
#
# run_episode_worker 内部主要做：
#
#   1. 创建 LiveEpisodeSynchronizer
#
#   2. 创建 ROS 2 EpisodeCameraNode
#
#   3. 订阅：
#
#          head
#          left_wrist
#          right_wrist
#
#      三路 sensor_msgs/Image。
#
#   4. 通过 image_decoder.py 中的 decode_ros_image()
#      把 ROS Image 转换成带 timestamp 的 CameraFrame。
#
#   5. 创建 ZeroMQ PULL state_socket，
#      从 x86 teleoperation process 接收 robot state/action。
#
#   6. 将：
#
#          CameraFrame
#              │
#              ├──────────────┐
#              │              │
#          robot state        │
#              │              │
#              └──────┬───────┘
#                     ▼
#            LiveEpisodeSynchronizer
#
#      根据 _capture_time_ns 对齐。
#
#   7. STOP_SAVE 后得到：
#
#          synchronizer.episode_data
#
#   8. 通过 dataset_writer.py 中的 write_episode_pickle()
#      写入 config.output_dir。
#
#
# 【worker 怎么和当前 RecorderServer 通信】
#
# 通过 multiprocessing.Pipe：
#
#       RecorderServer
#             │
#             │ STOP_SAVE / DISCARD
#             ▼
#       run_episode_worker
#             │
#             │ ready / saving / saved / error
#             ▼
#       RecorderServer
#
#
# 所以整个项目可以粗略概括成：
#
#       recorder_server.py
#           负责“管一次录制任务”
#
#       episode_worker.py
#           负责“真正执行一次录制任务”
# ============================================================================
from .episode_worker import run_episode_worker


# ============================================================================
# protocol.py
#
# 定义来源：
#
#       tiangong_recorder/protocol.py
#
#
# 【这个文件的职责】
#
# 统一定义 Recorder 控制面的协议。
#
# 当前协议版本：
#
#       PROTOCOL_VERSION = 1
#
#
# 控制命令：
#
#       START
#       STOP_SAVE
#       DISCARD
#       STATUS
#       PING
#       SHUTDOWN
#
#
# validate_episode_id()
#
#   用途：
#       检查 episode_id。
#
#   允许：
#       字母
#       数字
#       .
#       _
#       -
#
#   长度最多 128。
#
#
# ok()
#
#   用途：
#       构造统一的成功响应，例如：
#
#       {
#           "ok": True,
#           "protocol_version": 1,
#           ...
#       }
#
#
# error()
#
#   用途：
#       构造统一的失败响应，例如：
#
#       {
#           "ok": False,
#           "protocol_version": 1,
#           "message": "...",
#           ...
#       }
#
#
# 整体流向：
#
#       客户端 JSON request
#               │
#               ▼
#       RecorderServer._handle_request()
#               │
#       根据 protocol.py 中的 type 常量分发
#               │
#               ▼
#       _start / _stop_save / _status / ...
#               │
#               ▼
#           ok() / error()
#               │
#               ▼
#       socket.send_json(reply)
#               │
#               ▼
#             客户端
# ============================================================================
from .protocol import (
    DISCARD,
    PING,
    PROTOCOL_VERSION,
    SHUTDOWN,
    START,
    STATUS,
    STOP_SAVE,
    error,
    ok,
    validate_episode_id,
)


# ============================================================================
# WorkerHandle
#
# 定义来源：
#   当前 recorder_server.py。
#
#
# 【它是什么】
#
# 它不是 worker 本身，
# 而是 RecorderServer 主进程中用来“持有/描述一个 worker”的数据对象。
#
#
# 字段来源：
#
# episode_id
#   ← START request["episode_id"]
#
# process
#   ← mp.Process(target=run_episode_worker)
#
# connection
#   ← Pipe() 的 parent_connection
#
# status
#   ← Server 设置：
#         starting
#         recording
#         finalizing
#         discarding
#
#     或由 worker 的 message["type"] 更新：
#         saving
#         saved
#         discarded
#         error
#
# state_addr
#   ← worker 返回的 ready 消息：
#
#       {
#           "type": "ready",
#           "state_addr": ...
#       }
#
# last_message
#   ← worker 最近通过 Pipe 发回的完整消息。
#
#
# WorkerHandle 创建后主要会进入：
#
#       self.active
#           当前唯一正在 recording 的 worker
#
# 或：
#
#       self.finishing[episode_id]
#           已 STOP_SAVE / DISCARD、
#           但仍在收尾的 worker
#
# 最终由 _poll_workers() 持续更新和清理。
# ============================================================================
@dataclass
class WorkerHandle:
    episode_id: str
    process: mp.Process
    connection: Connection
    status: str
    state_addr: str | None = None
    last_message: dict | None = None


# ============================================================================
# RecorderServer
#
# 定义来源：
#   当前 recorder_server.py。
#
#
# 【核心职责】
#
# RecorderServer 是整个 Recorder 服务的控制器。
#
# 它负责：
#
#   ① 对外提供 ZeroMQ REP 控制接口
#
#   ② 解析 START / STOP_SAVE / STATUS 等请求
#
#   ③ 创建 episode worker
#
#   ④ 管理 worker 生命周期
#
#   ⑤ 通过 multiprocessing.Pipe 与 worker 通信
#
#   ⑥ 保存 active / finishing / recent 状态
#
#
# 它不直接：
#
#   × 订阅 ROS 相机
#   × 对齐相机和 state
#   × 写 PKL
#
# 这些工作主要由 episode_worker.py 负责。
# ============================================================================
class RecorderServer:
    def __init__(self, config: RecorderConfig):

        # ====================================================================
        # 局部功能块：验证并保存 RecorderConfig
        #
        # 输入来源：
        #
        #   config
        #       ← main()
        #       ← RecorderConfig.from_yaml(args.config)
        #       ← config/recorder.yaml
        #
        #
        # 当前处理：
        #
        #   config.validate()
        #
        # 会检查：
        #
        #   camera_topics 是否正好包含
#       head / left_wrist / right_wrist
        #
        #   image_width / image_height 是否 > 0
        #
        #   state port 范围是否合法
        #
        #   timeout 是否为正数
        #
        #
        # self.config：
        #
        #   将验证后的配置保存到 RecorderServer。
        #
        #
        # 后续去向：
        #
        #   self.config
        #       ├── _start()
        #       │      └── run_episode_worker(...)
        #       │
        #       ├── startup_timeout_s
        #       │
        #       └── serve_forever()
        #              └── control_bind_addr
        # ====================================================================
        config.validate()
        self.config = config

        # ====================================================================
        # 局部功能块：创建 ZeroMQ 控制服务
        #
        # zmq.Context：
        #
        #   ZeroMQ socket 所属的运行上下文。
        #   一般一个进程创建一个 Context，
        #   再从 Context 创建一个或多个 socket。
        #
        #
        # zmq.REP：
        #
        #   ZeroMQ Request-Reply 模式中的 Reply 端。
        #
        #   对端通常是 REQ：
        #
        #       控制客户端
        #          REQ
        #           │
        #           │ request
        #           ▼
        #          REP
        #       RecorderServer
        #           │
        #           │ reply
        #           ▼
        #       控制客户端
        #
        #
        # 当前 socket 的输入来源：
        #
        #   serve_forever()
        #       ↓
        #   self.socket.recv_json()
        #
        #
        # 当前 socket 的输出：
        #
        #   reply
        #       ↓
        #   self.socket.send_json(reply)
        #
        #
        # zmq.LINGER = 0：
        #
        #   socket 关闭时不继续等待未发送完的消息，
        #   有利于程序退出阶段快速释放资源。
        #
        #
        # bind 地址来源：
        #
        #   config/recorder.yaml
        #       ↓
        #   RecorderConfig.control_bind_addr
        #       ↓
        #   config.control_bind_addr
        #
        # 当前默认配置为：
        #
        #   tcp://0.0.0.0:5560
        #
        # 表示监听本机所有网络接口的 5560 端口。
        # ====================================================================
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(config.control_bind_addr)

        # ====================================================================
        # 局部功能块：初始化 episode 生命周期容器
        #
        # self.active：
        #
        #   当前正在 recording 的 WorkerHandle。
        #
        #   项目当前设计一次只允许一个 active episode。
        #
        #
        # self.finishing：
        #
        #   key：
        #       episode_id
        #
        #   value：
        #       WorkerHandle
        #
        #   保存已经收到 STOP_SAVE / DISCARD，
        #   但 worker 还没有完全退出的 episode。
        #
        #
        # self.recent：
        #
        #   Deque[dict]
        #
        #   保存 worker 最近返回的最多 20 个终态消息：
        #
        #       saved
        #       discarded
        #       error
        #
        #   这样 worker 即使已经退出，
        #   STATUS 仍可以查询最近完成的 episode。
        #
        #
        # 三者形成：
        #
        #             START
        #               │
        #               ▼
        #            active
        #               │
        #        STOP_SAVE/DISCARD
        #               │
        #               ▼
        #           finishing
        #               │
        #         worker 结束
        #               │
        #               ▼
        #            recent
        # ====================================================================
        self.active: WorkerHandle | None = None
        self.finishing: Dict[str, WorkerHandle] = {}
        self.recent: Deque[dict] = deque(maxlen=20)

        # ====================================================================
        # 局部功能块：初始化 Server 主循环和 multiprocessing 上下文
        #
        # self.running：
        #
        #   serve_forever() 的循环开关。
        #
        #   初始：
        #       True
        #
        #   以下情况会改为 False：
        #
        #       SHUTDOWN 请求
        #       SIGINT
        #       SIGTERM
        #
        #
        # mp.get_context("spawn")：
        #
        #   获取使用 spawn 启动方式的 multiprocessing context。
        #
        # spawn 的基本含义：
        #
        #   创建全新的 Python 子进程，
        #   子进程重新导入相关 Python 模块，
        #   然后执行指定 target。
        #
        # 后续：
        #
        #   self.mp_context.Pipe()
        #
        #   self.mp_context.Process()
        #
        # 都从这个 context 创建。
        # ====================================================================
        self.running = True
        self.mp_context = mp.get_context("spawn")

    def _start(self, request: dict) -> dict:
        # ====================================================================
        # _start()
        #
        # 调用来源：
        #
        #   serve_forever()
        #       │
        #       ▼
        #   _handle_request(request)
        #       │
        #       │ request["type"] == START
        #       ▼
        #   _start(request)
        #
        #
        # request 来源：
        #
        #   控制客户端
        #       ↓
        #   ZeroMQ
        #       ↓
        #   self.socket.recv_json()
        #
        #
        # 函数目标：
        #
        #   启动一个新的 episode worker，
        #   等它初始化三路相机和 state socket，
        #   收到 ready 后把 state_addr 返回客户端。
        # ====================================================================

        # ====================================================================
        # 局部功能块：避免同时启动两个 active episode
        #
        # 输入来源：
        #
        #   self.active
        #       ← 之前成功启动的 WorkerHandle
        #
        #   self.active.process
        #       ← mp.Process(run_episode_worker)
        #
        #
        # 当前处理：
        #
        #   如果：
        #
        #       active 存在
        #       AND
        #       worker process 仍然存活
        #
        #   则拒绝新的 START。
        #
        #
        # 输出去向：
        #
        #   error(...)
        #       ↓
        #   _handle_request()
        #       ↓
        #   serve_forever()
        #       ↓
        #   socket.send_json(reply)
        #       ↓
        #   控制客户端
        # ====================================================================
        if self.active is not None and self.active.process.is_alive():
            return error(
                "another episode is already recording",
                status="busy",
                active_episode_id=self.active.episode_id,
            )

        # ====================================================================
        # 局部功能块：读取并校验 episode_id
        #
        # 输入：
        #
        #   request["episode_id"]
        #       ← 控制客户端 START 请求
        #
        #
        # validate_episode_id()
        #
        #   来源：
        #       protocol.py
        #
        #   作用：
        #       将值转换成 str，
        #       并检查字符和长度是否合法。
        #
        #
        # 输出：
        #
        #   episode_id
        #
        # 后续去向：
        #
        #   ├── run_episode_worker()
        #   ├── Process name
        #   ├── WorkerHandle
        #   └── 所有 reply / status 标识
        # ====================================================================
        episode_id = validate_episode_id(request.get("episode_id"))

        # ====================================================================
        # 局部功能块：创建 Server ↔ Worker 的双向 Pipe
        #
        # self.mp_context：
        #   来源于 __init__() 的 spawn multiprocessing context。
        #
        #
        # duplex=True：
        #
        #   Pipe 两边都可以 send / recv。
        #
        #
        # 返回：
        #
        #   parent_connection
        #       留在 RecorderServer 主进程
        #
        #   child_connection
        #       传给 run_episode_worker 子进程
        #
        #
        # 通信方向：
        #
        #   Server
        #     │
        #     │ STOP_SAVE / DISCARD
        #     ▼
        #   Worker
        #
        #   Worker
        #     │
        #     │ ready / saving / saved / error
        #     ▼
        #   Server
        # ====================================================================
        parent_connection, child_connection = self.mp_context.Pipe(duplex=True)

        # ====================================================================
        # 局部功能块：创建 episode worker 子进程
        #
        # Process：
        #   Python multiprocessing 中表示一个独立子进程的对象。
        #
        #
        # target：
        #
        #   run_episode_worker
        #
        #   定义于：
        #       episode_worker.py
        #
        #
        # args 三个变量来源：
        #
        #   episode_id
        #       ← 当前 START request
        #
        #   self.config
        #       ← RecorderServer.__init__()
        #       ← RecorderConfig.from_yaml()
        #       ← recorder.yaml
        #
        #   child_connection
        #       ← 上一个局部功能块创建的 Pipe
        #
        #
        # 当下面 process.start() 后：
        #
        #       recorder_server.py
        #
        #           process.start()
        #                │
        #                ▼
        #
        #       新 Python 子进程
        #
        #           run_episode_worker(
        #               episode_id,
        #               self.config,
        #               child_connection,
        #           )
        #
        #
        # worker 内进一步：
        #
        #   config.camera_topics
        #          │
        #          ▼
        #   EpisodeCameraNode
        #          │
        #          ▼
        #   ROS camera Image
        #          │
        #          ▼
        #   decode_ros_image()
        #          │
        #          ▼
        #   CameraFrame
        #          │
        #          ▼
        #   LiveEpisodeSynchronizer
        #
        #
        # 同时：
        #
        #   x86 robot state/action
        #          │
        #          ▼
        #   worker ZMQ PULL socket
        #          │
        #          ▼
        #   synchronizer.add_state()
        #
        #
        # 两路最终按 timestamp 对齐。
        # ====================================================================
        process = self.mp_context.Process(
            target=run_episode_worker,
            args=(episode_id, self.config, child_connection),
            name=f"episode-{episode_id}",
        )

        # ====================================================================
        # 局部功能块：真正启动 worker
        #
        # process 来源：
        #   上一步创建的 Process 对象。
        #
        # process.start()：
        #   这里才真正创建并运行新的子进程。
        #
        #
        # child_connection.close()：
        #
        #   子进程已经持有 child_connection，
        #   主进程不应该继续保留自己的这份 child 端描述符。
        #
        #   主进程以后只使用：
        #
        #       parent_connection
        #
        # 和 worker 通信。
        # ====================================================================
        process.start()
        child_connection.close()

        # ====================================================================
        # 局部功能块：创建 worker 的主进程侧状态句柄
        #
        # 输入：
        #
        #   episode_id
        #       ← request
        #
        #   process
        #       ← 刚启动的 worker Process
        #
        #   parent_connection
        #       ← Pipe 主进程端
        #
        #
        # status="starting"：
        #
        #   worker 已经启动，
        #   但是还没有确认三路 camera ready。
        #
        #
        # 输出：
        #
        #   handle
        #
        # 此时只是局部变量。
        #
        # 如果后面收到 worker 的 ready：
        #
        #       self.active = handle
        # ====================================================================
        handle = WorkerHandle(
            episode_id=episode_id,
            process=process,
            connection=parent_connection,
            status="starting",
        )

        # ====================================================================
        # 局部功能块：计算启动等待 deadline
        #
        # startup_timeout_s 来源：
        #
        #   config/recorder.yaml
        #       ↓
        #   RecorderConfig
        #       ↓
        #   self.config.startup_timeout_s
        #
        #
        # worker 自己内部也有 startup_timeout_s，
        # Server 这里额外 +2 秒，
        # 给 worker 报错和 Pipe 消息传递留出一定时间。
        #
        #
        # time.monotonic()：
        #
        #   用于计算相对时间，
        #   不受系统墙上时间调整影响。
        #
        #
        # deadline 去向：
        #
        #   下面 while 循环判断 worker 是否启动超时。
        # ====================================================================
        deadline = time.monotonic() + self.config.startup_timeout_s + 2.0

        while time.monotonic() < deadline:

            # ================================================================
            # 局部功能块：轮询 worker 的启动消息
            #
            # parent_connection：
            #   Pipe 的 Server 端。
            #
            # poll(0.1)：
            #   最多等待 0.1 秒看是否有 worker 消息。
            #
            # recv()：
            #   真正读取 worker 发来的 Python 对象。
            #
            #
            # message 来源：
            #
            #   episode_worker.py
            #
            # worker 启动成功时会发送：
            #
            #   {
            #       "type": "ready",
            #       "episode_id": ...,
            #       "state_addr": ...
            #   }
            #
            # worker 异常则可能发送：
            #
            #   {
            #       "type": "error",
            #       ...
            #   }
            #
            #
            # handle.last_message：
            #
            #   保存最新消息，
            #   之后 STATUS 查询时可以返回给客户端。
            # ================================================================
            if parent_connection.poll(0.1):
                message = parent_connection.recv()
                handle.last_message = message

                # ============================================================
                # 局部功能块：处理 worker ready
                #
                # ready 的产生条件在 episode_worker.py 中：
                #
                #   synchronizer.cameras_ready()
                #
                # 即三路 camera 都至少已经收到一帧。
                #
                #
                # state_addr 来源：
                #
                # worker 内：
                #
                #   state_socket.bind_to_random_port(
                #       "tcp://0.0.0.0",
                #       min_port=config.state_port_min,
                #       max_port=config.state_port_max,
                #   )
                #
                # 再构造：
                #
                #   tcp://{config.advertise_host}:{state_port}
                #
                #
                # state_addr 的用途：
                #
                #   返回给外部 teleoperation/client，
                #   后者之后可以向该 worker 的 ZMQ PULL socket
                #   发送当前 episode 的 robot state/action。
                #
                #
                # 状态迁移：
                #
                #   starting
                #       ↓
                #   recording
                #
                #
                # WorkerHandle 去向：
                #
                #   self.active = handle
                # ============================================================
                if message.get("type") == "ready":
                    handle.status = "recording"
                    handle.state_addr = message["state_addr"]
                    self.active = handle

                    print(
                        f"[READY] {episode_id} state={handle.state_addr}",
                        flush=True,
                    )

                    # ========================================================
                    # 输出：
                    #
                    # ok(...) 来源于 protocol.py。
                    #
                    # reply 去向：
                    #
                    #   _start()
                    #       ↓
                    #   _handle_request()
                    #       ↓
                    #   serve_forever()
                    #       ↓
                    #   self.socket.send_json(reply)
                    #       ↓
                    #   控制客户端
                    #
                    # state_addr 是其中最关键的输出之一，
                    # 客户端后续需要它发送 state 数据。
                    # ========================================================
                    return ok(
                        status="ready",
                        episode_id=episode_id,
                        state_addr=handle.state_addr,
                    )

                # ============================================================
                # 局部功能块：worker 启动阶段主动报告 error
                #
                # message 来源：
                #
                #   run_episode_worker 的 except Exception：
                #
                #       control_connection.send({
                #           "type": "error",
                #           ...
                #       })
                #
                #
                # 当前处理：
                #
                #   process.join()
                #       回收已经退出/正在退出的 worker。
                #
                #   parent_connection.close()
                #       关闭 Pipe。
                #
                #
                # message["message"]：
                #   worker 具体异常信息。
                #
                # 输出：
                #   error reply 返回控制客户端。
                # ============================================================
                if message.get("type") == "error":
                    process.join(timeout=1.0)
                    parent_connection.close()

                    return error(
                        message.get("message", "worker startup failed"),
                        status="error",
                        episode_id=episode_id,
                    )

            # ================================================================
            # 局部功能块：worker 没有发 ready/error 但已经死亡
            #
            # process.is_alive()：
            #
            #   multiprocessing.Process 提供的方法，
            #   判断操作系统子进程是否还存在。
            #
            #
            # 当前处理：
            #
            #   如果 worker 在启动过程中直接退出：
            #
            #       join
            #       close Pipe
            #       返回 error
            #
            # 避免 Server 一直等到 startup timeout。
            # ================================================================
            if not process.is_alive():
                process.join(timeout=0.1)
                parent_connection.close()

                return error(
                    "episode worker exited during startup",
                    status="error",
                    episode_id=episode_id,
                )

        # ====================================================================
        # 局部功能块：Server 等待 worker ready 超时
        #
        # 到达这里说明：
        #
        #   time.monotonic() >= deadline
        #
        # worker 仍没有成功进入 ready。
        #
        #
        # terminate()：
        #   强制终止子进程。
        #
        # join()：
        #   回收进程资源。
        #
        # close()：
        #   关闭 Pipe。
        #
        #
        # 输出：
        #
        #   startup timed out
        #
        # 返回控制客户端。
        # ====================================================================
        process.terminate()
        process.join(timeout=2.0)
        parent_connection.close()

        return error(
            "episode worker startup timed out",
            status="error",
            episode_id=episode_id,
        )

    def _stop_save(self, request: dict) -> dict:
        # ====================================================================
        # _stop_save()
        #
        # 调用来源：
        #
        #   request["type"] == STOP_SAVE
        #       ↓
        #   _handle_request()
        #       ↓
        #   _stop_save(request)
        #
        #
        # 作用：
        #
        #   告诉当前 worker：
        #
        #       “停止接收当前 episode，
        #        等 state 收齐、完成相机对齐，然后保存。”
        #
        #
        # 注意：
        #
        #   _stop_save() 自己并不写 PKL。
        #
        # 实际保存：
        #
        #   episode_worker.py
        #       ↓
        #   write_episode_pickle()
        # ====================================================================

        # ====================================================================
        # 局部功能块：读取并验证待停止的 episode
        #
        # episode_id：
        #   ← STOP_SAVE request["episode_id"]
        #
        #
        # self.active：
        #   ← START ready 后保存的 WorkerHandle。
        #
        #
        # 要求：
        #
        #   当前确实存在 active worker，
        #   并且 episode_id 和请求中的一致。
        #
        # 否则直接返回：
        #
        #   not_recording
        # ====================================================================
        episode_id = validate_episode_id(request.get("episode_id"))

        if self.active is None or self.active.episode_id != episode_id:
            return error(
                "episode is not currently recording",
                status="not_recording",
                episode_id=episode_id,
            )

        # ====================================================================
        # 局部功能块：读取 expected_state_count
        #
        # 来源：
        #
        #   STOP_SAVE request["expected_state_count"]
        #
        #
        # 含义：
        #
        #   客户端声明：
        #
        #       “这个 episode 我一共发送了多少条 robot state。”
        #
        #
        # 后续去向：
        #
        #   RecorderServer
        #       ↓ Pipe
        #   run_episode_worker
        #
        #
        # worker 会把它与：
        #
        #   received_state_count
        #
        # 比较。
        #
        # 如果数据还没收齐，
        # worker 会等待 state_drain_timeout_s；
        #
        # 如果实际收到的 state 超过声明数量，
        # worker 会报错。
        # ====================================================================
        expected_state_count = int(request.get("expected_state_count", -1))

        if expected_state_count < 0:
            return error("expected_state_count must be non-negative")

        # ====================================================================
        # 局部功能块：向 worker 发送 STOP_SAVE
        #
        # handle：
        #
        #   当前 self.active。
        #
        # handle.connection：
        #
        #   _start() 中 Pipe 创建的 parent_connection。
        #
        #
        # Server 发送：
        #
        #   {
        #       "type": STOP_SAVE,
        #       "expected_state_count": ...
        #   }
        #
        #
        # worker 中：
        #
        #   control_connection.poll()
        #       ↓
        #   control_connection.recv()
        #       ↓
        #   command_type == "STOP_SAVE"
        #
        # 然后设置：
        #
        #   stop_expected_count
        #   stop_received_monotonic
        #
        # 后续进入 state drain / 相机 tail 对齐 / 保存阶段。
        # ====================================================================
        handle = self.active
        handle.connection.send(
            {
                "type": STOP_SAVE,
                "expected_state_count": expected_state_count,
            }
        )

        # ====================================================================
        # 局部功能块：Server 侧状态从 active → finishing
        #
        # STOP_SAVE 发出后：
        #
        #   worker 并没有立即结束。
        #
        # 它还可能需要：
        #
        #   1. 等待最后几条 state 到达
        #   2. 等待对应 camera frame
        #   3. 完成对齐
        #   4. 写 PKL
        #
        #
        # 所以状态：
        #
        #   handle.status = "finalizing"
        #
        # 并进入：
        #
        #   self.finishing[episode_id]
        #
        #
        # self.active = None：
        #
        #   表示当前 episode 已经不再处于 recording 阶段。
        #
        # 后续：
        #
        #   _poll_workers()
        #
        # 继续跟踪这个 worker。
        # ====================================================================
        handle.status = "finalizing"
        self.finishing[episode_id] = handle
        self.active = None

        print(
            f"[STOP_SAVE] {episode_id} expected={expected_state_count}",
            flush=True,
        )

        # ====================================================================
        # 返回的是 finalizing 而不是 saved
        #
        # 因为此时只表示：
        #
        #   STOP_SAVE 已经成功下发给 worker。
        #
        # 真正 saved：
        #
        #   worker 完成 write_episode_pickle()
        #       ↓
        #   Pipe 发送 {"type": "saved", ...}
        #       ↓
        #   _poll_workers()
        #
        # 客户端可以通过 STATUS 查询后续结果。
        # ====================================================================
        return ok(
            status="finalizing",
            episode_id=episode_id,
            expected_state_count=expected_state_count,
        )

    def _discard(self, request: dict) -> dict:
        # ====================================================================
        # _discard()
        #
        # 作用：
        #
        #   放弃当前正在录制的 episode，
        #   不执行正常保存。
        #
        #
        # 调用链：
        #
        #   DISCARD request
        #       ↓
        #   _handle_request()
        #       ↓
        #   _discard()
        # ====================================================================

        # ====================================================================
        # 局部功能块：确认请求操作的是当前 active episode
        #
        # episode_id：
        #   ← DISCARD request
        #
        # self.active：
        #   ← 当前 recording WorkerHandle
        #
        # 如果不匹配：
        #   返回 not_recording。
        # ====================================================================
        episode_id = validate_episode_id(request.get("episode_id"))

        if self.active is None or self.active.episode_id != episode_id:
            return error(
                "episode is not currently recording",
                status="not_recording",
                episode_id=episode_id,
            )

        # ====================================================================
        # 局部功能块：向 worker 发送 DISCARD
        #
        # handle.connection：
        #   Server ↔ Worker Pipe。
        #
        #
        # worker 收到：
        #
        #   {"type": DISCARD}
        #
        # 后：
        #
        #   discard_requested = True
        #
        # 随后发送：
        #
        #   {
        #       "type": "discarded",
        #       "episode_id": ...
        #   }
        #
        # 然后 return。
        #
        #
        # Server 侧同时：
        #
        #   recording
        #       ↓
        #   discarding
        #
        #   active
        #       ↓
        #   finishing
        #
        # 后续仍由 _poll_workers() 等待 worker 真正退出。
        # ====================================================================
        handle = self.active
        handle.connection.send({"type": DISCARD})

        handle.status = "discarding"
        self.finishing[episode_id] = handle
        self.active = None

        print(f"[DISCARD] {episode_id}", flush=True)

        # ====================================================================
        # 此处返回 discarding，
        # 而不是 discarded。
        #
        # discarded 必须等 worker 真正处理完 DISCARD 后
        # 通过 Pipe 返回。
        # ====================================================================
        return ok(status="discarding", episode_id=episode_id)

    def _status(self, request: dict) -> dict:
        # ====================================================================
        # _status()
        #
        # 作用：
        #
        #   查询：
        #
        #   ① 一个具体 episode 的状态
        #
        # 或：
        #
        #   ② 整个 RecorderServer 当前状态
        #
        #
        # request 来源：
        #
        #   STATUS JSON request。
        # ====================================================================

        # ====================================================================
        # episode_id：
        #
        #   可选。
        #
        # 如果有：
        #   查询指定 episode。
        #
        # 如果没有：
        #   查询 RecorderServer 全局状态。
        # ====================================================================
        episode_id = request.get("episode_id")

        if episode_id:
            episode_id = validate_episode_id(episode_id)

            # ================================================================
            # 局部功能块：先查 active
            #
            # self.active：
            #
            #   当前正在 recording 的 worker。
            #
            #
            # 返回：
            #
            #   status
            #       当前 WorkerHandle.status
            #
            #   state_addr
            #       worker ready 时提供的数据面地址
            # ================================================================
            if self.active is not None and self.active.episode_id == episode_id:
                return ok(
                    status=self.active.status,
                    episode_id=episode_id,
                    state_addr=self.active.state_addr,
                )

            # ================================================================
            # 局部功能块：再查 finishing
            #
            # self.finishing：
            #
            #   保存正在 finalizing / discarding 等阶段的 worker。
            #
            #
            # details：
            #
            #   handle.last_message
            #
            #   即该 worker 最近一次通过 Pipe 返回的消息。
            #
            # 例如可能是：
            #
            #   saving
            #   error
            #   ...
            # ================================================================
            handle = self.finishing.get(episode_id)

            if handle is not None:
                return ok(
                    status=handle.status,
                    episode_id=episode_id,
                    details=handle.last_message,
                )

            # ================================================================
            # 局部功能块：最后查 recent
            #
            # self.recent：
            #
            #   worker 已经退出后，
            #   保存最近 20 个终态消息。
            #
            #
            # reversed(self.recent)：
            #
            #   从最新结果向旧结果查。
            #
            #
            # result：
            #
            #   来自 worker 的：
            #
            #       saved
            #       discarded
            #       error
            #
            #
            # status：
            #
            #   直接使用 result["type"]。
            # ================================================================
            for result in reversed(self.recent):
                if result.get("episode_id") == episode_id:
                    return ok(
                        status=result.get("type", "unknown"),
                        episode_id=episode_id,
                        details=result,
                    )

            # ================================================================
            # active / finishing / recent 都不存在
            #
            # → Server 不知道这个 episode。
            # ================================================================
            return error(
                "episode is unknown",
                status="unknown",
                episode_id=episode_id,
            )

        # ====================================================================
        # 局部功能块：没有指定 episode_id，返回服务器整体状态
        #
        # status：
        #
        #   self.active 存在
        #       → recording
        #
        #   否则
        #       → idle
        #
        #
        # active_episode_id：
        #   当前 recording episode。
        #
        #
        # finalizing_episode_ids：
        #
        #   当前 self.finishing 中所有 episode ID。
        #
        #   list(self.finishing)
        #
        # 对 dict 直接 list() 得到的是 key 列表，
        # 所以这里得到 episode_id 列表。
        # ====================================================================
        return ok(
            status="recording" if self.active else "idle",
            active_episode_id=self.active.episode_id if self.active else None,
            finalizing_episode_ids=list(self.finishing),
        )

    def _handle_request(self, request: dict) -> dict:
        # ====================================================================
        # _handle_request()
        #
        # 这是 RecorderServer 的“控制请求路由器”。
        #
        #
        # 输入来源：
        #
        #   serve_forever()
        #
        #       request = self.socket.recv_json()
        #
        #
        # 输出：
        #
        #   reply dict
        #
        # 再回到：
        #
        #   serve_forever()
        #       ↓
        #   self.socket.send_json(reply)
        #
        #
        # 数据流：
        #
        #               request
        #                  │
        #                  ▼
        #          _handle_request()
        #                  │
        #       ┌──────────┼────────────┐
        #       ▼          ▼            ▼
        #     START     STOP_SAVE      STATUS ...
        #       │          │            │
        #       ▼          ▼            ▼
        #    _start    _stop_save    _status
        #       │          │            │
        #       └──────────┴──────┬─────┘
        #                         ▼
        #                       reply
        # ====================================================================

        # ====================================================================
        # 局部功能块：协议版本检查
        #
        # request["protocol_version"]：
        #   ← 客户端。
        #
        #
        # PROTOCOL_VERSION：
        #   ← protocol.py。
        #
        #
        # 如果客户端没有提供 protocol_version：
        #
        #   默认按服务器当前版本处理。
        #
        #
        # 如果明确提供但版本不同：
        #
        #   返回 unsupported protocol version。
        # ====================================================================
        if int(request.get("protocol_version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
            return error("unsupported protocol version")

        # ====================================================================
        # 局部功能块：提取请求类型
        #
        # request_type：
        #   ← request["type"]
        #
        # 后续用于和 protocol.py 定义的常量比较。
        # ====================================================================
        request_type = request.get("type")

        # ====================================================================
        # START
        #
        # request 原样传给 _start()，
        # _start() 再读取 episode_id。
        # ====================================================================
        if request_type == START:
            return self._start(request)

        # ====================================================================
        # STOP_SAVE
        #
        # request 原样传给 _stop_save()，
        # 后者读取：
        #
        #   episode_id
        #   expected_state_count
        # ====================================================================
        if request_type == STOP_SAVE:
            return self._stop_save(request)

        # ====================================================================
        # DISCARD
        #
        # request → _discard()
        # ====================================================================
        if request_type == DISCARD:
            return self._discard(request)

        # ====================================================================
        # STATUS
        #
        # request → _status()
        # ====================================================================
        if request_type == STATUS:
            return self._status(request)

        # ====================================================================
        # PING
        #
        # 不需要访问 worker。
        #
        # 直接返回：
        #
        #   alive
        #   server_time_ns
        #
        #
        # time.time_ns()：
        #
        #   当前系统 wall-clock 的纳秒时间。
        #
        # 可用于：
        #
        #   服务存活检查
        #   时间相关诊断
        # ====================================================================
        if request_type == PING:
            return ok(
                status="alive",
                server_time_ns=time.time_ns(),
            )

        # ====================================================================
        # SHUTDOWN
        #
        # 不直接在这里：
        #
        #   kill worker
        #   close socket
        #
        # 而只设置：
        #
        #   self.running = False
        #
        #
        # 下一轮：
        #
        #   while self.running
        #
        # 条件失败，
        # serve_forever() 返回。
        #
        # main() finally：
        #
        #   server.close()
        #
        # 再统一清理资源。
        # ====================================================================
        if request_type == SHUTDOWN:
            self.running = False
            return ok(status="shutting_down")

        # ====================================================================
        # 未识别的 type
        #
        # → protocol error reply。
        # ====================================================================
        return error(f"unsupported request type: {request_type}")

    def _poll_workers(self) -> None:
        # ====================================================================
        # _poll_workers()
        #
        # 作用：
        #
        #   定期检查所有 worker：
        #
        #   1. 有没有通过 Pipe 发新消息
        #   2. worker Process 是否已经退出
        #
        #
        # 调用位置：
        #
        #   serve_forever() 每轮主循环首先调用：
        #
        #       self._poll_workers()
        #
        #
        # 所以即使外部没有控制请求，
        # Server 也会持续更新 worker 状态。
        # ====================================================================

        # ====================================================================
        # 局部功能块：构造当前需要轮询的 worker 列表
        #
        # 来源：
        #
        #   self.finishing.values()
        #       正在收尾的 workers
        #
        #   self.active
        #       当前正在 recording 的 worker
        #
        #
        # 输出：
        #
        #   handles
        #
        # 后面统一 for 循环处理。
        #
        #
        # 为什么先 list()：
        #
        #   因为后续处理过程中 self.finishing 可能被修改，
        #   复制为独立列表后再遍历更安全。
        # ====================================================================
        handles = list(self.finishing.values())

        if self.active is not None:
            handles.append(self.active)

        for handle in handles:
            try:
                # ============================================================
                # 局部功能块：把 worker Pipe 中当前已有的消息全部读完
                #
                # handle.connection：
                #
                #   WorkerHandle 中保存的 parent_connection。
                #
                #
                # poll()：
                #
                #   非阻塞检查是否有消息。
                #
                #
                # while：
                #
                #   一次循环可能已经积压了多条 worker 消息，
                #   因此不是只读取一条，
                #   而是一直读取到 Pipe 暂时为空。
                #
                #
                # message 来源：
                #
                #   run_episode_worker
                #
                # 可能包括：
                #
                #   ready
                #   saving
                #   saved
                #   discarded
                #   error
                # ============================================================
                while handle.connection.poll():
                    message = handle.connection.recv()

                    # ========================================================
                    # 局部功能块：同步 worker 的最新状态
                    #
                    # message：
                    #   ← worker Pipe
                    #
                    #
                    # handle.last_message：
                    #
                    #   保存完整消息，
                    #   给 STATUS details 使用。
                    #
                    #
                    # message_type：
                    #
                    #   message["type"]
                    #
                    #
                    # handle.status：
                    #
                    #   直接同步为该 message type。
                    #
                    # 例如：
                    #
                    #   finalizing
                    #       ↓ worker 发 saving
                    #   saving
                    #
                    #       ↓ worker 发 saved
                    #   saved
                    # ========================================================
                    handle.last_message = message
                    message_type = message.get("type", "unknown")
                    handle.status = message_type

                    # ========================================================
                    # 局部功能块：处理 worker 终态
                    #
                    # 三种终态：
                    #
                    #   saved
                    #       episode 已成功写入 PKL。
                    #
                    #   discarded
                    #       episode 已被放弃。
                    #
                    #   error
                    #       worker 发生异常。
                    #
                    #
                    # self.recent.append(message)：
                    #
                    #   把终态结果保存到最近历史中。
                    #
                    # 后续 worker Process 即使已经清理，
                    # STATUS 仍然能查到结果。
                    # ========================================================
                    if message_type in {"saved", "discarded", "error"}:
                        self.recent.append(message)

                        print(f"[{message_type.upper()}] {message}", flush=True)

                        # ====================================================
                        # 局部功能块：active worker 自己异常进入终态时，
                        # 把它从 active 转移到 finishing
                        #
                        # 正常 STOP_SAVE：
                        #
                        #   _stop_save() 已经提前：
                        #
                        #       active → finishing
                        #
                        #
                        # 但如果 recording worker 自己发生异常：
                        #
                        #   _stop_save() 没有执行，
                        #   self.active 仍可能指向它。
                        #
                        #
                        # 此时：
                        #
                        #   self.finishing[id] = handle
                        #   self.active = None
                        #
                        # 后续继续等待 Process 真正退出。
                        # ====================================================
                        if (
                            self.active is not None
                            and self.active.episode_id == handle.episode_id
                        ):
                            self.finishing[handle.episode_id] = handle
                            self.active = None

            # =================================================================
            # EOFError
            #
            # 来源：
            #
            #   Pipe 对端已经关闭时，
            #   recv() 可能抛 EOFError。
            #
            #
            # 这里不立刻做处理，
            # 因为下面还会通过：
            #
            #   handle.process.is_alive()
            #
            # 判断 worker Process 的真实状态并统一清理。
            # =================================================================
            except EOFError:
                pass

            # =================================================================
            # 局部功能块：worker Process 已经退出
            #
            # process：
            #   ← WorkerHandle.process
            #
            #
            # join(timeout=0.1)：
            #
            #   回收已经退出的子进程。
            # =================================================================
            if not handle.process.is_alive():
                handle.process.join(timeout=0.1)

                # ============================================================
                # 局部功能块：worker 没发任何消息就直接死掉
                #
                # handle.last_message is None：
                #
                #   表示 Server 从未收到该 worker 的状态消息。
                #
                #
                # 这种情况无法依赖 worker 自己发送 error，
                # 因此 Server 人工构造：
                #
                #   {
                #       "type": "error",
                #       "episode_id": ...,
                #       "message": "worker exited with code ..."
                #   }
                #
                #
                # exitcode：
                #
                #   multiprocessing.Process 提供的退出码。
                #
                #
                # 人工 error 最终进入：
                #
                #   handle.last_message
                #   self.recent
                #
                # 供 STATUS 查询。
                # ============================================================
                if handle.last_message is None:
                    handle.last_message = {
                        "type": "error",
                        "episode_id": handle.episode_id,
                        "message": (
                            f"worker exited with code {handle.process.exitcode}"
                        ),
                    }

                    self.recent.append(handle.last_message)

                # ============================================================
                # 局部功能块：清理 worker 的实时运行状态
                #
                # worker 已确认退出，
                # 因此：
                #
                #   1. 从 finishing 移除
                #
                #   2. 如果 active 仍然意外指向它，
                #      清空 active
                #
                #   3. 关闭 parent_connection
                #
                #
                # 注意：
                #
                #   self.recent 中保存的最终结果不会因此删除，
                #   所以 STATUS 还能查询历史结果。
                # ============================================================
                self.finishing.pop(handle.episode_id, None)

                if (
                    self.active is not None
                    and self.active.episode_id == handle.episode_id
                ):
                    self.active = None

                handle.connection.close()

    def serve_forever(self) -> None:
        # ====================================================================
        # serve_forever()
        #
        # RecorderServer 的主事件循环。
        #
        #
        # 负责同时处理：
        #
        #   A. episode worker 状态
        #
        #       _poll_workers()
        #
        #
        #   B. 控制客户端请求
        #
        #       ZMQ Poller
        #       recv_json()
        #       _handle_request()
        #       send_json()
        #
        #
        # 主循环结构：
        #
        #   while running:
        #
        #       更新 worker
        #
        #       ↓
        #
        #       最多等待 100 ms 控制请求
        #
        #       ↓
        #
        #       有请求则处理
        #
        #       ↓
        #
        #       回到下一轮
        # ====================================================================

        # ====================================================================
        # zmq.Poller
        #
        # 来源：
        #   PyZMQ。
        #
        # 作用：
        #   等待一个或多个 ZeroMQ socket 上发生事件，
        #   又不会像直接 recv() 一样无限阻塞。
        #
        #
        # register(self.socket, zmq.POLLIN)：
        #
        #   关注 self.socket 上的“可读事件”。
        #
        # 即：
        #
        #   有没有新的控制 request 可以 recv。
        #
        #
        # 为什么需要 Poller：
        #
        #   如果直接：
        #
        #       self.socket.recv_json()
        #
        #   在没有客户端请求时会阻塞，
        #   Server 就不能继续 _poll_workers()。
        #
        # Poller + 100 ms timeout 可以让 Server
        # 在控制请求和 worker 状态之间轮转。
        # ====================================================================
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        print(
            f"Recorder server listening on {self.config.control_bind_addr}",
            flush=True,
        )

        # ====================================================================
        # self.running 来源：
        #
        #   __init__():
        #       True
        #
        # 会被：
        #
        #   SHUTDOWN
        #   SIGINT
        #   SIGTERM
        #
        # 设置成 False。
        # ====================================================================
        while self.running:

            # ================================================================
            # 局部功能块：首先同步 worker 状态
            #
            # 输入：
            #
            #   self.active
            #   self.finishing
            #
            # 输出：
            #
            #   更新：
            #       WorkerHandle.status
            #       WorkerHandle.last_message
            #       self.active
            #       self.finishing
            #       self.recent
            #
            #
            # 这样紧接着进来的 STATUS request
            # 能读到尽量新的 worker 状态。
            # ================================================================
            self._poll_workers()

            # ================================================================
            # 局部功能块：最多等待 100ms 的控制请求
            #
            # poller.poll(timeout=100)
            #
            # timeout 单位：
            #   毫秒。
            #
            #
            # 返回的 events：
            #
            #   可以转换成：
            #
            #       {
            #           socket: event_flags
            #       }
            #
            #
            # 如果当前 self.socket 没有事件：
            #
            #   continue
            #
            # 回到下一轮，
            # 再次 _poll_workers()。
            # ================================================================
            events = dict(poller.poll(timeout=100))

            if self.socket not in events:
                continue

            try:
                # ============================================================
                # 局部功能块：接收客户端 JSON request
                #
                # request 来源：
                #
                #   外部控制客户端
                #       ↓ ZeroMQ
                #   self.socket
                #       ↓ recv_json()
                #   Python dict
                #
                #
                # 例如：
                #
                #   {
                #       "type": "START",
                #       "episode_id": ...
                #   }
                #
                #
                # request 去向：
                #
                #   _handle_request(request)
                #
                # 再根据 type 分发。
                # ============================================================
                request = self.socket.recv_json()

                # ============================================================
                # reply 来源：
                #
                #   _handle_request()
                #
                # 内部可能来自：
                #
                #   _start()
                #   _stop_save()
                #   _discard()
                #   _status()
                #   PING
                #   SHUTDOWN
                #
                # 最终都是一个 dict。
                # ============================================================
                reply = self._handle_request(request)

            # =================================================================
            # 局部功能块：统一把请求处理异常转成协议 error
            #
            # exc 可能来自：
            #
            #   recv_json()
            #   validate_episode_id()
            #   int(...)
            #   Pipe send
            #   或其他请求处理逻辑。
            #
            #
            # 这里不让异常直接破坏 REP 服务循环，
            # 而是构造：
            #
            #   error(str(exc))
            #
            # 仍然作为当前 request 的 reply 返回。
            # =================================================================
            except Exception as exc:
                reply = error(str(exc))

            # =================================================================
            # 局部功能块：将 reply 返回控制客户端
            #
            # 输入：
            #
            #   reply
            #       ← _handle_request()
            #       或 error(str(exc))
            #
            #
            # send_json：
            #
            #   把 Python dict 序列化成 JSON 后，
            #   通过 REP socket 返回。
            #
            #
            # 数据链完成：
            #
            #   client
            #      │
            #      │ request
            #      ▼
            #   RecorderServer
            #      │
            #      │ reply
            #      ▼
            #   client
            # =================================================================
            self.socket.send_json(reply)

    def close(self) -> None:
        # ====================================================================
        # close()
        #
        # 作用：
        #
        #   RecorderServer 退出前统一释放：
        #
        #       worker
        #       Pipe
        #       ZeroMQ socket
        #       ZeroMQ Context
        #
        #
        # 调用来源：
        #
        #   main():
        #
        #       try:
        #           server.serve_forever()
        #       finally:
        #           server.close()
        #
        #
        # 所以无论：
        #
        #   正常 SHUTDOWN
        #   Ctrl+C
        #   SIGTERM
        #   serve_forever 抛异常
        #
        # 都尽量执行 close()。
        # ====================================================================

        # ====================================================================
        # 局部功能块：如果仍有 active episode，先请求 DISCARD
        #
        # self.active：
        #   当前还在 recording 的 worker。
        #
        #
        # connection.send(DISCARD)：
        #
        #   尝试让 worker 自己正常结束，
        #   而不是第一时间 terminate。
        #
        #
        # 为什么 catch Exception：
        #
        #   close() 已经处于清理阶段，
        #   worker 或 Pipe 可能已经坏掉，
        #   不应因为一次 DISCARD 发送失败
        #   阻止后续其他资源继续清理。
        # ====================================================================
        if self.active is not None:
            try:
                self.active.connection.send({"type": DISCARD})
            except Exception:
                pass

        # ====================================================================
        # 局部功能块：给全部 worker 一个共享的 3 秒优雅退出窗口
        #
        # deadline：
        #
        #   当前 monotonic 时间 + 3 秒。
        #
        #
        # 后面对不同 worker：
        #
        #   remaining =
        #       deadline - 当前时间
        #
        #
        # 因此不是：
        #
        #   每个 worker 分别等 3 秒
        #
        # 而是：
        #
        #   所有 worker 总体共享约 3 秒等待时间。
        # ====================================================================
        deadline = time.monotonic() + 3.0

        # ====================================================================
        # 局部功能块：收集所有尚未完成清理的 worker
        #
        # handles：
        #
        #   finishing workers
        #       +
        #   active worker（如果还有）
        #
        # 后续统一 join / terminate / close connection。
        # ====================================================================
        handles = list(self.finishing.values())

        if self.active is not None:
            handles.append(self.active)

        for handle in handles:

            # ================================================================
            # 局部功能块：先等待 worker 自己正常退出
            #
            # remaining：
            #
            #   全局 deadline 剩余时间。
            #
            #
            # process.join(timeout=remaining)：
            #
            #   在 remaining 时间内等待该 Process 完成。
            # ================================================================
            remaining = max(0.0, deadline - time.monotonic())
            handle.process.join(timeout=remaining)

            # ================================================================
            # 局部功能块：优雅等待失败后强制 terminate
            #
            # 如果 process 仍存活：
            #
            #   terminate()
            #
            # 请求操作系统结束子进程。
            #
            # 随后：
            #
            #   join(timeout=1.0)
            #
            # 再等待最多一秒回收进程资源。
            # ================================================================
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=1.0)

            # ================================================================
            # worker 生命周期结束，
            # 关闭 RecorderServer 侧的 Pipe connection。
            # ================================================================
            handle.connection.close()

        # ====================================================================
        # 局部功能块：关闭控制面的 ZeroMQ 资源
        #
        # self.socket：
        #
        #   __init__() 创建的 REP socket。
        #
        # self.context：
        #
        #   __init__() 创建的 zmq.Context。
        #
        #
        # 顺序：
        #
        #   socket.close()
        #       ↓
        #   context.term()
        #
        #
        # 至此 RecorderServer 的网络资源释放完成。
        # ====================================================================
        self.socket.close()
        self.context.term()


def main() -> None:
    # ========================================================================
    # main()
    #
    # 当前 Python 模块的程序入口。
    #
    #
    # 项目正常启动路径：
    #
    #   用户
    #     │
    #     ▼
    #   ./scripts/start_recorder.sh
    #     │
    #     │ source ROS Humble 环境等
    #     │ 设置 PYTHONPATH
    #     ▼
    #
    #   python3 -m tiangong_recorder.recorder_server \
    #       --config config/recorder.yaml
    #
    #     │
    #     ▼
    #   main()
    #
    #
    # main() 主要负责：
    #
    #   1. 解析 config 文件位置
    #   2. 创建 RecorderConfig
    #   3. 创建 RecorderServer
    #   4. 注册系统信号
    #   5. 启动 serve_forever()
    #   6. 最终 close()
    # ========================================================================

    # ========================================================================
    # 局部功能块：创建命令行参数解析器
    #
    # argparse.ArgumentParser：
    #
    #   Python 标准库 argparse 提供。
    #
    #
    # --config：
    #
    #   可以由启动命令覆盖。
    #
    # 默认：
    #
    #   config/recorder.yaml
    #
    #
    # 这个 YAML 当前定义：
    #
    #   control_bind_addr
    #   advertise_host
    #   state_port_min
    #   state_port_max
    #   output_dir
    #   camera_topics
    #   image_width
    #   image_height
    #   timeout
    #   state_receive_hwm
    # ========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/recorder.yaml",
        help="Path to recorder YAML configuration",
    )

    # ========================================================================
    # args：
    #
    #   ← 当前进程命令行参数。
    #
    # 主要使用：
    #
    #   args.config
    #
    # 继续传给 RecorderConfig.from_yaml()。
    # ========================================================================
    args = parser.parse_args()

    # ========================================================================
    # 局部功能块：YAML → RecorderConfig
    #
    # RecorderConfig.from_yaml()
    #
    # 定义：
    #   config.py
    #
    #
    # 内部实际做：
    #
    #   Path(path).open(...)
    #       ↓
    #   yaml.safe_load(stream)
    #       ↓
    #   dict values
    #       ↓
    #   RecorderConfig(**values)
    #
    #
    # 数据链：
    #
    #   config/recorder.yaml
    #          │
    #          ▼
    #      args.config
    #          │
    #          ▼
    #   RecorderConfig.from_yaml()
    #          │
    #          ▼
    #        config
    # ========================================================================
    config = RecorderConfig.from_yaml(args.config)

    # ========================================================================
    # 局部功能块：创建 RecorderServer
    #
    # 输入：
    #
    #   config
    #       ← RecorderConfig.from_yaml()
    #
    #
    # RecorderServer.__init__：
    #
    #   1. validate config
    #   2. 创建 ZMQ REP control socket
    #   3. 初始化 active / finishing / recent
    #   4. 初始化 spawn multiprocessing context
    #
    #
    # 输出：
    #
    #   server
    #
    # 后续：
    #
    #   signal handler
    #   serve_forever()
    #   close()
    #
    # 都围绕这个对象工作。
    # ========================================================================
    server = RecorderServer(config)

    # ========================================================================
    # stop_server()
    #
    # 这是当前 main() 内定义的 signal handler。
    #
    #
    # _signum：
    #   signal number。
    #
    # _frame：
    #   Python 当前执行栈 frame。
    #
    # 当前代码不需要这两个值，
    # 所以前面加 "_" 表示参数存在但不使用。
    #
    #
    # 收到信号后只做：
    #
    #   server.running = False
    #
    #
    # 数据去向：
    #
    #   RecorderServer.serve_forever()
    #
    #       while self.running:
    #
    # 下一轮判断失败，
    # 主循环结束。
    # ========================================================================
    def stop_server(_signum, _frame):
        server.running = False

    # ========================================================================
    # 局部功能块：注册 SIGINT / SIGTERM
    #
    # SIGINT：
    #   通常 Ctrl+C。
    #
    # SIGTERM：
    #   外部正常终止请求。
    #
    #
    # 两者：
    #
    #   → stop_server()
    #   → server.running = False
    #   → serve_forever() 返回
    #   → finally
    #   → server.close()
    # ========================================================================
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    try:
        # ====================================================================
        # 局部功能块：正式进入 RecorderServer 主循环
        #
        # server：
        #   ← RecorderServer(config)
        #
        #
        # serve_forever() 内持续处理：
        #
        #       Worker Pipe
        #           │
        #           ▼
        #       _poll_workers()
        #
        #
        #       Control client
        #           │
        #           ▼
        #       ZMQ REP
        #           │
        #           ▼
        #       _handle_request()
        #
        #
        # 直到：
        #
        #   SHUTDOWN
        #   SIGINT
        #   SIGTERM
        #   或异常
        #
        # 导致该函数返回/退出。
        # ====================================================================
        server.serve_forever()

    finally:
        # ====================================================================
        # 局部功能块：统一释放 RecorderServer 资源
        #
        # finally 的含义：
        #
        #   无论 try 中：
        #
        #       正常结束
        #
        #   还是：
        #
        #       抛出异常
        #
        #   都会进入这里。
        #
        #
        # server.close() 最终负责：
        #
        #   active worker
        #       ↓ DISCARD
        #
        #   workers
        #       ↓ join / terminate
        #
        #   Pipe
        #       ↓ close
        #
        #   ZMQ REP socket
        #       ↓ close
        #
        #   ZMQ Context
        #       ↓ term
        #
        # 最终完成 Recorder 服务退出。
        # ====================================================================
        server.close()


# ============================================================================
# Python 模块直接运行入口
#
# 当执行：
#
#   python3 -m tiangong_recorder.recorder_server
#
# 时：
#
#   __name__ == "__main__"
#
# 因此调用：
#
#   main()
#
#
# 最外层程序链：
#
#   start_recorder.sh
#          │
#          ▼
#   recorder_server module
#          │
#          ▼
#        main()
#          │
#          ▼
#   RecorderConfig
#          │
#          ▼
#   RecorderServer
#          │
#          ▼
#   serve_forever()
#          │
#          ├── control requests
#          │
#          └── episode workers
#          │
#          ▼
#        close()
# ============================================================================
if __name__ == "__main__":
    main()
