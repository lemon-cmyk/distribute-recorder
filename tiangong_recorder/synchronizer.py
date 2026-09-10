from __future__ import annotations


# ============================================================================
# 当前文件：tiangong_recorder/synchronizer.py
#
# 【这个文件在整个项目中的职责】
#
# 当前文件负责：
#
#       “把不同时间到达的 robot state 和三路 camera frame
#        根据采集时间戳组合成一条完整的训练数据。”
#
#
# 整个项目中的主要数据链可以理解为：
#
#       x86 teleoperation process
#               │
#               │ robot state/action
#               │ 约 50 Hz
#               │ ZMQ
#               ▼
#       episode_worker.py
#               │
#               │ synchronizer.add_state(state)
#               │
#               │
#       ROS 2 camera topics
#               │
#               │ sensor_msgs/Image
#               ▼
#       EpisodeCameraNode._camera_callback()
#               │
#               ▼
#       image_decoder.py
#               │
#               │ decode_ros_image()
#               ▼
#           CameraFrame
#               │
#               │ synchronizer.add_camera_frame()
#               ▼
#       LiveEpisodeSynchronizer          ← 当前文件
#               │
#               │ 按 _capture_time_ns / CameraFrame.timestamp_ns
#               │ 做时间对齐
#               ▼
#           episode_data
#               │
#               ▼
#       episode_worker.py
#               │
#               ▼
#       dataset_writer.py
#               │
#               │ write_episode_pickle()
#               ▼
#          <episode_id>.pkl
#
#
# 【这里所谓的“对齐”是什么】
#
# robot state 有自己的采集时间：
#
#       state["_capture_time_ns"]
#
# camera frame 有自己的采集时间：
#
#       CameraFrame.timestamp_ns
#
# 对于一条 state：
#
#       T_state
#
# 正常情况下，每一路相机都选择：
#
#       timestamp <= T_state
#
# 中时间最接近 T_state 的那一帧，也就是：
#
#       latest causal frame
#       “state 发生之前最近的一张图”
#
#
# 例如某一路 camera：
#
#       33ms     66ms     99ms
#        │        │        │
#        ▼        ▼        ▼
#       img0     img1     img2
#
#                 state = 80ms
#
# 最终选择：
#
#       img1 = 66ms
#
# 而不是 99ms，
# 因为 99ms 已经发生在 state 之后。
#
#
# 【为什么还要等待 99ms 这样的未来 frame 到达】
#
# 如果当前队列只有：
#
#       33ms
#       66ms
#
# 而 state = 80ms，
#
# 现在还不能确定 66ms 是最终应该使用的帧，
# 因为后面可能马上收到：
#
#       75ms
#
# 所以 merge_ready(force=False) 会等待每一路 camera 的“最新时间戳”
# 都已经越过 state 时间：
#
#       newest_camera_timestamp >= state_timestamp
#
# 这相当于用最新 camera timestamp 作为 watermark。
#
# 当 watermark 已越过 state，
# 才可以确定 state 之前不会再正常到达一张时间更近的有序 camera frame。
#
#
# 【最终输出的数据结构】
#
# 每个 state 最后变成：
#
#       entry = {
#           ... robot observation/action fields ...,
#
#           "image": {
#               "head": {
#                   "color": np.ndarray,
#               },
#               "left_wrist": {
#                   "color": np.ndarray,
#               },
#               "right_wrist": {
#                   "color": np.ndarray,
#               },
#           },
#       }
#
# 多个 entry 按时间顺序组成：
#
#       self.episode_data: list[dict]
#
# 后续由 episode_worker.py 交给：
#
#       dataset_writer.write_episode_pickle()
#
# 写成 PKL。
# ============================================================================


# ============================================================================
# deque
#
# 来源：
#   Python 标准库 collections。
#
# deque = double-ended queue，双端队列。
#
# 当前文件主要用它保存两类按时间排列的数据：
#
#   ① 每一路 camera 的 CameraFrame
#
#       self.frames["head"]
#       self.frames["left_wrist"]
#       self.frames["right_wrist"]
#
#   ② 还没有与 camera 完成匹配的 robot state
#
#       self.pending_states
#
#
# 为什么适合这里：
#
# 对齐完成以后需要不断删除最旧的数据：
#
#       queue.popleft()
#
# deque 从左侧删除元素是 O(1)，
# 比普通 list 的 pop(0) 更适合这种流式队列。
# ============================================================================
from collections import deque


# ============================================================================
# dataclass
#
# 来源：
#   Python 标准库 dataclasses。
#
# 当前文件用它定义 CameraFrame。
#
# dataclass 可以把：
#
#       timestamp_ns
#       image
#       encoding
#
# 这几个本来相互独立的变量封装成一个明确的“相机帧对象”。
# ============================================================================
from dataclasses import dataclass


# ============================================================================
# typing
#
# 来源：
#   Python 标准库。
#
# Deque：
#   用来表示 deque 的元素类型。
#
# Dict：
#   表示字典的 key/value 类型。
#
# Iterable：
#   表示“可以被遍历的一组 camera name”。
#
# 例如 LiveEpisodeSynchronizer 可以接受：
#
#       ["head", "left_wrist", "right_wrist"]
#
# 也可以接受 tuple、dict_keys 等其他 iterable。
# ============================================================================
from typing import Deque, Dict, Iterable


# ============================================================================
# NumPy
#
# 来源：
#   第三方数值计算库 numpy。
#
# 当前文件最重要的用途是：
#
#       CameraFrame.image: np.ndarray
#
# 真正的 ndarray 创建发生在 image_decoder.py：
#
#       ROS sensor_msgs/Image
#               │
#               ▼
#       np.frombuffer(message.data)
#               │
#               ▼
#       H × W × 3 ndarray
#               │
#               ▼
#       CameraFrame.image
#
# 当前 synchronizer.py 不负责解码图片，
# 只负责保存、选择和复制这些 ndarray。
# ============================================================================
import numpy as np


# ============================================================================
# CameraFrame
#
# 定义来源：
#   当前 tiangong_recorder/synchronizer.py。
#
#
# 【它是什么】
#
# CameraFrame 是项目内部用于表示“一张已经解码完成的相机图像”的数据对象。
#
# 它不是 ROS 原始：
#
#       sensor_msgs.msg.Image
#
# ROS Image 会先在：
#
#       tiangong_recorder/image_decoder.py
#
# 经过：
#
#       decode_ros_image()
#
# 才被转换成 CameraFrame。
#
#
# 数据流：
#
#       ROS Image
#           │
#           ▼
#       image_decoder.decode_ros_image()
#           │
#           ▼
#       CameraFrame
#           │
#           ▼
#       EpisodeCameraNode._camera_callback()
#           │
#           ▼
#       LiveEpisodeSynchronizer.add_camera_frame()
#
#
# frozen=True：
#
#   dataclass 创建以后，不允许直接重新赋值：
#
#       frame.timestamp_ns = ...
#
#   这样可以避免已经进入同步队列的 frame
#   在对齐过程中被意外修改时间戳等元数据。
# ============================================================================
@dataclass(frozen=True)
class CameraFrame:

    # ========================================================================
    # timestamp_ns
    #
    # 来源：
    #
    #   image_decoder.image_timestamp_ns()
    #
    # 优先读取 ROS Image：
    #
    #       message.header.stamp.sec
    #       message.header.stamp.nanosec
    #
    # 并转换成：
    #
    #       timestamp_ns
    #
    # 如果 ROS header 没有有效时间戳，
    # image_decoder.py 才会使用 fallback timestamp。
    #
    #
    # 后续用途：
    #
    #       add_camera_frame()
    #             │
    #             ├── 检查相机帧时间是否乱序
    #             │
    #             ▼
    #       _select_frame()
    #             │
    #             └── 与 state["_capture_time_ns"] 比较
    #
    # 这是 camera 和 robot state 时间同步的核心字段。
    # ========================================================================
    timestamp_ns: int

    # ========================================================================
    # image
    #
    # 来源：
    #
    #   image_decoder.decode_ros_image()
    #
    # 由 ROS Image.message.data 解码得到。
    #
    # 当前 recorder.yaml 配置期望尺寸：
    #
    #       640 × 480
    #
    # 因此通常结构是：
    #
    #       np.ndarray
    #       shape = (480, 640, 3)
    #       dtype = uint8
    #
    #
    # 后续用途：
    #
    #   merge_ready()
    #       ↓
    #   selected_frames[camera_name].image.copy()
    #       ↓
    #   entry["image"][camera_name]["color"]
    #       ↓
    #   self.episode_data
    #       ↓
    #   PKL
    # ========================================================================
    image: np.ndarray

    # ========================================================================
    # encoding
    #
    # 来源：
    #
    #   ROS Image.message.encoding
    #
    # image_decoder.py 当前支持：
    #
    #       rgb8
    #       bgr8
    #
    #
    # 注意：
    #
    # 当前 synchronizer 在生成最终 entry 时只写：
    #
    #       selected_frame.image
    #
    # 并没有把 encoding 一起写进 episode_data。
    #
    # 因此 encoding 当前主要作为 CameraFrame 的图像格式元数据存在，
    # 不会直接进入最终 PKL entry。
    # ========================================================================
    encoding: str


# ============================================================================
# LiveEpisodeSynchronizer
#
# 定义来源：
#   当前 synchronizer.py。
#
#
# 【它是什么】
#
# 这是一次 episode 内部的“实时时间同步器”。
#
# 一个 episode worker 启动时，
# episode_worker.py 会创建：
#
#       synchronizer = LiveEpisodeSynchronizer(
#           config.camera_topics.keys()
#       )
#
# config.camera_topics 来自：
#
#       config/recorder.yaml
#
# 当前是：
#
#       head
#       left_wrist
#       right_wrist
#
#
# 【输入】
#
# 两条独立的数据流：
#
#   Camera：
#
#       ROS 2
#         ↓
#       EpisodeCameraNode
#         ↓
#       decode_ros_image()
#         ↓
#       CameraFrame
#         ↓
#       add_camera_frame()
#
#
#   Robot：
#
#       x86 teleoperation process
#         ↓
#       ZMQ
#         ↓
#       episode_worker.state_socket
#         ↓
#       recv_pyobj()
#         ↓
#       state: dict
#         ↓
#       add_state()
#
#
# 【输出】
#
#       self.episode_data
#
# 后续：
#
#       episode_worker.py
#           ↓
#       write_episode_pickle()
#           ↓
#       /home/nvidia/teleop_logs/<episode_id>.pkl
# ============================================================================
class LiveEpisodeSynchronizer:
    """Causally aligns ordered camera frames with ordered robot states."""

    # ========================================================================
    # INTERNAL_STATE_KEYS
    #
    # 这些字段来自 x86 发送过来的 robot state。
    #
    # 它们用于 recorder 内部：
    #
    #       _episode_id
    #           用于确认 state 属于哪个 episode。
    #
    #       _frame_index
    #           用于 episode_worker 检查 state 是否连续：
    #
    #               0, 1, 2, 3, ...
    #
    #       _capture_time_ns
    #           robot state 的采集时间，
    #           synchronizer 用它和 camera timestamp 做对齐。
    #
    #
    # 这三个字段属于：
    #
    #       Recorder 的传输/同步元数据
    #
    # 而不是最终训练 sample 本身希望保留的数据。
    #
    # 所以 merge_ready() 构建最终 entry 时会过滤：
    #
    #       if key not in self.INTERNAL_STATE_KEYS
    #
    # 最终不会写进 episode_data。
    #
    #
    # 注意：
    #
    # 这个 distribute-recorder 仓库负责“接收”这些字段；
    # x86 teleoperation process 的 state 构造逻辑不在当前
    # synchronizer.py 中。
    # ========================================================================
    INTERNAL_STATE_KEYS = {
        "_episode_id",
        "_frame_index",
        "_capture_time_ns",
    }

    def __init__(self, camera_names: Iterable[str]):

        # ====================================================================
        # 局部功能块：固定本次同步器需要处理的 camera 列表
        #
        # 输入来源：
        #
        #   camera_names
        #       ← episode_worker.py
        #       ← config.camera_topics.keys()
        #       ← config/recorder.yaml
        #
        # 当前默认得到：
        #
        #       "head"
        #       "left_wrist"
        #       "right_wrist"
        #
        #
        # tuple(camera_names)：
        #
        #   把传入的 iterable 固定成 tuple，
        #   之后整个 episode 都使用同一组 camera。
        #
        #
        # 输出：
        #
        #       self.camera_names
        #
        # 后续传给/用于：
        #
        #       self.frames 初始化
        #       cameras_ready()
        #       can_match()
        #       merge_ready()
        #       alignment_summary()
        # ====================================================================
        self.camera_names = tuple(camera_names)

        # ====================================================================
        # 局部功能块：为每一路 camera 创建独立的时间有序帧队列
        #
        # 输入来源：
        #
        #       self.camera_names
        #
        # 当前形成：
        #
        #   self.frames = {
        #       "head": deque(),
        #       "left_wrist": deque(),
        #       "right_wrist": deque(),
        #   }
        #
        #
        # 每个 deque 后续接收：
        #
        #       CameraFrame
        #
        # 来源：
        #
        #       EpisodeCameraNode._camera_callback()
        #           ↓
        #       decode_ros_image()
        #           ↓
        #       add_camera_frame(camera_name, frame)
        #
        #
        # 后续主要被：
        #
        #       cameras_ready()
        #       can_match()
        #       _select_frame()
        #       _prune_before_selected()
        #       merge_ready()
        #
        # 使用。
        # ====================================================================
        self.frames: Dict[str, Deque[CameraFrame]] = {
            name: deque() for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：创建“等待与相机匹配的 state”队列
        #
        # 数据来源：
        #
        #       x86 teleoperation process
        #           ↓
        #       ZMQ
        #           ↓
        #       episode_worker.state_socket.recv_pyobj()
        #           ↓
        #       synchronizer.add_state(state)
        #
        # add_state() 会把 state append 到这里。
        #
        #
        # 后续：
        #
        #       merge_ready()
        #
        # 总是从：
        #
        #       self.pending_states[0]
        #
        # 取最早的一条 state 进行对齐。
        #
        # 对齐完成以后：
        #
        #       self.pending_states.popleft()
        #
        # 删除该 state。
        # ====================================================================
        self.pending_states: Deque[dict] = deque()

        # ====================================================================
        # 局部功能块：创建最终 episode 数据缓存
        #
        # 初始：
        #
        #       []
        #
        # 数据由：
        #
        #       merge_ready()
        #
        # 一条一条 append。
        #
        #
        # 每个元素大致：
        #
        #       {
        #           ... robot state/action ...,
        #           "image": {
        #               "head": {"color": ...},
        #               "left_wrist": {"color": ...},
        #               "right_wrist": {"color": ...},
        #           }
        #       }
        #
        #
        # 后续去向：
        #
        #       episode_worker.py
        #           ↓
        #       write_episode_pickle(
        #           episode_id,
        #           synchronizer.episode_data,
        #           config.output_dir,
        #       )
        #
        # 最终写成 PKL。
        # ====================================================================
        self.episode_data: list[dict] = []

        # ====================================================================
        # 局部功能块：统计被丢弃的乱序 camera frame
        #
        # 每一路 camera 单独统计。
        #
        # 初始：
        #
        #       0
        #
        # add_camera_frame() 如果发现：
        #
        #       新 frame.timestamp
        #           <
        #       当前队尾 frame.timestamp
        #
        # 就认为新帧是 out-of-order，
        # 不把它加入同步队列，并将计数 +1。
        #
        #
        # 后续去向：
        #
        #       alignment_summary()
        #
        # 最终随 worker 的 "saved" 消息返回，
        # 用于诊断 camera 时间序列是否异常。
        # ====================================================================
        self.dropped_out_of_order_images: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：统计 fallback frame 匹配次数
        #
        # 正常匹配希望找到：
        #
        #       frame.timestamp_ns <= state_timestamp_ns
        #
        # 中最近的一帧。
        #
        # 但如果当前 camera 队列中所有帧都比 state 更新：
        #
        #       frame.timestamp_ns > state_timestamp_ns
        #
        # _select_frame() 就只能退化选择：
        #
        #       queue[0]
        #
        # 也就是当前保存的最早一张图。
        #
        # 每发生一次这种情况：
        #
        #       fallback_matches[camera] += 1
        #
        #
        # 后续进入 alignment_summary()，
        # 用于诊断同步开始阶段或时钟异常。
        # ====================================================================
        self.fallback_matches: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：初始化时间对齐误差统计
        #
        # alignment_count：
        #   已经完成多少次 state-camera 匹配。
        #
        # alignment_abs_sum_ns：
        #   所有 |state_timestamp - camera_timestamp| 的累计值。
        #
        # alignment_abs_max_ns：
        #   到目前为止最大的绝对时间差。
        #
        #
        # 这些变量在：
        #
        #       merge_ready()
        #
        # 每成功生成一条 entry 时更新。
        #
        # 后续：
        #
        #       alignment_summary()
        #
        # 转换成：
        #
        #       mean_abs_delta_ms
        #       max_abs_delta_ms
        #
        # 用来判断相机和 robot state 的同步质量。
        # ====================================================================
        self.alignment_count: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.alignment_abs_sum_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.alignment_abs_max_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

    def add_camera_frame(self, camera_name: str, frame: CameraFrame) -> bool:
        # ====================================================================
        # add_camera_frame()
        #
        # 调用来源：
        #
        #       episode_worker.py
        #           ↓
        #       EpisodeCameraNode._camera_callback()
        #           ↓
        #       decode_ros_image()
        #           ↓
        #       frame: CameraFrame
        #           ↓
        #       synchronizer.add_camera_frame(camera_name, frame)
        #
        #
        # 输入：
        #
        #   camera_name
        #       ← config.camera_topics 中的 key
        #
        #       当前通常是：
        #           head
        #           left_wrist
        #           right_wrist
        #
        #   frame
        #       ← image_decoder.decode_ros_image()
        #
        #
        # 目标：
        #
        #       保证每一路 self.frames[camera_name]
        #       中的 CameraFrame 按 timestamp 非递减排列。
        # ====================================================================

        # ====================================================================
        # 局部功能块：取得对应 camera 的帧队列
        #
        # camera_name：
        #   ← _camera_callback()
        #
        # self.frames：
        #   ← __init__()
        #
        # 输出 queue：
        #
        #   例如 camera_name == "head"：
        #
        #       queue = self.frames["head"]
        #
        # 后续用于检查时间顺序并 append 新帧。
        # ====================================================================
        queue = self.frames[camera_name]

        # ====================================================================
        # 局部功能块：拒绝时间戳倒退的 camera frame
        #
        # queue[-1]：
        #   当前已经接收到的最新 camera frame。
        #
        # frame：
        #   新到达的 CameraFrame。
        #
        #
        # 如果：
        #
        #       frame.timestamp_ns < queue[-1].timestamp_ns
        #
        # 表示新收到的图像在采集时间上反而更早。
        #
        # 如果允许它进入队列：
        #
        #       _select_frame()
        #
        # 就不能再依赖“队列按时间排序”这一前提。
        #
        # 所以：
        #
        #   ① 不加入 queue
        #   ② dropped_out_of_order_images += 1
        #   ③ return False
        #
        #
        # dropped 统计之后会进入：
        #
        #       alignment_summary()
        # ====================================================================
        if queue and frame.timestamp_ns < queue[-1].timestamp_ns:
            self.dropped_out_of_order_images[camera_name] += 1
            return False

        # ====================================================================
        # 局部功能块：接受合法 camera frame
        #
        # 经过上面的检查后：
        #
        #       queue
        #
        # 继续保持时间有序。
        #
        # 新 frame 后续由：
        #
        #       can_match()
        #       _select_frame()
        #       merge_ready()
        #
        # 使用。
        # ====================================================================
        queue.append(frame)
        return True

    def add_state(self, state: dict) -> None:
        # ====================================================================
        # add_state()
        #
        # 输入 state 的直接来源：
        #
        #       episode_worker.py
        #           ↓
        #       state_socket.recv_pyobj()
        #
        # 更上游：
        #
        #       x86 teleoperation process
        #           ↓
        #       ZMQ PUSH/PULL 数据通道
        #           ↓
        #       Orin episode worker
        #
        #
        # episode_worker 在调用 add_state() 以前已经检查：
        #
        #   state["_episode_id"]
        #       是否与当前 episode_id 相同
        #
        #   state["_frame_index"]
        #       是否与 received_state_count 连续
        #
        #
        # 当前函数进一步负责：
        #
        #       检查 state 的采集 timestamp 是否保持单调，
        #       然后加入 pending_states。
        # ====================================================================

        # ====================================================================
        # 局部功能块：提取 robot state 的采集时间
        #
        # 来源：
        #
        #       state["_capture_time_ns"]
        #
        # 这是由 x86 state 数据携带过来的同步元数据。
        #
        # int()：
        #   将其统一转换成 Python int。
        #
        #
        # 输出：
        #
        #       timestamp_ns
        #
        # 后续：
        #
        #       与上一条 pending state 的时间比较。
        #
        # state 本身则在随后 append 到 pending_states，
        # 最终 merge_ready() 再次读取这个字段，
        # 与 CameraFrame.timestamp_ns 做匹配。
        # ====================================================================
        timestamp_ns = int(state["_capture_time_ns"])

        # ====================================================================
        # 局部功能块：验证 pending state 的 timestamp 单调性
        #
        # self.pending_states[-1]：
        #   当前尚未完成 camera 匹配的最后一条 state。
        #
        # previous：
        #   它的 _capture_time_ns。
        #
        #
        # 如果：
        #
        #       当前 timestamp < previous
        #
        # 说明 robot state 时间发生倒退。
        #
        # 因为 merge_ready() 默认认为 pending_states
        # 是按照采集时间排列的，
        # 此时直接抛 ValueError，不允许继续同步。
        # ====================================================================
        if self.pending_states:
            previous = int(self.pending_states[-1]["_capture_time_ns"])
            if timestamp_ns < previous:
                raise ValueError("state timestamps must be monotonic")

        # ====================================================================
        # 局部功能块：把 state 放入待匹配队列
        #
        # 输入：
        #
        #       state
        #
        # 输出：
        #
        #       self.pending_states
        #
        # 后续：
        #
        #       merge_ready()
        #
        # 会始终从 pending_states[0]
        # 处理最早尚未完成匹配的 state。
        # ====================================================================
        self.pending_states.append(state)

    def cameras_ready(self) -> bool:
        # ====================================================================
        # 局部功能块：检查所有 camera 是否至少有一张图
        #
        # 数据来源：
        #
        #       self.frames
        #           ← add_camera_frame()
        #
        # all(...) 要求：
        #
        #       head queue 非空
        #       AND
        #       left_wrist queue 非空
        #       AND
        #       right_wrist queue 非空
        #
        #
        # 输出：
        #
        #       True / False
        #
        #
        # 主要去向：
        #
        #   ① episode_worker.py
        #
        #       synchronizer.cameras_ready()
        #
        #       所有 camera 首帧到齐以后，
        #       worker 才向 RecorderServer 发送 "ready"。
        #
        #   ② can_match()
        #
        #       判断某条 state 是否已经具备匹配条件。
        #
        #   ③ merge_ready()
        #
        #       如果 camera 还没全部准备好就停止 merge。
        # ====================================================================
        return all(self.frames[name] for name in self.camera_names)

    def can_match(self, state_timestamp_ns: int) -> bool:
        # ====================================================================
        # can_match()
        #
        # 作用：
        #
        #   判断“现在是否已经可以安全地处理这个 state”。
        #
        #
        # state_timestamp_ns 来源：
        #
        #       merge_ready()
        #           ↓
        #       state["_capture_time_ns"]
        #
        #
        # 第一层条件：
        #
        #       self.cameras_ready()
        #
        # 所有 camera 至少必须有一帧。
        #
        #
        # 第二层条件：
        #
        # 对每一路 camera：
        #
        #       self.frames[name][-1].timestamp_ns
        #           >=
        #       state_timestamp_ns
        #
        # 即该 camera 当前最新帧的时间已经“越过”这条 state。
        #
        #
        # 可以把：
        #
        #       frames[name][-1].timestamp_ns
        #
        # 理解成该 camera 当前的时间 watermark。
        #
        #
        # 为什么需要这个 watermark：
        #
        # 假设 state = 100ms，
        # 当前 camera 只有：
        #
        #       70ms
        #
        # 不能马上选择 70ms，
        # 因为后面可能还有：
        #
        #       90ms
        #
        # 当已经收到：
        #
        #       110ms
        #
        # 且 camera frame 保持有序时，
        # 才能认为 state=100ms 之前正常到达的帧已经收齐，
        # 此时再从中寻找“<=100ms 的最近一帧”。
        #
        #
        # 输出：
        #
        #       True
        #           → merge_ready() 可以调用 _select_frame()
        #
        #       False
        #           → merge_ready() 暂停，等待更多 camera frame。
        # ====================================================================
        return self.cameras_ready() and all(
            self.frames[name][-1].timestamp_ns >= state_timestamp_ns
            for name in self.camera_names
        )

    @staticmethod
    def _select_frame(
        queue: Deque[CameraFrame],
        state_timestamp_ns: int,
    ) -> tuple[CameraFrame, bool]:
        # ====================================================================
        # _select_frame()
        #
        # 这是实际执行：
        #
        #       “给一条 state 选哪张 camera image”
        #
        # 的核心函数。
        #
        #
        # 输入 queue：
        #
        #       self.frames[camera_name]
        #
        # 来源：
        #       add_camera_frame()
        #
        #
        # 输入 state_timestamp_ns：
        #
        #       state["_capture_time_ns"]
        #
        # 来源：
        #       add_state() → pending_states → merge_ready()
        #
        #
        # 输出：
        #
        #       tuple[CameraFrame, bool]
        #
        # 第一个值：
        #       被选中的 CameraFrame。
        #
        # 第二个值：
        #       是否使用了 fallback。
        # ====================================================================

        # ====================================================================
        # 局部功能块：从最新帧向过去搜索 causal frame
        #
        # queue 本身按照时间从旧到新：
        #
        #       [t0, t1, t2, t3]
        #
        # reversed(queue)：
        #
        #       t3 → t2 → t1 → t0
        #
        # 因此遇到第一个：
        #
        #       frame.timestamp_ns <= state_timestamp_ns
        #
        # 就一定是：
        #
        #       state 之前距离它最近的一张 camera frame。
        #
        #
        # 例如：
        #
        #       camera: 33, 66, 99
        #       state : 80
        #
        # 搜索：
        #
        #       99 > 80      ×
        #       66 <= 80     ✓
        #
        # 返回 66。
        #
        #
        # False：
        #   表示这是正常 causal match，
        #   没有使用 fallback。
        # ====================================================================
        for frame in reversed(queue):
            if frame.timestamp_ns <= state_timestamp_ns:
                return frame, False

        # ====================================================================
        # 局部功能块：没有任何历史帧时使用最早可用帧作为 fallback
        #
        # 能执行到这里表示：
        #
        #       queue 中所有 frame.timestamp_ns
        #           >
        #       state_timestamp_ns
        #
        # 例如：
        #
        #       state = 50
        #
        #       camera queue:
        #           66
        #           99
        #
        # 没有 state 发生之前的 camera frame。
        #
        # 此时选择：
        #
        #       queue[0]
        #
        # 即当前能够获得的最早一帧。
        #
        # True：
        #
        #   告诉 merge_ready()：
        #
        #       这次不是正常的 historical/causal match。
        #
        # merge_ready() 会：
        #
        #       fallback_matches[camera_name] += 1
        #
        # 便于之后诊断这种异常/边界情况出现了多少次。
        # ====================================================================
        return queue[0], True

    @staticmethod
    def _prune_before_selected(
        queue: Deque[CameraFrame],
        selected: CameraFrame,
    ) -> None:
        # ====================================================================
        # _prune_before_selected()
        #
        # 作用：
        #
        #   一条 state 已经完成匹配后，
        #   删除“比本次 selected frame 更旧”的 camera frame。
        #
        #
        # 输入 queue：
        #
        #       self.frames[camera_name]
        #
        # 输入 selected：
        #
        #       本轮 _select_frame() 选择的 CameraFrame。
        #
        #
        # 例如：
        #
        # 原来：
        #
        #       [33, 66, 99, 132]
        #
        # 本次 selected：
        #
        #       66
        #
        # 处理后：
        #
        #       [66, 99, 132]
        #
        #
        # 注意：
        #
        #       selected 自己不会删除。
        #
        # 这是刻意的。
        #
        # 因为下一条 robot state 可能仍然需要复用这张图片。
        #
        # 例如：
        #
        #       camera = 66ms
        #
        #       state1 = 70ms
        #       state2 = 75ms
        #
        # 两条 state 都可能匹配 66ms。
        #
        #
        # 输出：
        #
        #       原地修改 queue
        #
        # 没有返回值。
        #
        # 修改后的 queue 会继续被后面的：
        #
        #       can_match()
        #       _select_frame()
        #
        # 使用。
        # ====================================================================
        while len(queue) > 1 and queue[0] is not selected:
            queue.popleft()

    def merge_ready(self, force: bool = False) -> int:
        # ====================================================================
        # merge_ready()
        #
        # 这是整个 synchronizer 的核心调度函数。
        #
        #
        # 正常调用来源：
        #
        #       episode_worker.py 主循环
        #
        #       synchronizer.merge_ready(force=False)
        #
        #
        # episode 结束时，如果最后几条 state 等不到新的 camera watermark：
        #
        #       synchronizer.merge_ready(force=True)
        #
        #
        # 输入：
        #
        #   self.pending_states
        #       ← add_state()
        #
        #   self.frames
        #       ← add_camera_frame()
        #
        #
        # 输出：
        #
        #   self.episode_data
        #       新增完成对齐的数据 entry
        #
        #   return merged
        #       本次函数一共完成了多少条 state 的合并。
        # ====================================================================

        # ====================================================================
        # 局部功能块：初始化本次调用的 merge 数量
        #
        # merged 只统计：
        #
        #       “这一次 merge_ready() 调用”
        #
        # 合并了多少条。
        #
        # 它不会跨调用累计。
        #
        # 最后 return 给 episode_worker。
        # ====================================================================
        merged = 0

        # ====================================================================
        # 局部功能块：按时间顺序持续处理最老的 pending state
        #
        # 只要 pending_states 非空就尝试继续。
        #
        # 为什么总是处理队头：
        #
        #       pending_states
        #
        # 在 add_state() 中按时间加入，
        # 所以：
        #
        #       pending_states[0]
        #
        # 是当前最早尚未完成 camera 对齐的 robot state。
        #
        # 必须先处理它，
        # 才能安全裁剪旧 camera frame。
        # ====================================================================
        while self.pending_states:

            # ================================================================
            # 局部功能块：取得当前最早的 state 及其时间
            #
            # state：
            #   ← add_state()
            #   ← x86 robot state
            #
            # state_timestamp_ns：
            #   ← state["_capture_time_ns"]
            #
            #
            # 后续 state_timestamp_ns 会传给：
            #
            #       can_match()
            #       _select_frame()
            #
            # 作为 camera-state 对齐的时间基准。
            # ================================================================
            state = self.pending_states[0]
            state_timestamp_ns = int(state["_capture_time_ns"])

            # ================================================================
            # 局部功能块：没有三路 camera 首帧时暂停合并
            #
            # cameras_ready() 检查：
            #
            #       每一路 camera queue 都非空。
            #
            # 如果有任何一路没有图片：
            #
            #       break
            #
            # state 仍然留在 pending_states，
            # 等 episode_worker 收到更多 camera frame 后
            # 下次再调用 merge_ready()。
            # ================================================================
            if not self.cameras_ready():
                break

            # ================================================================
            # 局部功能块：正常模式下等待 camera watermark 越过 state
            #
            # force=False：
            #
            #   正常实时录制模式。
            #
            #   必须满足 can_match(state_timestamp_ns)。
            #
            #   否则说明至少一路 camera 的最新时间
            #   还没有越过当前 state，
            #   暂时不能确定“state 前最近一帧”是哪一张。
            #
            #
            # force=True：
            #
            #   episode 收尾时使用。
            #
            #   不再等待每一路最新 camera timestamp 越过 state，
            #   直接使用现有 queue 尽可能完成剩余匹配。
            #
            #   但上面的 cameras_ready() 仍然必须成立，
            #   所以 force 并不是“没有图也强行生成”。
            # ================================================================
            if not force and not self.can_match(state_timestamp_ns):
                break

            # ================================================================
            # 局部功能块：为当前 state 创建“各 camera 选中帧”容器
            #
            # 最终结构：
            #
            #       selected_frames = {
            #           "head": CameraFrame(...),
            #           "left_wrist": CameraFrame(...),
            #           "right_wrist": CameraFrame(...),
            #       }
            #
            #
            # 后续：
            #
            #   ① 用于创建 entry["image"]
            #
            #   ② 用于 _prune_before_selected()
            #
            # ================================================================
            selected_frames: Dict[str, CameraFrame] = {}

            # ================================================================
            # 局部功能块：分别给每一路 camera 选择与当前 state 对齐的帧
            #
            # camera_name 来源：
            #
            #       self.camera_names
            #
            # 对每一路：
            #
            #       self.frames[camera_name]
            #               ↓
            #       _select_frame(...)
            #
            # 得到：
            #
            #       selected
            #       used_fallback
            # ================================================================
            for camera_name in self.camera_names:
                selected, used_fallback = self._select_frame(
                    self.frames[camera_name],
                    state_timestamp_ns,
                )

                # ============================================================
                # 局部功能块：保存本路 camera 的匹配结果
                #
                # selected：
                #   ← _select_frame()
                #
                # 输出到：
                #
                #       selected_frames[camera_name]
                #
                # 后续用于：
                #
                #       selected_frames[camera_name].image.copy()
                #
                # 构建最终训练 entry。
                # ============================================================
                selected_frames[camera_name] = selected

                # ============================================================
                # 局部功能块：记录 fallback 匹配
                #
                # used_fallback：
                #
                #   False
                #       找到了 timestamp <= state 的历史帧。
                #
                #   True
                #       没有历史帧，只能使用 queue[0]。
                #
                # fallback_matches 后续进入：
                #
                #       alignment_summary()
                #
                # 用于评估同步数据质量。
                # ============================================================
                if used_fallback:
                    self.fallback_matches[camera_name] += 1

                # ============================================================
                # 局部功能块：计算当前 camera-state 时间差
                #
                # state_timestamp_ns：
                #   ← robot state["_capture_time_ns"]
                #
                # selected.timestamp_ns：
                #   ← CameraFrame
                #   ← ROS Image.header.stamp
                #
                #
                # delta_ns：
                #
                #       state time - camera time
                #
                # 正常 causal frame 一般：
                #
                #       delta_ns >= 0
                #
                # fallback 使用 state 之后的图片时可能：
                #
                #       delta_ns < 0
                #
                #
                # 统计指标只关心偏差大小，
                # 所以再计算：
                #
                #       abs_delta_ns = abs(delta_ns)
                # ============================================================
                delta_ns = state_timestamp_ns - selected.timestamp_ns
                abs_delta_ns = abs(delta_ns)

                # ============================================================
                # 局部功能块：累计 alignment 统计
                #
                # alignment_count：
                #   匹配次数 +1
                #
                # alignment_abs_sum_ns：
                #   累加绝对误差，
                #   后续用于计算平均值。
                #
                # alignment_abs_max_ns：
                #   保存目前最大的单次时间误差。
                #
                #
                # 后续统一由：
                #
                #       alignment_summary()
                #
                # 转换成 ms。
                # ============================================================
                self.alignment_count[camera_name] += 1
                self.alignment_abs_sum_ns[camera_name] += abs_delta_ns
                self.alignment_abs_max_ns[camera_name] = max(
                    self.alignment_abs_max_ns[camera_name],
                    abs_delta_ns,
                )

            # ================================================================
            # 局部功能块：从 robot state 构建最终数据 entry
            #
            # 输入：
            #
            #       state
            #
            # 来源：
            #
            #       x86 teleoperation process
            #           ↓
            #       episode_worker
            #           ↓
            #       add_state()
            #           ↓
            #       pending_states
            #
            #
            # 当前处理：
            #
            # 删除 Recorder 内部同步字段：
            #
            #       _episode_id
            #       _frame_index
            #       _capture_time_ns
            #
            # 其他 robot observation/action 字段保持原值。
            #
            #
            # 输出：
            #
            #       entry
            #
            # 此时 entry 还没有 image，
            # 下一功能块再添加。
            # ================================================================
            entry = {
                key: value
                for key, value in state.items()
                if key not in self.INTERNAL_STATE_KEYS
            }

            # ================================================================
            # 局部功能块：把三路已经对齐的 image 添加到 robot entry
            #
            # selected_frames 来源：
            #
            #       _select_frame()
            #
            #
            # selected_frames[camera_name].image：
            #
            #       CameraFrame.image
            #           ← image_decoder.py
            #           ← ROS sensor_msgs/Image.data
            #
            #
            # 最终生成：
            #
            #       entry["image"] = {
            #           "head": {
            #               "color": <np.ndarray>
            #           },
            #           "left_wrist": {
            #               "color": <np.ndarray>
            #           },
            #           "right_wrist": {
            #               "color": <np.ndarray>
            #           },
            #       }
            #
            #
            # 为什么 .copy()：
            #
            #   给最终 episode entry 保存独立的 ndarray 数据，
            #   避免它继续依赖 CameraFrame 中原来的数组对象。
            #
            #
            # 输出 entry 后续：
            #
            #       self.episode_data.append(entry)
            # ================================================================
            entry["image"] = {
                camera_name: {
                    "color": selected_frames[camera_name].image.copy(),
                }
                for camera_name in self.camera_names
            }

            # ================================================================
            # 局部功能块：提交一条完整的同步数据
            #
            # entry 此时已经同时包含：
            #
            #       robot state/action
            #       +
            #       对齐后的三路 camera image
            #
            # append 后进入：
            #
            #       self.episode_data
            #
            #
            # episode 完成后：
            #
            #       episode_worker.py
            #           ↓
            #       synchronizer.episode_data
            #           ↓
            #       dataset_writer.write_episode_pickle()
            #           ↓
            #       <episode_id>.pkl
            # ================================================================
            self.episode_data.append(entry)

            # ================================================================
            # 局部功能块：从 pending queue 删除已经处理完成的 state
            #
            # 当前 state：
            #
            #       self.pending_states[0]
            #
            # 已经生成对应 entry，
            # 因此不再需要等待 camera。
            #
            # popleft 后下一轮 while：
            #
            #       self.pending_states[0]
            #
            # 就变成下一条 robot state。
            # ================================================================
            self.pending_states.popleft()

            # ================================================================
            # 局部功能块：增加本次 merge_ready 的完成计数
            #
            # 最后：
            #
            #       return merged
            #
            # 告诉调用者这一次实际消费了多少条 pending state。
            # ================================================================
            merged += 1

            # ================================================================
            # 局部功能块：清理已经不可能再次使用的旧 camera frame
            #
            # selected_frames：
            #
            #       当前 state 实际选中的 frame。
            #
            # 对每一路 camera 调用：
            #
            #       _prune_before_selected()
            #
            # 删除 selected 之前更老的帧。
            #
            #
            # 例如：
            #
            #       [33, 66, 99]
            #
            # selected = 66
            #
            # 变成：
            #
            #       [66, 99]
            #
            #
            # selected 自身保留下来，
            # 因为下一条 state 仍可能使用它。
            #
            # 清理后的 self.frames 会直接传给下一轮：
            #
            #       can_match()
            #       _select_frame()
            #
            # 同时避免 camera queue 随 episode 持续无限增长。
            # ================================================================
            for camera_name, selected in selected_frames.items():
                self._prune_before_selected(
                    self.frames[camera_name],
                    selected,
                )

        # ====================================================================
        # 局部功能块：返回本轮完成的 state 数量
        #
        # merged：
        #
        #       当前这一次 merge_ready() 调用
        #       成功生成了多少条 episode_data entry。
        #
        # 正常实时循环里 episode_worker 并不依赖这个数值做保存，
        # 但它可以用于调用者判断本次是否发生了实际合并，
        # 测试代码也会使用这个返回值验证同步行为。
        # ====================================================================
        return merged

    def alignment_summary(self) -> dict:
        # ====================================================================
        # alignment_summary()
        #
        # 作用：
        #
        #       把整个 episode 累积的 camera-state 对齐统计
        #       整理成容易读取的毫秒级 summary。
        #
        #
        # 调用来源：
        #
        #       episode_worker.py
        #
        # episode 保存完成以后发送：
        #
        #       {
        #           "type": "saved",
        #           ...
        #           "alignment": synchronizer.alignment_summary(),
        #       }
        #
        #
        # 因此这个结果主要是：
        #
        #       “录制质量诊断信息”
        #
        # 而不是 episode_data 本身。
        # ====================================================================

        # ====================================================================
        # 局部功能块：创建最终统计字典
        #
        # 最终结构：
        #
        #       {
        #           "head": {...},
        #           "left_wrist": {...},
        #           "right_wrist": {...},
        #       }
        # ====================================================================
        summary = {}

        # ====================================================================
        # 局部功能块：逐 camera 汇总统计
        #
        # name：
        #
        #       ← self.camera_names
        #
        # 对每一路独立计算，
        # 因为不同 camera 的延迟和丢帧情况可能完全不同。
        # ====================================================================
        for name in self.camera_names:

            # ================================================================
            # 局部功能块：取得该 camera 的有效匹配次数
            #
            # 来源：
            #
            #       merge_ready()
            #
            # 每匹配一条 state：
            #
            #       alignment_count[name] += 1
            #
            #
            # count 后续既用于：
            #
            #   ① summary["count"]
            #
            #   ② 计算平均绝对时间误差
            # ================================================================
            count = self.alignment_count[name]

            # ================================================================
            # 局部功能块：计算平均绝对对齐误差
            #
            # alignment_abs_sum_ns：
            #
            #   merge_ready() 累积的：
            #
            #       Σ |state_timestamp - camera_timestamp|
            #
            # 除以 count：
            #
            #       平均误差，单位 ns
            #
            # 再除：
            #
            #       1_000_000
            #
            # 转换：
            #
            #       ns → ms
            #
            #
            # 如果 count == 0：
            #
            #       average_ms = 0.0
            #
            # 避免除零。
            # ================================================================
            average_ms = (
                self.alignment_abs_sum_ns[name]
                / count
                / 1_000_000.0
                if count
                else 0.0
            )

            # ================================================================
            # 局部功能块：构建单路 camera 的 summary
            #
            # count：
            #   完成多少次 camera-state match。
            #
            # mean_abs_delta_ms：
            #   平均绝对时间差。
            #
            # max_abs_delta_ms：
            #   整个 episode 最大绝对时间差。
            #
            # fallback_matches：
            #   没有 timestamp <= state 的历史帧，
            #   因而使用最早可用帧的次数。
            #
            # dropped_out_of_order_images：
            #   add_camera_frame() 因时间戳倒退而丢弃的帧数。
            #
            #
            # 输出：
            #
            #       summary[name]
            #
            # 最终整个 summary 返回 episode_worker。
            # ================================================================
            summary[name] = {
                "count": count,
                "mean_abs_delta_ms": average_ms,
                "max_abs_delta_ms": (
                    self.alignment_abs_max_ns[name]
                    / 1_000_000.0
                ),
                "fallback_matches": self.fallback_matches[name],
                "dropped_out_of_order_images": (
                    self.dropped_out_of_order_images[name]
                ),
            }

        # ====================================================================
        # 局部功能块：返回整个 episode 的 alignment 质量信息
        #
        # 去向：
        #
        #       episode_worker.py
        #           ↓
        #       control_connection.send({
        #           "type": "saved",
        #           ...
        #           "alignment": summary,
        #       })
        #           ↓
        #       RecorderServer
        #
        # 因此可以在 episode 保存之后看到三路相机各自的：
        #
        #       平均时间误差
        #       最大时间误差
        #       fallback 次数
        #       乱序丢帧次数
        # ====================================================================
        return summary