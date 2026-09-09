from __future__ import annotations

# ============================================================================
# 当前文件：tiangong_recorder/synchronizer.py
#
# 【这个文件在整个项目中的职责】
#
# 这个文件负责：
#
#     把“按时间到达的三路相机帧”
#
# 和
#
#     “按时间到达的机器人 state/action”
#
# 根据时间戳进行对齐，最终生成可以保存到 episode PKL 中的一帧帧数据。
#
#
# 整体数据流大致为：
#
#   ROS 2 camera topics
#          │
#          ▼
#   episode_worker.py
#          │
#          │ decode_ros_image()
#          ▼
#      CameraFrame
#          │
#          │ add_camera_frame()
#          ▼
#   ┌──────────────────────┐
#   │                      │
#   │ LiveEpisodeSynchronizer
#   │                      │
#   └──────────────────────┘
#          ▲
#          │ add_state()
#          │
#   robot state/action
#          ▲
#          │
#   episode_worker.py
#          ▲
#          │ ZMQ PULL
#          │
#   x86 teleoperation process
#
#
# 然后：
#
#   CameraFrame queues
#          │
#          ├───────────┐
#          │           │
#          ▼           ▼
#      merge_ready()  pending_states
#          │
#          │ 按 _capture_time_ns 对齐
#          ▼
#      episode_data
#          │
#          ▼
#   episode_worker.py
#          │
#          ▼
#   write_episode_pickle()
#          │
#          ▼
#       *.pkl
#
#
# 所以这个文件可以理解为整个录制系统中的：
#
#     “相机数据和机器人状态数据的时间同步/融合模块”
#
# 它本身：
#
#   × 不订阅 ROS topic
#   × 不接收 ZMQ 网络数据
#   × 不写 PKL
#
# 它只负责：
#
#   接收已经拿到的 CameraFrame 和 state
#       ↓
#   缓存
#       ↓
#   根据 timestamp 对齐
#       ↓
#   生成 episode_data
# ============================================================================


# ============================================================================
# deque
#
# 来源：
#   Python 标准库 collections。
#
#
# 【它是什么】
#
# deque = double-ended queue，双端队列。
#
# 相比普通 list：
#
#   deque.popleft()
#
# 可以高效地从队列最左侧删除元素。
#
#
# 当前文件中主要用在两个地方：
#
#   1. 每一路相机自己的 frame queue
#
#       self.frames["head"]
#       self.frames["left_wrist"]
#       self.frames["right_wrist"]
#
#
#   2. 等待和相机对齐的 robot state queue
#
#       self.pending_states
#
#
# 为什么这里适合用 deque：
#
#   数据都是按时间顺序进入；
#
#   处理时也是从最旧的数据开始；
#
#   处理完成后需要频繁：
#
#       popleft()
#
#   删除已经不再需要的数据。
# ============================================================================
from collections import deque


# ============================================================================
# dataclass
#
# 来源：
#   Python 标准库 dataclasses。
#
#
# 当前用于定义：
#
#       CameraFrame
#
#
# CameraFrame 本质上是把一帧相机数据需要的：
#
#       timestamp
#       image
#       encoding
#
# 打包成一个对象，
# 避免在项目中到处分别传三个变量。
# ============================================================================
from dataclasses import dataclass


# ============================================================================
# Deque / Dict / Iterable
#
# 来源：
#   Python 标准库 typing。
#
#
# 主要用于类型标注。
#
#
# Deque[CameraFrame]
#
#   表示：
#
#       一个双端队列，
#       其中每个元素都是 CameraFrame。
#
#
# Dict[str, Deque[CameraFrame]]
#
#   表示：
#
#       key：
#           camera name
#
#       value：
#           该 camera 对应的 CameraFrame 队列
#
#
# Iterable[str]
#
#   表示 camera_names 可以是任何可迭代字符串集合，
#   例如：
#
#       list
#       tuple
#       dict_keys
#
#
# 当前项目中实际传进来的通常是：
#
#       config.camera_topics.keys()
#
# 也就是：
#
#       head
#       left_wrist
#       right_wrist
# ============================================================================
from typing import Deque, Dict, Iterable


# ============================================================================
# numpy
#
# 来源：
#   第三方数值计算库 NumPy。
#
#
# 当前文件本身并没有进行复杂的 NumPy 运算，
# 主要使用：
#
#       np.ndarray
#
# 表示一帧已经解码好的图像。
#
#
# 图像 ndarray 的真正创建发生在：
#
#       tiangong_recorder/image_decoder.py
#
# 其中：
#
#       ROS sensor_msgs/Image
#               │
#               ▼
#       np.frombuffer(...)
#               │
#               ▼
#       reshape(height, width, 3)
#               │
#               ▼
#           np.ndarray
#               │
#               ▼
#          CameraFrame.image
#
#
# 当前 synchronizer.py 主要负责保存和复制这个 ndarray，
# 而不负责解码 ROS Image。
# ============================================================================
import numpy as np


# ============================================================================
# CameraFrame
#
# 定义来源：
#
#       当前 synchronizer.py
#
#
# 【它是什么】
#
# CameraFrame 是整个 Recorder 内部表示“一帧已经解码好的相机图像”的对象。
#
#
# 【它通常在哪里创建】
#
#       tiangong_recorder/image_decoder.py
#
# 中的：
#
#       decode_ros_image(...)
#
#
# 数据链：
#
#   ROS sensor_msgs/Image
#          │
#          ▼
#   decode_ros_image()
#          │
#          ├── 解析 timestamp
#          ├── 检查 width / height
#          ├── 检查 rgb8 / bgr8
#          ├── np.frombuffer()
#          └── reshape()
#          │
#          ▼
#      CameraFrame
#
#
# 【然后传到哪里】
#
# episode_worker.py 的相机 callback：
#
#       frame = decode_ros_image(...)
#                   │
#                   ▼
#       synchronizer.add_camera_frame(
#           camera_name,
#           frame,
#       )
#
#
# 所以 CameraFrame 是：
#
#       image_decoder.py
#
# 和
#
#       synchronizer.py
#
# 之间的相机数据传输格式。
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
    # 优先使用：
    #
    #   ROS Image.header.stamp
    #
    # 即：
    #
    #   stamp.sec * 1_000_000_000
    #       +
    #   stamp.nanosec
    #
    #
    # 如果 ROS timestamp 无效，
    # image_decoder 可以使用 fallback timestamp。
    #
    #
    # 单位：
    #   ns，纳秒。
    #
    #
    # 后续用途：
    #
    #   add_camera_frame()
    #       ↓
    #   检查相机帧时间是否单调
    #
    #   can_match()
    #       ↓
    #   判断是否已经拥有足够新的相机数据
    #
    #   _select_frame()
    #       ↓
    #   和 state["_capture_time_ns"] 做时间对齐
    # ========================================================================
    timestamp_ns: int

    # ========================================================================
    # image
    #
    # 来源：
    #
    #   image_decoder.decode_ros_image()
    #
    #
    # 类型：
    #
    #   np.ndarray
    #
    # 一般结构：
    #
    #   [height, width, 3]
    #
    #
    # 后续去向：
    #
    #   CameraFrame
    #       ↓
    #   self.frames[camera_name]
    #       ↓
    #   _select_frame()
    #       ↓
    #   selected_frames
    #       ↓
    #   entry["image"][camera_name]["color"]
    #       ↓
    #   self.episode_data
    #       ↓
    #   write_episode_pickle()
    # ========================================================================
    image: np.ndarray

    # ========================================================================
    # encoding
    #
    # 来源：
    #
    #   ROS Image.encoding
    #       ↓
    #   image_decoder.decode_ros_image()
    #
    # 当前 image_decoder 支持：
    #
    #   rgb8
    #   bgr8
    #
    #
    # 当前 synchronizer.py 中：
    #
    #   encoding 不参与时间同步，
    #   merge_ready() 最终也只把 image 写入 entry。
    #
    # 所以它目前主要是 CameraFrame 所携带的图像格式元信息。
    # ========================================================================
    encoding: str


# ============================================================================
# LiveEpisodeSynchronizer
#
# 定义来源：
#
#       当前 synchronizer.py
#
#
# 【它是什么】
#
# 这是一次 episode 内：
#
#       robot state
#
# 和
#
#       多路 camera frame
#
# 的实时时间同步器。
#
#
# 【在哪里创建】
#
#       episode_worker.py
#
# 中：
#
#       synchronizer = LiveEpisodeSynchronizer(
#           config.camera_topics.keys()
#       )
#
#
# config.camera_topics 来源：
#
#       config/recorder.yaml
#
# 当前通常包含：
#
#       head
#       left_wrist
#       right_wrist
#
#
# 【它接收两条数据流】
#
# 第一条：相机
#
#   ROS camera
#       ↓
#   EpisodeCameraNode._camera_callback()
#       ↓
#   decode_ros_image()
#       ↓
#   CameraFrame
#       ↓
#   add_camera_frame()
#       ↓
#   self.frames
#
#
# 第二条：robot state/action
#
#   x86 teleoperation process
#       ↓
#   ZMQ
#       ↓
#   episode_worker.state_socket.recv_pyobj()
#       ↓
#   state dict
#       ↓
#   add_state()
#       ↓
#   self.pending_states
#
#
# 【真正同步发生在哪里】
#
#       merge_ready()
#
#
# 【同步后的结果】
#
#       self.episode_data
#
# 最终被 episode_worker 传给：
#
#       write_episode_pickle(
#           episode_id,
#           synchronizer.episode_data,
#           config.output_dir,
#       )
#
#
# 所以这个类的核心关系可以理解成：
#
#           camera queues
#               │
#               │
#               ▼
#          merge_ready()
#               ▲
#               │
#               │
#         pending_states
#               │
#               ▼
#          episode_data
# ============================================================================
class LiveEpisodeSynchronizer:
    """Causally aligns ordered camera frames with ordered robot states."""

    # ========================================================================
    # INTERNAL_STATE_KEYS
    #
    # 【来源】
    #
    # 这些 key 来自 x86 发给 episode_worker 的 robot state dict。
    #
    # 当前 synchronizer 依赖其中至少：
    #
    #   _capture_time_ns
    #
    # 来执行时间同步。
    #
#
    # episode_worker 还会直接使用：
    #
    #   _episode_id
    #       检查 state 属于当前 episode。
    #
    #   _frame_index
    #       检查 state 是否连续。
    #
#
    # 【为什么叫 INTERNAL_STATE_KEYS】
    #
    # 这些字段是 Recorder 内部进行：
    #
    #   episode 校验
    #   frame 顺序校验
    #   时间同步
    #
    # 使用的“内部元数据”。
    #
    # 它们不是最终机器人训练样本中的业务 state/action。
    #
#
    # 【最终去向】
    #
    # merge_ready() 创建 entry 时：
    #
    #       if key not in self.INTERNAL_STATE_KEYS
    #
    # 会把这些内部字段过滤掉。
    #
   #
    # 因此：
    #
    #   原始 state
    #
    #       {
    #           "_episode_id": ...,
    #           "_frame_index": ...,
    #           "_capture_time_ns": ...,
    #           robot_state: ...,
    #           action: ...,
    #       }
    #
    # 最终 episode_data entry 中不会保留这些 "_" 内部 key。
    # ========================================================================
    INTERNAL_STATE_KEYS = {
        "_episode_id",
        "_frame_index",
        "_capture_time_ns",
    }

    def __init__(self, camera_names: Iterable[str]):
        # ====================================================================
        # __init__()
        #
       # 调用来源：
        #
        #   episode_worker.py
        #
        #       LiveEpisodeSynchronizer(
        #           config.camera_topics.keys()
        #       )
        #
       #
        # camera_names 实际来源：
        #
        #   config/recorder.yaml
        #           ↓
        #   RecorderConfig.camera_topics
        #           ↓
        #   config.camera_topics.keys()
        #
       # 当前默认通常为：
        #
        #   head
        #   left_wrist
        #   right_wrist
        #
       #
        # __init__ 的目标：
        #
        #   为当前 episode 初始化：
        #
        #       相机缓存
        #       state 缓存
        #       最终 episode_data
        #       对齐统计信息
        # ====================================================================

        # ====================================================================
        # 局部功能块：固定当前 episode 使用的 camera 名称
        #
        # 输入：
        #
        #   camera_names
        #       ← config.camera_topics.keys()
        #
       #
        # 当前处理：
        #
        #   tuple(camera_names)
        #
       # 把可能是 dict_keys / list 等 Iterable
        # 固定成不可变顺序的 tuple。
        #
       #
        # 例如：
        #
        #   (
        #       "head",
        #       "left_wrist",
        #       "right_wrist",
        #   )
        #
       #
        # 输出：
        #
        #   self.camera_names
        #
       #
        # 后续几乎整个类都会使用：
        #
        #   ├── 初始化 self.frames
        #   ├── cameras_ready()
        #   ├── can_match()
        #   ├── merge_ready()
        #   └── alignment_summary()
        #
       # 来确保所有 camera 都参与同步。
        # ====================================================================
        self.camera_names = tuple(camera_names)

        # ====================================================================
        # 局部功能块：为每一路 camera 创建独立的 frame queue
        #
        # 输入：
        #
        #   self.camera_names
        #
       #
        # 当前处理：
        #
        #   每一个 camera_name：
        #
        #       name → deque()
        #
       # 构造：
        #
        #   self.frames
        #
       # 结构类似：
        #
        #   {
        #       "head": deque([...]),
        #       "left_wrist": deque([...]),
        #       "right_wrist": deque([...]),
        #   }
        #
       #
        # 每个 deque 内按照 timestamp 顺序保存：
        #
        #   CameraFrame
        #
       #
        # 数据之后从哪里进入：
        #
        #   episode_worker camera callback
        #       ↓
        #   add_camera_frame()
        #       ↓
        #   self.frames[camera_name].append(frame)
        #
       #
        # 数据之后到哪里：
        #
        #   cameras_ready()
        #   can_match()
        #   _select_frame()
        #   merge_ready()
        #   _prune_before_selected()
        #
       # 共同使用这些 frame queues。
        # ====================================================================
        self.frames: Dict[str, Deque[CameraFrame]] = {
            name: deque() for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：初始化待匹配 robot state 队列
        #
        # 初始：
        #
        #   空 deque。
        #
       #
        # 数据来源：
        #
        #   episode_worker.py
        #
        #       state = state_socket.recv_pyobj()
        #           ↓
        #       synchronizer.add_state(state)
        #
       #
        # add_state() 最终：
        #
        #   self.pending_states.append(state)
        #
       #
        # 数据去向：
        #
        #   merge_ready()
        #
       # 会始终从：
        #
        #   self.pending_states[0]
        #
       # 即最早一个还没有匹配相机的 state 开始处理。
        #
       # 匹配完成后：
        #
        #   self.pending_states.popleft()
        # ====================================================================
        self.pending_states: Deque[dict] = deque()

        # ====================================================================
        # 局部功能块：初始化最终 episode 样本列表
        #
        # episode_data：
        #
        #   最终完成：
        #
        #       robot state/action
        #           +
        #       三路同步图像
        #
        # 后形成的一帧帧 episode sample。
        #
       #
        # 初始为空：
        #
        #   []
        #
       #
        # 数据来源：
        #
        #   merge_ready()
        #
       # 每成功匹配一个 state：
        #
        #   self.episode_data.append(entry)
        #
       #
        # 最终去向：
        #
        #   episode_worker.py
        #
        #       write_episode_pickle(
        #           episode_id,
        #           synchronizer.episode_data,
        #           config.output_dir,
        #       )
        #
       # 因此它实际上就是当前 episode 最终要保存的主体数据。
        # ====================================================================
        self.episode_data: list[dict] = []

        # ====================================================================
        # 局部功能块：记录“乱序相机帧”的丢弃数量
        #
        # 初始结构：
        #
        #   {
        #       "head": 0,
        #       "left_wrist": 0,
        #       "right_wrist": 0,
        #   }
        #
       #
        # 更新位置：
        #
        #   add_camera_frame()
        #
       # 如果新 frame：
        #
        #   frame.timestamp_ns
        #
       # 小于 queue 最后一帧 timestamp：
        #
        #   说明发生时间乱序，
       #   当前 frame 不加入 queue，
       #   这个计数 +1。
        #
       #
        # 最终去向：
        #
        #   alignment_summary()
        #
       # 作为：
        #
        #   dropped_out_of_order_images
        #
       # 返回。
        #
       # episode_worker 保存完成后会把 alignment_summary()
       # 放进 `"saved"` 状态消息返回 RecorderServer。
        # ====================================================================
        self.dropped_out_of_order_images: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：统计 fallback frame 使用次数
        #
        # fallback 的含义：
        #
       # 某一个 state timestamp：
        #
        #       T_state
        #
       # 理想情况下希望找到：
        #
        #       timestamp <= T_state
        #
       # 的最新 camera frame。
        #
       # 如果 camera queue 中所有 frame 都比 state 新：
        #
        #       frame.timestamp > T_state
        #
       # 就不存在过去/同时刻的帧。
        #
       # 此时 _select_frame() 会退化选择：
        #
        #       queue[0]
        #
       # 即当前最早的一帧，
       # 并返回：
        #
        #       used_fallback = True
        #
       #
        # 当前计数：
        #
        #   self.fallback_matches[camera_name] += 1
        #
       # 最终进入 alignment_summary()。
        # ====================================================================
        self.fallback_matches: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：统计每一路 camera 完成了多少次对齐
        #
        # 每次 merge_ready() 成功处理一个 state：
        #
        #   每个 camera 都选择一帧，
        #   因此：
        #
        #       alignment_count[camera] += 1
        #
       #
        # 最终用于：
        #
        #   alignment_summary()
        #
       # 计算：
        #
        #   mean_abs_delta_ms
        # ====================================================================
        self.alignment_count: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：累计 state 与 camera frame 的绝对时间差
        #
        # merge_ready() 中：
        #
        #   delta_ns =
        #       state_timestamp_ns
        #       -
        #       selected.timestamp_ns
        #
        #   abs_delta_ns = abs(delta_ns)
        #
       #
        # 每次累加：
        #
        #   alignment_abs_sum_ns[camera] += abs_delta_ns
        #
       #
        # 最终：
        #
        #   sum / count
        #
       # 得到每一路 camera 平均时间对齐误差。
        # ====================================================================
        self.alignment_abs_sum_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

        # ====================================================================
        # 局部功能块：记录最大绝对时间对齐误差
        #
        # 每次 merge_ready()：
        #
        #   max(
        #       当前最大值,
        #       当前 abs_delta_ns
        #   )
        #
       # 最终：
        #
        #   alignment_summary()
        #
       # 返回：
        #
        #   max_abs_delta_ms
        #
       # 用于观察某一路 camera 最差的同步偏差。
        # ====================================================================
        self.alignment_abs_max_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

    def add_camera_frame(self, camera_name: str, frame: CameraFrame) -> bool:
        # ====================================================================
        # add_camera_frame()
        #
       # 【调用来源】
        #
        #   episode_worker.py
        #
       # 相机 callback 内：
        #
        #   ROS Image
        #       ↓
        #   decode_ros_image()
        #       ↓
        #   frame: CameraFrame
        #       ↓
        #   synchronizer.add_camera_frame(
        #       camera_name,
        #       frame,
        #   )
        #
       #
        # camera_name 来源：
        #
        #   config.camera_topics
        #
       # 例如：
        #
        #   "head"
        #   "left_wrist"
        #   "right_wrist"
        #
       #
        # frame 来源：
        #
        #   image_decoder.decode_ros_image()
        #
       #
        # 【函数目标】
        #
       # 把新的 CameraFrame 加入对应 camera 的时间有序队列。
        #
       # 如果发现 timestamp 倒退，
       # 就丢弃该帧。
        # ====================================================================

        # ====================================================================
        # 局部功能块：找到当前 camera 对应的 frame queue
        #
        # 输入：
        #
        #   camera_name
        #       ← episode_worker callback
        #
       # self.frames：
        #   ← __init__()
        #
       #
        # 例如：
        #
        #   camera_name = "head"
        #
       # 得到：
        #
        #   queue = self.frames["head"]
        #
       #
        # queue 后续用于：
        #
        #   检查最后一帧 timestamp
        #   append 新 frame
        # ====================================================================
        queue = self.frames[camera_name]

        # ====================================================================
        # 局部功能块：检查新相机帧是否发生时间乱序
        #
        # queue：
        #   当前 camera 已经缓存的 frame。
        #
       # queue[-1]：
        #   当前最新一帧。
        #
       # frame：
        #   新到达的 CameraFrame。
        #
       #
        # 判断：
        #
        #   frame.timestamp_ns
        #       <
        #   queue[-1].timestamp_ns
        #
       # 如果成立：
        #
        #   新来的帧 timestamp 比已经收到的最后一帧还早，
       #   说明输入不是单调时间顺序。
        #
       #
        # 当前处理：
        #
        #   dropped_out_of_order_images += 1
        #
       # 并：
        #
        #   return False
        #
       #
        # 也就是说：
        #
        #   这个 frame 不会进入 self.frames，
       #   后续 merge_ready() 完全不会看到它。
        # ====================================================================
        if queue and frame.timestamp_ns < queue[-1].timestamp_ns:
            self.dropped_out_of_order_images[camera_name] += 1
            return False

        # ====================================================================
        # 局部功能块：加入当前 camera 的有序缓存
        #
        # 到这里说明：
        #
        #   queue 为空
        #
       # 或：
        #
        #   frame.timestamp_ns >= 最后一帧 timestamp
        #
       #
        # 因此可以：
        #
        #   queue.append(frame)
        #
       #
        # 数据去向：
        #
        #   self.frames[camera_name]
        #       ↓
        #   cameras_ready()
        #       判断所有 camera 是否已经有数据
        #
        #   can_match()
        #       判断是否有足够时间范围的数据
        #
        #   _select_frame()
        #       为某个 state 选择 camera frame
        #
       # 最终进入：
        #
        #   merge_ready()
        # ====================================================================
        queue.append(frame)

        # ====================================================================
        # True：
        #
        #   表示当前 frame 已经成功加入同步缓存。
        #
       # 当前 episode_worker 没有使用这个返回值，
       # 但接口本身可以让调用者判断：
        #
        #   True  → accepted
        #   False → timestamp 乱序被丢弃
        # ====================================================================
        return True

    def add_state(self, state: dict) -> None:
        # ====================================================================
        # add_state()
        #
       # 【调用来源】
        #
        # episode_worker.py：
        #
        #   state_socket.poll()
        #       ↓
        #   state_socket.recv_pyobj()
        #       ↓
        #   state
        #
       # episode_worker 先检查：
        #
        #   state["_episode_id"]
        #   state["_frame_index"]
        #
       # 然后：
        #
        #   synchronizer.add_state(state)
        #
       #
        # 所以 state 的更上游来源是：
        #
        #   x86 teleoperation process
        #       ↓ ZMQ
        #   episode_worker state PULL socket
        #       ↓
        #   当前函数
        #
       #
        # 【函数目标】
        #
       # 保证 state 的 _capture_time_ns 单调，
       # 然后把它加入 pending_states，
       # 等待相机数据与它对齐。
        # ====================================================================

        # ====================================================================
        # 局部功能块：提取当前 state 的采集 timestamp
        #
        # state：
        #   ← x86 发送过来的 state dict。
        #
       # _capture_time_ns：
        #
        #   Recorder 内部使用的 state capture timestamp。
        #
       #
        # 当前处理：
        #
        #   int(...)
        #
       # 确保后续时间比较使用整数。
        #
       #
        # timestamp_ns 去向：
        #
       # 下一段用于检查：
        #
        #   当前 state timestamp
       #   是否小于上一条 pending state timestamp。
        # ====================================================================
        timestamp_ns = int(state["_capture_time_ns"])

        # ====================================================================
        # 局部功能块：保证 robot state timestamp 单调递增
        #
       # self.pending_states：
        #   ← __init__() 创建。
        #
       # 如果里面已经有 state：
        #
        #   self.pending_states[-1]
        #
       # 就是最近加入的上一条未处理 state。
        #
       #
        # previous：
        #
        #   ← 上一条 state["_capture_time_ns"]
        #
       #
        # 判断：
        #
        #   当前 timestamp_ns < previous
        #
       # 如果成立：
        #
        #   robot state 时间顺序发生倒退。
        #
       # 这会破坏后面按队列顺序执行的因果匹配逻辑，
       # 所以直接：
        #
        #   raise ValueError
        #
       #
        # 异常会继续传回：
        #
        #   episode_worker
        #       ↓
       #   worker except Exception
        #       ↓
        #   {"type": "error"}
        #       ↓ Pipe
        #   RecorderServer
        # ====================================================================
        if self.pending_states:
            previous = int(self.pending_states[-1]["_capture_time_ns"])

            if timestamp_ns < previous:
                raise ValueError("state timestamps must be monotonic")

        # ====================================================================
        # 局部功能块：加入等待同步的 state queue
        #
        # state：
       #   已通过 timestamp 顺序检查。
        #
       # 当前处理：
        #
        #   pending_states.append(state)
        #
       #
        # 数据去向：
        #
        #   merge_ready()
        #
       # 其中始终先取：
        #
        #   self.pending_states[0]
        #
       # 即最早一个尚未匹配相机的 state。
        #
       # 成功匹配之后：
        #
        #   pending_states.popleft()
        # ====================================================================
        self.pending_states.append(state)

    def cameras_ready(self) -> bool:
        # ====================================================================
        # cameras_ready()
        #
       # 作用：
        #
        #   判断所有需要的 camera 是否至少已经缓存了一帧图像。
        #
       #
        # self.camera_names：
        #
        #   ← __init__()
        #
       # self.frames[name]：
        #
        #   ← add_camera_frame() 持续写入。
        #
       #
        # 例如：
        #
        #   head queue        非空
        #   left_wrist queue  非空
        #   right_wrist queue 非空
        #
       # 才返回 True。
        #
       #
        # 调用去向：
        #
        #   1. episode_worker
        #
       #       synchronizer.cameras_ready()
        #
       #       用来判断 worker 是否可以向 RecorderServer
       #       发送 ready。
        #
       #   2. can_match()
        #
       #       检查是否具备匹配基础。
        #
       #   3. merge_ready()
        #
       #       如果 camera 尚未全部 ready，
       #       暂停处理 pending state。
        # ====================================================================
        return all(self.frames[name] for name in self.camera_names)

    def can_match(self, state_timestamp_ns: int) -> bool:
        # ====================================================================
        # can_match()
        #
       # 输入来源：
        #
        #   state_timestamp_ns
        #
        # 通常来自 merge_ready()：
        #
        #   state = pending_states[0]
        #       ↓
        #   state["_capture_time_ns"]
        #
       #
        # 【这个函数解决的问题】
        #
       # 假设现在要处理：
        #
        #   state timestamp = 100
        #
       # 某 camera 当前只收到：
        #
        #   80
        #   90
        #
       # 那么现在还不能确定：
        #
        #   90
        #
       # 就是 state=100 最合适的历史 frame。
        #
       # 因为下一帧：
        #
        #   99
        #
       # 可能马上到。
        #
       # 所以这里要求：
        #
        #   每一路 camera 的“最新一帧”
        #
       # timestamp 都已经：
        #
        #   >= state_timestamp_ns
        #
       # 这样说明该 camera 的数据时间线已经至少走过 state 时刻，
       # 才可以稳定地从历史 frame 中选择最后一个 <= state timestamp 的帧。
        #
       #
        # 这就是类注释中：
        #
        #   "Causally aligns"
        #
       # 的核心之一。
        # ====================================================================

        # ====================================================================
        # 第一部分：
        #
        #   self.cameras_ready()
        #
       # 确保每一路 camera queue 至少非空。
        #
       #
        # 第二部分：
        #
       # 对每个 camera：
        #
        #   self.frames[name][-1]
        #
       # 是最新 frame。
        #
       # 要求：
        #
        #   latest.timestamp_ns >= state_timestamp_ns
        #
       #
        # 如果任意一路 camera 还没有走到该 state timestamp：
        #
        #   return False
        #
       #
        # 输出去向：
        #
        #   merge_ready()
        #
       # 如果 False：
        #
        #   break
        #
       # 当前 state 暂时继续留在 pending_states，
       # 等更多 camera frame 到来。
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
       # staticmethod：
        #
       # 这个函数不访问：
        #
        #   self
        #
       # 只根据：
        #
        #   一个 camera queue
        #   一个 state timestamp
        #
       # 完成 frame 选择。
        #
       #
        # 调用来源：
        #
        #   merge_ready()
        #
       # 对每一路 camera 都调用一次。
        #
       #
        # 输入 queue：
        #
        #   self.frames[camera_name]
        #
       # 例如：
        #
        #   deque([
        #       frame@80,
        #       frame@90,
        #       frame@105,
        #   ])
        #
       #
        # state_timestamp_ns：
        #
        #   ← pending state["_capture_time_ns"]
        #
       #
        # 【选择原则】
        #
        # 优先选择：
        #
        #   timestamp <= state timestamp
        #
       # 中时间最晚的一帧。
        #
       # 也就是：
        #
        #   latest frame not after state
        #
       #
        # 这比简单“绝对时间最近”更偏向因果匹配：
       # 正常情况下尽量不使用 state 之后才采到的未来帧。
        # ====================================================================

        # ====================================================================
        # 局部功能块：从最新 frame 向过去反向查找
        #
        # reversed(queue)：
        #
        #   newest → oldest
        #
       #
        # 第一帧满足：
        #
        #   frame.timestamp_ns <= state_timestamp_ns
        #
       # 就一定是：
        #
        #   所有“不晚于 state”的 frame 中最新的一帧。
        #
       #
        # 例如：
        #
        #   camera：
        #
        #       80, 90, 105
        #
       #   state：
        #
        #       100
        #
       # 反向：
        #
        #       105 → 不满足
       #        90 → 满足
        #
       # 所以选择：
        #
        #       90
        #
       #
        # 返回：
        #
        #   (frame, False)
        #
       # False 表示：
        #
        #   没有使用 fallback。
        #
       #
        # 去向：
        #
        #   merge_ready()
        #
       # 将 frame 放入：
        #
        #   selected_frames[camera_name]
        # ====================================================================
        for frame in reversed(queue):
            if frame.timestamp_ns <= state_timestamp_ns:
                return frame, False

        # ====================================================================
        # 局部功能块：没有任何 frame 早于或等于 state
        #
        # 到这里意味着：
        #
        #   queue 中所有 frame：
        #
        #       timestamp > state_timestamp
        #
       #
        # 例如：
        #
        #   camera：
       #
        #       105, 120
        #
       #   state：
        #
        #       100
        #
       #
        # 没有“过去帧”可用。
        #
       # 当前策略：
        #
        #   queue[0]
        #
       # 选择现有数据中最早的一帧：
        #
        #   105
        #
       #
        # 同时返回：
        #
        #   True
        #
       # 表示使用了 fallback。
        #
       #
        # merge_ready() 收到：
        #
        #   used_fallback = True
        #
       # 后：
        #
        #   self.fallback_matches[camera_name] += 1
        #
       # 最终可以通过 alignment_summary()
       # 看到这种非理想匹配发生了多少次。
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
       # 当一个 camera frame 已经被选来匹配当前 state 后，
       # 清理它之前那些以后再也不可能需要的旧 frame。
        #
       #
        # 调用来源：
        #
        #   merge_ready()
        #
       # 当前 state 成功写入 episode_data 后：
        #
        #   for camera_name, selected in selected_frames.items():
        #
       #       _prune_before_selected(...)
        #
       #
        # 输入：
        #
        #   queue
        #       ← self.frames[camera_name]
        #
       #   selected
        #       ← _select_frame() 选出的 CameraFrame
        # ====================================================================

        # ====================================================================
        # 举例：
        #
        # queue：
        #
        #   [frame80, frame90, frame105, frame120]
        #
       # 当前 state 选择：
        #
        #   selected = frame90
        #
       #
        # 那么：
        #
        #   frame80
        #
       # 对后面的 state 已经没有保留价值，
       # 因为：
        #
       #   frame90 比它更新，
       #   并且已经不晚于当前处理进度。
        #
       #
        # 所以删除：
        #
        #   frame80
        #
       # 变成：
        #
        #   [frame90, frame105, frame120]
        #
       #
        # 注意：
        #
        #   selected 本身不会删除。
        #
       # 这很重要，因为：
        #
       #   frame90
        #
       # 可能仍然是下一条 state 最合适的 camera frame，
       # 因此允许一个 CameraFrame 被相邻的多个 state 重用。
        #
       #
        # 条件：
        #
       #   len(queue) > 1
        #
       # 防止把 queue 清空。
        #
       #
        # queue[0] is not selected：
        #
       # 只删除 selected 之前的元素；
       # 当 selected 移到 queue 第一位时停止。
        # ====================================================================
        while len(queue) > 1 and queue[0] is not selected:
            queue.popleft()

    def merge_ready(self, force: bool = False) -> int:
        # ====================================================================
        # merge_ready()
        #
       # 这是 LiveEpisodeSynchronizer 最核心的函数。
        #
       #
        # 【调用来源】
        #
        # episode_worker 主循环中：
        #
        #   synchronizer.merge_ready(force=False)
        #
       # 正常录制阶段不断调用。
        #
       #
        # STOP_SAVE 后，如果最后几个 state 一直无法等到理想 camera frame，
       # 到 tail_wait_timeout_s 后：
        #
        #   synchronizer.merge_ready(force=True)
        #
       #
        # 【输入数据来源】
        #
       # robot state：
        #
        #   self.pending_states
        #       ← add_state()
        #       ← episode_worker ZMQ state_socket
        #       ← x86 teleoperation
        #
       #
        # camera：
        #
        #   self.frames
        #       ← add_camera_frame()
        #       ← decode_ros_image()
        #       ← ROS Image topics
        #
       #
        # 【输出】
        #
        # 每成功匹配一个 state：
        #
        #   entry
        #
       # 进入：
        #
        #   self.episode_data
        #
       #
        # 最终：
        #
        #   episode_worker
        #       ↓
        #   write_episode_pickle()
        #
       #
        # 返回值：
        #
        #   merged
        #
       # 表示本次调用成功融合了多少条 state。
        # ====================================================================

        # ====================================================================
        # 局部功能块：初始化本次调用的成功融合计数
        #
        # merged 只统计：
        #
        #   当前这一次 merge_ready() 调用
        #
       # 处理了多少条 state。
        #
       # 它不同于：
        #
        #   len(self.episode_data)
        #
       # 后者是整个 episode 累积完成的数据量。
        # ====================================================================
        merged = 0

        # ====================================================================
        # 只要还有等待处理的 state，
       # 就尝试从最早一条开始连续处理。
        #
       # 为什么必须从最早的 state 开始：
        #
       # camera 和 state 都是时间序列，
       # 保证前面的 state 先完成，
       # 可以维持 episode_data 的时间顺序。
        # ====================================================================
        while self.pending_states:

            # ================================================================
            # 局部功能块：取最早一个待匹配 state
            #
            # self.pending_states：
            #   ← add_state()
            #
           # [0]：
            #
            #   只读取，不立即删除。
            #
           # 因为此时还不知道 camera 数据是否已经足够。
            #
           #
            # state 只有真正成功融合后，
           # 才会在后面：
            #
            #   popleft()
            #
           # 删除。
            # ================================================================
            state = self.pending_states[0]

            # ================================================================
            # 局部功能块：提取当前 state 的同步基准 timestamp
            #
            # 来源：
            #
            #   state["_capture_time_ns"]
            #
           # 这个字段最初随 x86 state 一起发过来，
           # add_state() 已经检查过其时间顺序。
            #
           #
            # 后续用于：
            #
            #   can_match()
            #   _select_frame()
            #   时间误差统计
            # ================================================================
            state_timestamp_ns = int(state["_capture_time_ns"])

            # ================================================================
            # 局部功能块：所有 camera 至少要有一帧
            #
            # cameras_ready()：
            #
           # 检查：
            #
            #   head
            #   left_wrist
            #   right_wrist
            #
           # 等所有 camera queue 非空。
            #
           #
            # 如果有一路还完全没有图像：
            #
            #   break
            #
           #
            # 为什么不是 continue：
            #
           # 因为当前最早的 state 都无法处理，
           # 后面的 state 更不应该越过它先处理。
            #
           #
            # 当前 state：
            #
            #   保留在 pending_states
            #
           # 等下一次更多 camera frame 到来后，
           # 再调用 merge_ready()。
            # ================================================================
            if not self.cameras_ready():
                break

            # ================================================================
            # 局部功能块：正常模式下等待所有 camera 时间线走过 state
            #
            # force=False：
            #
           # 正常录制模式。
            #
           # 此时：
            #
            #   can_match(state_timestamp_ns)
            #
           # 必须为 True。
            #
           #
            # 也就是每一路 camera 的最新 frame：
            #
            #   latest.timestamp >= state timestamp
            #
           #
            # 如果还没满足：
            #
            #   break
            #
           # 等未来 camera frame 到来。
            #
           #
            # force=True：
            #
           # 这个检查被跳过。
            #
           # 主要用于 STOP_SAVE 的尾部收尾阶段：
            #
            #   已经等了一段时间，
           #   不再无限等待更合适的 future camera frame，
           #   而是用当前已有数据尽量完成最后的 state。
            # ================================================================
            if not force and not self.can_match(state_timestamp_ns):
                break

            # ================================================================
            # 局部功能块：创建当前 state 的多相机选择结果
            #
            # selected_frames：
            #
           # key：
            #   camera_name
            #
           # value：
            #   当前 state 对应选中的 CameraFrame
            #
           #
            # 初始：
            #
            #   {}
            #
           #
            # 循环完成后类似：
            #
            #   {
            #       "head": CameraFrame(...),
            #       "left_wrist": CameraFrame(...),
            #       "right_wrist": CameraFrame(...),
            #   }
            #
           #
            # 后续用于：
            #
            #   1. 构造 entry["image"]
            #   2. 清理旧 frame queue
            # ================================================================
            selected_frames: Dict[str, CameraFrame] = {}

            # ================================================================
            # 对每一路 camera 独立选择与当前 state 对应的 frame。
            # ================================================================
            for camera_name in self.camera_names:

                # ============================================================
                # 局部功能块：为当前 camera 选择对应帧
                #
                # 输入：
                #
                #   self.frames[camera_name]
                #       ← add_camera_frame()
                #
               #   state_timestamp_ns
                #       ← 当前 pending state
                #
               #
                # _select_frame() 正常原则：
                #
                #   找 timestamp <= state timestamp
               #   中最新的一帧。
                #
               #
                # 返回：
                #
                #   selected
                #       选中的 CameraFrame
                #
               #   used_fallback
                #       是否因为不存在过去帧而使用 queue[0]
                #
               #
                # 数据去向：
                #
                #   selected
                #       ↓
                #   selected_frames
                #
               #   used_fallback
                #       ↓
                #   fallback 统计
                # ============================================================
                selected, used_fallback = self._select_frame(
                    self.frames[camera_name],
                    state_timestamp_ns,
                )

                # ============================================================
                # 把该 camera 的选择结果保存下来。
                #
                # 后续构造：
                #
                #   entry["image"]
                #
               # 时需要一次性使用三路 selected frame。
                # ============================================================
                selected_frames[camera_name] = selected

                # ============================================================
                # 局部功能块：统计 fallback
                #
                # used_fallback：
                #   ← _select_frame()
                #
               # 如果 True：
                #
                #   当前 camera 没有 timestamp <= state 的帧，
               #   使用了 queue 中最早的一帧。
                #
               #
                # 计数去向：
                #
                #   alignment_summary()
                # ============================================================
                if used_fallback:
                    self.fallback_matches[camera_name] += 1

                # ============================================================
                # 局部功能块：计算 state 与选中 camera frame 的时间偏差
                #
                # state_timestamp_ns：
                #   robot state 采集时间。
                #
               # selected.timestamp_ns：
                #   camera frame 时间。
                #
               #
                # delta_ns：
                #
                #   state_timestamp - camera_timestamp
                #
               #
                # 如果正常选择过去帧：
                #
                #   delta_ns >= 0
                #
               # 如果使用了 state 之后的 fallback frame：
                #
                #   delta_ns < 0
                #
               #
                # abs_delta_ns：
                #
                #   只关心两者相差多少时间，
               #   不关心 frame 在 state 前还是后。
                #
               #
                # 后续用于三项统计：
                #
                #   alignment_count
                #   alignment_abs_sum_ns
                #   alignment_abs_max_ns
                # ============================================================
                delta_ns = state_timestamp_ns - selected.timestamp_ns
                abs_delta_ns = abs(delta_ns)

                # ============================================================
                # 当前 camera 又完成一次 state-frame 对齐。
                #
                # 去向：
                #   alignment_summary()["count"]
                # ============================================================
                self.alignment_count[camera_name] += 1

                # ============================================================
                # 累积绝对时间误差。
                #
                # 最终：
                #
                #   sum / count
                #
               # 得到 mean_abs_delta_ms。
                # ============================================================
                self.alignment_abs_sum_ns[camera_name] += abs_delta_ns

                # ============================================================
                # 更新当前 camera 目前见过的最大同步误差。
                #
                # 最终：
                #
                #   alignment_summary()
                #       ↓
                #   max_abs_delta_ms
                # ============================================================
                self.alignment_abs_max_ns[camera_name] = max(
                    self.alignment_abs_max_ns[camera_name],
                    abs_delta_ns,
                )

            # ================================================================
            # 局部功能块：从原始 state 构造最终 episode sample 的基础部分
            #
            # state：
            #
            #   ← pending_states
            #
           # 原始结构可能包含：
            #
            #   _episode_id
            #   _frame_index
            #   _capture_time_ns
            #
           # 以及真正机器人训练所需：
            #
            #   state
            #   action
            #   gripper
            #   ...
            #
           #
            # 当前 dict comprehension：
            #
            #   对 state 的所有 key/value 做复制，
           #   但是：
            #
            #       if key not in INTERNAL_STATE_KEYS
            #
           # 会过滤 Recorder 内部元数据。
            #
           #
            # 例如：
            #
            # 原始：
            #
            #   {
            #       "_episode_id": "123",
            #       "_frame_index": 10,
            #       "_capture_time_ns": ...,
            #       "state": ...,
            #       "action": ...,
            #   }
            #
           # 得到：
            #
            #   entry = {
            #       "state": ...,
            #       "action": ...,
            #   }
            #
           #
            # entry 后续还会加入：
            #
            #   entry["image"]
            #
           # 然后进入 episode_data。
            # ================================================================
            entry = {
                key: value
                for key, value in state.items()
                if key not in self.INTERNAL_STATE_KEYS
            }

            # ================================================================
            # 局部功能块：把刚才选好的三路 camera image 加到当前 entry
            #
            # selected_frames：
            #
            #   ← 上一个 camera loop
            #
           # 每个 CameraFrame 包含：
            #
            #   timestamp_ns
            #   image
            #   encoding
            #
           #
            # 当前最终只取：
            #
            #   selected_frames[camera_name].image
            #
           #
            # 输出结构：
            #
            #   entry["image"] = {
            #
            #       "head": {
            #           "color": np.ndarray
            #       },
            #
            #       "left_wrist": {
            #           "color": np.ndarray
            #       },
            #
            #       "right_wrist": {
            #           "color": np.ndarray
            #       },
            #   }
            #
           #
            # image.copy()：
            #
           # 不是直接把 CameraFrame 中的 ndarray 引用放进去，
           # 而是复制一份图像数据。
            #
           # 这样 episode_data 中保存的图像数据
           # 不依赖之后 frame queue 内对象的生命周期。
            #
           #
            # 最终去向：
            #
            #   entry
            #       ↓
            #   self.episode_data
            #       ↓
            #   write_episode_pickle()
            # ================================================================
            entry["image"] = {
                camera_name: {
                    "color": selected_frames[camera_name].image.copy(),
                }
                for camera_name in self.camera_names
            }

            # ================================================================
            # 局部功能块：将已经完成 state + camera 对齐的数据
            #             正式加入 episode_data
            #
            # entry：
            #
           # 已经包含：
            #
            #   robot state/action
            #       +
            #   三路 camera images
            #
           #
            # self.episode_data：
            #   ← __init__() 初始化。
            #
           #
            # append 后：
            #
           # 当前 state 就正式成为最终 episode 的一帧数据。
            #
           #
            # 最终：
            #
            #   episode_worker.py
            #
           # 会将整个：
            #
            #   synchronizer.episode_data
            #
           # 传给 write_episode_pickle()。
            # ================================================================
            self.episode_data.append(entry)

            # ================================================================
            # 局部功能块：从 pending queue 删除已经处理成功的 state
            #
            # 当前处理的是：
            #
            #   pending_states[0]
            #
           # 现在已经：
            #
            #   选择 camera
            #   构造 entry
            #   append 到 episode_data
            #
           # 因此可以：
            #
            #   popleft()
            #
           # 让下一轮 while 处理下一条 state。
            # ================================================================
            self.pending_states.popleft()

            # ================================================================
            # 当前 merge_ready() 调用的成功处理数量 +1。
            #
            # 最终 return merged。
            # ================================================================
            merged += 1

            # ================================================================
            # 局部功能块：清理每一路 camera 中已经过时的旧 frame
            #
            # selected_frames：
            #   ← 当前 state 的选择结果。
            #
           # 对每个 camera：
            #
            #   _prune_before_selected()
            #
           # 删除 selected 之前的 frame。
            #
           #
            # 例如：
            #
            #   [80, 90, 105]
            #
           # 当前 selected：
            #
            #   90
            #
           # 处理后：
            #
            #   [90, 105]
            #
           #
            # selected 本身保留，
           # 因为它可能仍然适合匹配下一条 robot state。
            #
           #
            # 这个步骤的主要作用：
            #
            #   1. 控制相机缓存大小
            #   2. 去掉未来不再可能被选择的旧数据
            #   3. 保留可能被下一 state 重用的 selected frame
            # ================================================================
            for camera_name, selected in selected_frames.items():
                self._prune_before_selected(self.frames[camera_name], selected)

        # ====================================================================
        # 返回：
        #
        #   本次 merge_ready() 一共成功融合的 state 数量。
        #
       #
        # 注意：
        #
        # episode_worker 当前主要依赖 merge_ready() 对
        # self.episode_data / pending_states 的副作用，
       # 并没有使用这个返回值进行核心控制。
        #
       # 但返回 merged 可以方便：
       #
        #   调试
        #   测试
        #   统计一次调用做了多少实际工作
        # ====================================================================
        return merged

    def alignment_summary(self) -> dict:
        # ====================================================================
        # alignment_summary()
        #
       # 作用：
        #
        #   把整个 episode 中累计的 camera-state 对齐统计
       #   整理成一个容易输出/记录的 dict。
        #
       #
        # 调用来源：
        #
        #   episode_worker.py
        #
       # 保存成功后 worker 会发送：
        #
        #   {
        #       "type": "saved",
        #       ...
        #       "alignment":
        #           synchronizer.alignment_summary(),
        #   }
        #
       #
        # 然后：
        #
        #   episode_worker
        #       ↓ Pipe
        #   RecorderServer._poll_workers()
        #       ↓
       #   handle.last_message / recent
        #
       # 所以这些数据主要用于：
        #
        #   观察一次 episode 的相机同步质量，
       #   而不是作为训练数据本身保存进 episode_data。
        # ====================================================================

        # ====================================================================
        # 局部功能块：初始化最终统计结果
        #
        # summary 最终结构：
        #
        #   {
        #       "head": {...},
        #       "left_wrist": {...},
        #       "right_wrist": {...},
        #   }
        # ====================================================================
        summary = {}

        # ====================================================================
        # 每一路 camera 单独整理统计。
        # ====================================================================
        for name in self.camera_names:

            # ================================================================
            # count 来源：
            #
            #   merge_ready()
            #
           # 每成功把一个 state 和这个 camera 对齐一次：
            #
            #   alignment_count[name] += 1
            #
           #
            # count 后续用于：
            #
            #   mean absolute delta
            # ================================================================
            count = self.alignment_count[name]

            # ================================================================
            # 局部功能块：计算平均绝对时间偏差
            #
            # alignment_abs_sum_ns：
            #
           #   merge_ready() 中累计所有：
            #
            #       abs(
            #           state_timestamp
            #           -
            #           frame_timestamp
            #       )
            #
           #
            # / count：
            #
           #   得到平均 ns。
            #
           # / 1_000_000：
            #
            #   ns → ms
            #
           # 因为：
            #
            #   1 ms = 1,000,000 ns
            #
           #
            # 如果 count == 0：
            #
            #   返回 0.0
            #
           # 避免除零。
            # ================================================================
            average_ms = (
                self.alignment_abs_sum_ns[name] / count / 1_000_000.0
                if count
                else 0.0
            )

            # ================================================================
            # 局部功能块：整理当前 camera 的最终统计
            #
           #
            # count
            #
            #   来源：
            #       alignment_count
            #
            #   表示：
            #       一共完成多少次 state-camera 匹配。
            #
           #
            # mean_abs_delta_ms
            #
            #   来源：
            #       alignment_abs_sum_ns / count
            #
            #   表示：
            #       平均时间同步误差。
            #
           #
            # max_abs_delta_ms
            #
            #   来源：
            #       alignment_abs_max_ns
            #
            #   表示：
            #       本 episode 最差的一次同步误差。
            #
           #
            # fallback_matches
            #
            #   来源：
            #       _select_frame()
            #
            #   表示：
            #       有多少次找不到 timestamp <= state timestamp 的 frame，
           #       被迫使用 queue 中最早的 frame。
            #
           #
            # dropped_out_of_order_images
            #
            #   来源：
            #       add_camera_frame()
            #
            #   表示：
            #       有多少相机帧因为 timestamp 比前一帧更早
           #       而被直接丢弃。
            #
           #
            # 当前 camera 的结果：
            #
           #   ↓
            #
            # summary[name]
            #
           # 所有 camera 循环完成后，
           # summary 返回 episode_worker。
            # ================================================================
            summary[name] = {
                "count": count,
                "mean_abs_delta_ms": average_ms,
                "max_abs_delta_ms": self.alignment_abs_max_ns[name]
                / 1_000_000.0,
                "fallback_matches": self.fallback_matches[name],
                "dropped_out_of_order_images": self.dropped_out_of_order_images[
                    name
                ],
            }

        # ====================================================================
        # summary 去向：
        #
        #   alignment_summary()
        #       ↓
        #   episode_worker.py
        #
       # worker 保存成功后：
        #
        #   control_connection.send({
        #       "type": "saved",
        #       ...
        #       "alignment": summary,
        #   })
        #
       #       ↓
        #   multiprocessing.Pipe
        #       ↓
       #   RecorderServer._poll_workers()
        #       ↓
       #   recent / STATUS details / log
        #
       #
        # 所以这个 summary 主要用于：
        #
       #   对录制数据的时间同步质量进行诊断。
        # ====================================================================
        return summary