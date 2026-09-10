from __future__ import annotations

# ============================================================================
# 当前文件：tiangong_recorder/video_progress.py
#
# 【这个文件在整个项目中的职责】
#
# 这个文件负责 LeRobot 转换过程中“阶段 7/8”的视频编码部分，主要完成：
#
#   1. 接收已经保存在内存中的 RGB NumPy 图像帧；
#   2. 把这些图像逐帧写入 GStreamer 子进程；
#   3. 使用 Jetson 上的 AV1 硬件编码组件进行编码；
#   4. 先生成一个临时 IVF 文件；
#   5. 再调用 FFmpeg，不重新编码，只把 IVF 中的 AV1 视频流
#      remux（重新封装）成最终 MP4；
#   6. 在编码过程中显示终端进度；
#   7. 编码失败或用户 Ctrl+C 时清理临时文件和子进程。
#
#
# 【这个文件在整个 LeRobot 转换链路中的位置】
#
# recorder 生成的 PKL
#       │
#       ▼
# lerobot_converter.py
#       │
#       ▼
# LeRobotV21Writer
#       │
#       ▼
# DirectLeRobotDataset
#       │
#       │ save_episode()
#       ▼
# prepared[image_key]
#       │
#       │ list[np.ndarray]
#       ▼
# _encode_direct_video()
#       │
#       │ frames / video_path / fps / width / height / bitrate
#       ▼
# encode_rgb_frames_with_jetson_av1()      ← 当前文件核心函数
#       │
#       ├── GStreamer + Jetson AV1 encoder
#       │           │
#       │           ▼
#       │       临时 .ivf
#       │
#       └── FFmpeg remux
#                   │
#                   ▼
#               最终 .mp4
#
#
# 上游 direct_lerobot_dataset.py 中：
#
#   for image_key in self.meta.video_keys:
#
#       video_path = ...
#
#       self._encode_direct_video(
#           prepared[image_key],
#           video_path,
#           image_key,
#       )
#
# _encode_direct_video() 再调用：
#
#       encode_rgb_frames_with_jetson_av1(...)
#
#
# 因此本文件并不负责：
#
#   × 从 PKL 提取图片
#   × 构造 LeRobot frame
#   × 写 Parquet
#   × 计算 state/action
#
# 它只负责：
#
#   NumPy RGB frames
#           ↓
#       AV1 编码
#           ↓
#       MP4 文件
# ============================================================================


# ============================================================================
# subprocess
#
# 来源：
#   Python 标准库。
#
# 作用：
#   从当前 Python 进程启动外部程序。
#
# 当前文件会启动两个外部程序：
#
#   ① gst-launch-1.0
#
#       用 GStreamer 建立视频编码 pipeline，
#       并调用 Jetson AV1 硬件编码组件。
#
#
#   ② ffmpeg
#
#       不再次压缩视频，
#       而是把前一步得到的 IVF 文件重新封装成 MP4。
#
#
# subprocess.Popen：
#
#   表示已经启动的一个外部子进程。
#
# 后面会得到：
#
#   process
#       → GStreamer 子进程
#
#   remux_process
#       → FFmpeg 子进程
#
# 两者在异常情况下都会传给：
#
#   _stop_subprocess()
#
# 进行 terminate / kill 清理。
# ============================================================================
import subprocess


# ============================================================================
# sys
#
# 来源：
#   Python 标准库。
#
# 当前文件主要使用：
#
#       sys.stdout
#
# 作为 TerminalProgressBar 默认输出位置。
#
# 即调用：
#
#       TerminalProgressBar(label)
#
# 没有显式提供 stream 时：
#
#       self.stream = sys.stdout
#
# 进度条就打印到当前程序标准输出。
# ============================================================================
import sys


# ============================================================================
# pathlib.Path
#
# 来源：
#   Python 标准库 pathlib。
#
# 作用：
#   用面向对象方式处理文件路径。
#
# 当前文件用于管理三类文件：
#
#   video_path
#       最终目标 MP4。
#
#   ivf_path
#       Jetson AV1 编码阶段产生的临时 IVF。
#
#   remux_path
#       FFmpeg remux 阶段产生的临时 MP4。
#
#
# 最终文件流：
#
#   NumPy frames
#       │
#       ▼
#   ivf_path
#       │
#       ▼
#   remux_path
#       │
#       │ Path.replace()
#       ▼
#   video_path
#
# 然后：
#
#   ivf_path.unlink()
#
# 删除临时 IVF。
# ============================================================================
from pathlib import Path


# ============================================================================
# Sequence / TextIO
#
# 来源：
#   Python 标准库 typing。
#
#
# Sequence[np.ndarray]
#
#   表示 frames 可以是支持：
#
#       len()
#       顺序遍历
#
#   的 NumPy 图像序列。
#
# 在当前项目实际调用中，
# frames 来自 DirectLeRobotDataset 中：
#
#       prepared[image_key]
#
# 它实际是：
#
#       list[np.ndarray]
#
#
# TextIO
#
#   表示文本输出流类型。
#
#   TerminalProgressBar.stream 可以接受：
#
#       sys.stdout
#       sys.stderr
#       文件对象
#       其他类似文本流
# ============================================================================
from typing import Sequence, TextIO


# ============================================================================
# NumPy
#
# 来源：
#   第三方数值计算库 numpy。
#
# 当前文件中 NumPy 图像采用：
#
#       ndarray
#
# 预期格式：
#
#       shape = (height, width, 3)
#       dtype = np.uint8
#
# 即：
#
#       H × W × RGB
#
#
# 图像来源：
#
#   DirectLeRobotDataset
#       │
#       ▼
#   prepared[image_key]
#       │
#       ▼
#   frames
#       │
#       ▼
#   _rgb_frame()
#
#
# 最终通过：
#
#       ndarray.tobytes()
#
# 转成原始字节流，
# 写入 GStreamer stdin。
# ============================================================================
import numpy as np


# ============================================================================
# TerminalProgressBar
#
# 定义来源：
#   当前 video_progress.py。
#
#
# 【职责】
#
# 用来显示 AV1 编码过程中：
#
#       已编码帧数 / 总帧数
#
# 以及百分比。
#
#
# 它支持两种输出环境：
#
#   ① interactive=True
#
#       通常表示当前输出直接连接真正的终端 TTY。
#
#       这种情况下使用：
#
#           "\r"
#
#       在同一行刷新进度条。
#
#
#   ② interactive=False
#
#       比如输出被重定向到日志文件。
#
#       这种情况下不能不断在一行覆盖，
#       所以代码只在大约每 10% 时打印一行日志。
#
#
# 【当前项目中的创建位置】
#
# encode_rgb_frames_with_jetson_av1()
#
#       progress = TerminalProgressBar(label)
#
#
# label 来源：
#
# DirectLeRobotDataset._encode_direct_video()
#
#       label = image_key.rsplit(".", 1)[-1]
#
# 因此通常对应摄像头输出名称，例如：
#
#       front
#       left_wrist
#       right_wrist
#
#
# progress 后续被：
#
#   progress.update(...)
#       → 更新编码进度
#
#   progress.break_line()
#       → 异常时结束当前正在覆盖输出的终端行
# ============================================================================
class TerminalProgressBar:
    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        width: int = 30,
    ) -> None:
        # ====================================================================
        # TerminalProgressBar.__init__()
        #
        # 输入变量来源：
        #
        # label
        #   ← encode_rgb_frames_with_jetson_av1(label=...)
        #   ← DirectLeRobotDataset._encode_direct_video()
        #   ← image_key 最后一段
        #
        #
        # stream
        #
        #   调用者可显式传入输出流。
        #
        #   当前正常调用：
        #
        #       TerminalProgressBar(label)
        #
        #   没有提供 stream，
        #   所以使用 sys.stdout。
        #
        #
        # width
        #
        #   进度条字符宽度。
        #
        #   默认：
        #
        #       30
        #
        #
        # 当前功能块处理后的结果：
        #
        #   保存为 self.label / self.stream / self.width，
        #   后续全部由 update() 和 break_line() 使用。
        # ====================================================================
        self.label = label
        self.stream = stream or sys.stdout
        self.width = width

        # ====================================================================
        # 局部功能块：判断输出是否是交互式终端
        #
        # 输入：
        #
        #   self.stream
        #       ← 上一个功能块确定的输出流。
        #
        #
        # getattr(self.stream, "isatty", lambda: False)
        #
        #   尝试获取 stream.isatty()。
        #
        #   如果这个 stream 对象根本没有 isatty 方法，
        #   就使用：
        #
        #       lambda: False
        #
        #   作为安全默认值。
        #
        #
        # isatty()：
        #
        #   用来判断这个文本流是否连接到交互式终端。
        #
        #
        # 输出：
        #
        #   self.interactive
        #
        # 后续传给：
        #
        #   update()
        #       → 决定使用“单行动态刷新”
        #         还是“每 10% 打一行日志”
        #
        #   break_line()
        #       → 只有 interactive 模式才需要主动换行。
        # ====================================================================
        self.interactive = bool(
            getattr(self.stream, "isatty", lambda: False)()
        )

        # ====================================================================
        # 局部功能块：初始化进度输出状态
        #
        # _last_logged_bucket：
        #
        #   非交互式日志模式使用。
        #
        #   bucket 按：
        #
        #       0, 1, 2, ... 10
        #
        #   大致对应：
        #
        #       0%, 10%, 20%, ... 100%
        #
        # 初始 -1：
        #   保证第一次 update() 可以打印。
        #
        #
        # _last_rendered_percent：
        #
        #   interactive 模式使用。
        #
        #   保存上一次已经显示的整数百分比。
        #
        #   用于避免同一个百分比被重复刷新很多次。
        #
        #
        # 两个变量后续都只传给：
        #
        #       update()
        #
        # 作为“是否需要再次输出”的状态。
        # ====================================================================
        self._last_logged_bucket = -1
        self._last_rendered_percent = -1

    def update(
        self,
        current: int,
        total: int,
        *,
        force: bool = False,
    ) -> None:
        # ====================================================================
        # update()
        #
        # 【调用来源】
        #
        # encode_rgb_frames_with_jetson_av1() 中有两种调用：
        #
        # ① 编码开始前：
        #
        #       progress.update(
        #           0,
        #           len(frames),
        #           force=True,
        #       )
        #
        #
        # ② 每成功向 GStreamer 写入一帧后：
        #
        #       progress.update(
        #           index,
        #           len(frames),
        #       )
        #
        #
        # 因此：
        #
        # current
        #   ← 当前已经送入 GStreamer 的帧数 index。
        #
        # total
        #   ← len(frames)，当前摄像头总图像帧数。
        #
        # force
        #   ← 是否无视去重规则强制输出。
        #
        #
        # 输出：
        #
        #   这个函数不返回数据。
        #
        #   它最终把可视化进度：
        #
        #       message
        #
        #   打印到 self.stream。
        # ====================================================================

        # ====================================================================
        # 局部功能块：规范 total
        #
        # 输入：
        #
        #   total
        #       ← 调用者传入，正常为 len(frames)。
        #
        #
        # int(total)：
        #   转换为整数。
        #
        # max(1, ...)：
        #
        #   保证 total 至少为 1，
        #   避免下面：
        #
        #       current / total
        #
        #   发生除零。
        #
        #
        # 输出：
        #
        #   新的局部变量 total。
        #
        # 下一步传给：
        #
        #   current 范围限制
        #   percent 计算
        #   message
        # ====================================================================
        total = max(1, int(total))

        # ====================================================================
        # 局部功能块：把 current 限制到合法范围
        #
        # 输入：
        #
        #   current
        #       ← 当前已处理帧数量。
        #
        #   total
        #       ← 上一步规范后的总帧数。
        #
        #
        # 处理：
        #
        #   int(current)
        #       ↓
        #   max(0, ...)
        #       ↓
        #   min(..., total)
        #
        #
        # 最终保证：
        #
        #       0 <= current <= total
        #
        #
        # 输出：
        #
        #   current
        #
        # 下一步传给：
        #
        #   percent = current / total
        #   message 中的 current/total
        # ====================================================================
        current = min(max(0, int(current)), total)

        # ====================================================================
        # 局部功能块：计算完成比例
        #
        # 输入：
        #
        #   current
        #   total
        #
        #
        # 输出：
        #
        #   percent
        #
        # 范围：
        #
        #       0.0 ~ 1.0
        #
        #
        # 后续传给：
        #
        #   completed
        #       → 进度条填充长度
        #
        #   message
        #       → 百分比显示
        #
        #   rendered_percent
        #       → interactive 输出去重
        #
        #   bucket
        #       → 非 interactive 每 10% 日志输出
        # ====================================================================
        percent = current / total

        # ====================================================================
        # 局部功能块：计算进度条已完成字符数
        #
        # 输入：
        #
        #   percent
        #       ← 上一步完成比例。
        #
        #   self.width
        #       ← __init__() 中设置的进度条宽度。
        #
        #
        # percent * width：
        #
        #   把 0~1 的完成比例转换成：
        #
        #       0 ~ width
        #
        #
        # int()：
        #   向下取整成字符数量。
        #
        # min(self.width, ...)：
        #   确保不会超过总宽度。
        #
        #
        # 输出：
        #
        #   completed
        #
        # 下一步用于构造 bar。
        # ====================================================================
        completed = min(
            self.width,
            int(percent * self.width),
        )

        # ====================================================================
        # 局部功能块：构造进度条字符串
        #
        # 输入：
        #
        #   completed
        #       ← 已完成的字符数量。
        #
        #   self.width - completed
        #       ← 尚未完成字符数量。
        #
        #
        # 已完成部分：
        #
        #       █
        #
        # 未完成部分：
        #
        #       ░
        #
        #
        # 例如 width=10、完成 30%：
        #
        #       ███░░░░░░░
        #
        #
        # 输出：
        #
        #   bar
        #
        # 下一步传给 message。
        # ====================================================================
        bar = (
            "█" * completed
            + "░" * (self.width - completed)
        )

        # ====================================================================
        # 局部功能块：构造最终进度显示字符串
        #
        # 输入来源：
        #
        #   self.label
        #       ← 当前摄像头名称。
        #
        #   bar
        #       ← 上一步构造的字符进度条。
        #
        #   percent
        #       ← 当前比例。
        #
        #   current / total
        #       ← 当前帧数 / 总帧数。
        #
        #
        # 输出：
        #
        #   message
        #
        # 例如：
        #
        #   [LeRobot转换][阶段 7/8][MP4编码][front]
        #   [████████░░░...] 25.0% 250/1000
        #
        #
        # message 下一步会根据：
        #
        #   self.interactive
        #
        # 分成两种输出策略。
        # ====================================================================
        message = (
            f"[LeRobot转换][阶段 7/8][MP4编码][{self.label}] "
            f"[{bar}] {percent:6.1%} {current}/{total}"
        )

        # ====================================================================
        # 局部功能块：交互式终端输出
        #
        # 判断变量：
        #
        #   self.interactive
        #       ← __init__() 通过 stream.isatty() 得到。
        #
        #
        # 如果当前输出是终端，
        # 希望在“同一行”动态刷新进度，而不是不断换行。
        # ====================================================================
        if self.interactive:

            # ================================================================
            # 局部功能块：把浮点百分比转成整数百分比
            #
            # 输入：
            #
            #   percent
            #
            # 例如：
            #
            #   0.2387
            #       ↓
            #   23
            #
            #
            # 输出：
            #
            #   rendered_percent
            #
            # 用于和：
            #
            #   self._last_rendered_percent
            #
            # 比较，防止同一个 1% 范围内重复刷新很多次。
            # ================================================================
            rendered_percent = int(percent * 100)

            # ================================================================
            # 局部功能块：判断这次是否需要重新渲染
            #
            # 输入：
            #
            # force
            #   ← 调用者是否要求强制输出。
            #
            # current != total
            #
            #   如果已经是最后一帧，
            #   即使百分比没有变化也应该显示最终完成状态。
            #
            # rendered_percent
            #   ← 当前整数百分比。
            #
            # self._last_rendered_percent
            #   ← 上一次已经显示的百分比。
            #
            #
            # 条件满足时：
            #
            #   return
            #
            # 即：
            #
            #   当前百分比并没有比上一次更大，
            #   又不是强制显示，也不是最终完成，
            #   那就不重复打印。
            #
            #
            # 这样例如 1000 帧视频不会因为每一帧都调用 update()
            # 而刷新 1000 次终端，
            # 而是大约每变化 1% 刷一次。
            # ================================================================
            if (
                not force
                and current != total
                and rendered_percent
                <= self._last_rendered_percent
            ):
                return

            # ================================================================
            # 局部功能块：确定 print 是否换行
            #
            # 输入：
            #
            #   current
            #   total
            #
            #
            # 如果：
            #
            #   current == total
            #
            # 表示视频已经全部送入编码器，
            # 最后一次输出使用：
            #
            #   "\n"
            #
            # 正式结束这一行。
            #
            #
            # 否则：
            #
            #   ending = ""
            #
            # 不换行，让下一次 "\r" 回到当前行开头继续覆盖。
            #
            #
            # 输出：
            #
            #   ending
            #
            # 下一步传给 print(end=...)。
            # ================================================================
            ending = "\n" if current == total else ""

            # ================================================================
            # 局部功能块：打印交互式进度
            #
            # "\r"：
            #
            #   carriage return，
            #   把光标移动到当前行开头。
            #
            #
            # message：
            #   ← 前面构造好的完整进度文本。
            #
            # ending：
            #   ← 当前是否已经完成。
            #
            # self.stream：
            #   ← 默认 sys.stdout。
            #
            # flush=True：
            #
            #   立即把输出刷新到终端，
            #   不等待 Python 输出缓冲区。
            #
            #
            # 输出去向：
            #
            #   用户当前终端。
            # ================================================================
            print(
                f"\r{message}",
                end=ending,
                file=self.stream,
                flush=True,
            )

            # ================================================================
            # 局部功能块：记录本次已显示百分比
            #
            # 输入：
            #
            #   rendered_percent
            #
            #
            # 输出：
            #
            #   self._last_rendered_percent
            #
            # 下一次 update() 会拿它来判断：
            #
            #   “是否已经显示过这个百分比？”
            # ================================================================
            self._last_rendered_percent = rendered_percent

            # ================================================================
            # interactive 分支已经完成所有输出，
            # 不需要继续执行下面的非交互日志逻辑。
            # ================================================================
            return

        # ====================================================================
        # 局部功能块：非交互式输出计算日志 bucket
        #
        # 到这里表示：
        #
        #   self.interactive == False
        #
        # 常见情况例如：
        #
        #   stdout 被重定向到文件
        #   后台服务日志
        #   CI 日志
        #
        #
        # 输入：
        #
        #   percent
        #
        #
        # int(percent * 10)：
        #
        #   把进度划成约 10 个区间：
        #
        #   0  →   0%~9%
        #   1  →  10%~19%
        #   ...
        #   10 → 100%
        #
        #
        # 输出：
        #
        #   bucket
        #
        # 下一步和：
        #
        #   self._last_logged_bucket
        #
        # 比较。
        # ====================================================================
        bucket = int(percent * 10)

        # ====================================================================
        # 局部功能块：每跨过一个约 10% 区间才打印一行
        #
        # 输入：
        #
        # force
        #   → 强制打印。
        #
        # bucket
        #   → 当前所在的 10% 区间。
        #
        # self._last_logged_bucket
        #   → 上一次已经打印的区间。
        #
        #
        # 如果当前进入了新的 bucket：
        #
        #   print(message)
        #
        #
        # 输出去向：
        #
        #   self.stream，
        #   通常是日志文件或非 TTY stdout。
        #
        #
        # 然后：
        #
        #   self._last_logged_bucket = bucket
        #
        # 供下一次 update() 去重。
        # ====================================================================
        if force or bucket > self._last_logged_bucket:
            print(
                message,
                file=self.stream,
                flush=True,
            )
            self._last_logged_bucket = bucket

    def break_line(self) -> None:
        # ====================================================================
        # break_line()
        #
        # 【调用来源】
        #
        # encode_rgb_frames_with_jetson_av1() 的异常处理：
        #
        #   except KeyboardInterrupt:
        #       progress.break_line()
        #
        #   except Exception:
        #       progress.break_line()
        #
        #
        # 为什么需要它：
        #
        #   interactive 进度条通过：
        #
        #       "\r"
        #       end=""
        #
        #   一直停留在同一行。
        #
        # 如果编码中途突然异常，
        # 当前光标仍然可能停留在进度条这一行。
        #
        # 此函数补一个换行，
        # 避免后面的错误信息直接接在进度条后面。
        #
        #
        # 非 interactive：
        #
        #   本来每条日志都有换行，
        #   所以什么都不用做。
        #
        #
        # 输出：
        #
        #   只影响终端排版，不产生业务数据。
        # ====================================================================
        if self.interactive:
            print(
                file=self.stream,
                flush=True,
            )


# ============================================================================
# HardwareVideoEncodingError
#
# 定义来源：
#   当前 video_progress.py。
#
#
# 父类：
#
#   RuntimeError
#
#   Python 内置运行时异常类型。
#
#
# 【为什么单独定义这个异常】
#
# 视频编码过程可能出现很多底层异常：
#
#   BrokenPipeError
#   OSError
#   GStreamer 非零退出码
#   FFmpeg 非零退出码
#   等等
#
# 当前文件把这些与“硬件视频编码 / remux”相关的失败统一包装成：
#
#       HardwareVideoEncodingError
#
#
# 上层调用关系：
#
#   encode_rgb_frames_with_jetson_av1()
#           │
#           │ raise HardwareVideoEncodingError
#           ▼
#   DirectLeRobotDataset._encode_direct_video()
#           │
#           ▼
#   DirectLeRobotDataset.save_episode()
#           │
#           ▼
#   更上层 converter 的错误处理
#
#
# 这样上层可以知道：
#
#   “这是视频编码阶段失败”
#
# 而不是普通的数据校验 ValueError。
# ============================================================================
class HardwareVideoEncodingError(RuntimeError):
    pass


# ============================================================================
# _rgb_frame()
#
# 定义来源：
#   当前 video_progress.py。
#
#
# 【职责】
#
# 在 NumPy 图像真正写入 GStreamer stdin 之前，
# 对每一帧做最后一次格式检查并保证内存连续。
#
#
# 调用位置：
#
# encode_rgb_frames_with_jetson_av1()
#
#   for frame in frames:
#
#       process.stdin.write(
#           _rgb_frame(
#               frame,
#               width,
#               height,
#           ).tobytes()
#       )
#
#
# 数据链：
#
# DirectLeRobotDataset
#       │
#       ▼
# frames[index]
#       │
#       ▼
# _rgb_frame()
#       │
#       │ 校验 shape / dtype
#       │ 转 contiguous ndarray
#       ▼
# .tobytes()
#       │
#       ▼
# GStreamer stdin
# ============================================================================
def _rgb_frame(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    # ========================================================================
    # 局部功能块：把输入转换成 NumPy ndarray 视图/对象
    #
    # frame 来源：
    #
    #   encode_rgb_frames_with_jetson_av1()
    #       ↓
    #   frames 中当前遍历到的一帧。
    #
    #
    # frames 的上游：
    #
    #   DirectLeRobotDataset.save_episode()
    #       ↓
    #   prepared[image_key]
    #       ↓
    #   _encode_direct_video()
    #       ↓
    #   encode_rgb_frames_with_jetson_av1(frames, ...)
    #
    #
    # np.asarray(frame)：
    #
    #   保证后续按 ndarray 接口访问：
    #
    #       shape
    #       dtype
    #
    # 如果 frame 已经是 ndarray，
    # 通常不需要复制数据。
    #
    #
    # 输出：
    #
    #   value
    #
    # 下一步：
    #
    #   校验 shape/dtype。
    # ========================================================================
    value = np.asarray(frame)

    # ========================================================================
    # 局部功能块：构造预期图像 shape
    #
    # width / height 来源：
    #
    # DirectLeRobotDataset._encode_direct_video()
    #
    #   width =
    #       self.features[image_key]["shape"][1]
    #
    #   height =
    #       self.features[image_key]["shape"][0]
    #
    #
    # 项目使用 HWC RGB：
    #
    #       height × width × 3
    #
    #
    # 最后一个 3 表示 RGB 三个通道。
    #
    #
    # 输出：
    #
    #   expected_shape
    #
    # 下一步和：
    #
    #   value.shape
    #
    # 比较。
    # ========================================================================
    expected_shape = (
        height,
        width,
        3,
    )

    # ========================================================================
    # 局部功能块：验证视频帧格式
    #
    # 输入：
    #
    #   value.shape
    #       ← 当前 NumPy 图像。
    #
    #   value.dtype
    #       ← 当前 NumPy 图像数据类型。
    #
    #   expected_shape
    #       ← width / height 构造。
    #
    #
    # 要求：
    #
    #   shape == (height, width, 3)
    #
    # 且：
    #
    #   dtype == np.uint8
    #
    #
    # 原因：
    #
    #   后面的 GStreamer pipeline 明确声明输入：
    #
    #       rawvideoparse
    #       format=rgb
    #       width=...
    #       height=...
    #
    # Python 这里只发送“裸字节”，
    # 不会为每一帧附带 NumPy shape/dtype 元信息。
    #
    # 因此 Python 写出的内存布局必须和
    # GStreamer 被告知的输入格式保持一致。
    #
    #
    # 不符合时：
    #
    #   raise ValueError
    #
    # 异常下一步会被主编码函数的：
    #
    #   except Exception as exc
    #
    # 捕获，
    #
    # 再包装为：
    #
    #   HardwareVideoEncodingError。
    # ========================================================================
    if (
        value.shape != expected_shape
        or value.dtype != np.uint8
    ):
        raise ValueError(
            f"video frame must be uint8 with shape {expected_shape}; "
            f"got {value.dtype} {value.shape}"
        )

    # ========================================================================
    # 局部功能块：保证 NumPy 内存连续
    #
    # 输入：
    #
    #   value
    #       ← shape/dtype 已经验证合法的 RGB ndarray。
    #
    #
    # np.ascontiguousarray():
    #
    #   保证数组采用连续的 C-order 内存布局。
    #
    # 这一点对于下一步：
    #
    #       .tobytes()
    #
    # 非常重要，
    # 因为 GStreamer 接收到的只是连续 raw RGB bytes。
    #
    #
    # 输出：
    #
    #   contiguous np.ndarray
    #
    # 返回给：
    #
    #   encode_rgb_frames_with_jetson_av1()
    #
    # 随后：
    #
    #   .tobytes()
    #       ↓
    #   process.stdin.write(...)
    #       ↓
    #   GStreamer
    # ========================================================================
    return np.ascontiguousarray(value)


# ============================================================================
# _stop_subprocess()
#
# 定义来源：
#   当前 video_progress.py。
#
#
# 【职责】
#
# 在：
#
#   Ctrl+C
#   Python 异常
#
# 等异常退出路径中，
# 尽量可靠地停止已经启动的外部进程。
#
#
# process 可能来源于：
#
#   ① GStreamer：
#
#       process = subprocess.Popen(command, ...)
#
#
#   ② FFmpeg：
#
#       remux_process = subprocess.Popen(remux_command, ...)
#
#
# 调用：
#
#   _stop_subprocess(process)
#
# 或：
#
#   _stop_subprocess(remux_process)
#
#
# 停止策略：
#
#   已退出？
#      │
#      ├── 是 → 什么都不做
#      │
#      └── 否
#           │
#           ▼
#       terminate()
#           │
#       等待最多 3 秒
#           │
#        ┌──┴──┐
#        │     │
#      成功   超时
#              │
#              ▼
#            kill()
#              │
#              ▼
#        再等待最多 3 秒
# ============================================================================
def _stop_subprocess(
    process: subprocess.Popen,
) -> None:
    # ========================================================================
    # 局部功能块：检查子进程是否已经结束
    #
    # process 来源：
    #
    #   GStreamer Popen
    #
    # 或：
    #
    #   FFmpeg Popen
    #
    #
    # process.poll()：
    #
    #   检查子进程当前状态，不阻塞。
    #
    # 返回：
    #
    #   None
    #       → 子进程仍然运行。
    #
    #   整数退出码
    #       → 子进程已经退出。
    #
    #
    # 如果已经退出：
    #
    #   return
    #
    # 不再重复 terminate。
    # ========================================================================
    if process.poll() is not None:
        return

    # ========================================================================
    # 局部功能块：先尝试正常终止子进程
    #
    # 输入：
    #
    #   process
    #       ← 尚未退出的 Popen。
    #
    #
    # terminate()：
    #
    #   请求终止子进程。
    #
    #
    # 下一步：
    #
    #   process.wait(timeout=3)
    #
    # 给它最多 3 秒完成退出。
    # ========================================================================
    process.terminate()

    try:
        # ====================================================================
        # 局部功能块：等待 terminate 生效
        #
        # process.wait(timeout=3)：
        #
        #   阻塞最多 3 秒，
        #   等子进程退出。
        #
        #
        # 如果正常结束：
        #
        #   _stop_subprocess() 后续自然返回。
        #
        #
        # 如果 3 秒还没退出：
        #
        #   subprocess.TimeoutExpired
        #
        # 进入下面更强制的 kill 路径。
        # ====================================================================
        process.wait(timeout=3)

    except subprocess.TimeoutExpired:
        # ====================================================================
        # 局部功能块：terminate 无法及时退出时强制 kill
        #
        # subprocess.TimeoutExpired：
        #
        #   Python subprocess 模块定义的异常，
        #   表示 wait(timeout=...) 超时。
        #
        #
        # process.kill()：
        #
        #   使用更强制的方式终止子进程。
        #
        #
        # kill 后再次：
        #
        #   process.wait(timeout=3)
        #
        # 等待操作系统完成进程回收。
        #
        #
        # 完成后：
        #
        #   控制流回到调用方的异常清理逻辑，
        #   继续删除 ivf/remux/video 等临时文件。
        # ====================================================================
        process.kill()
        process.wait(timeout=3)


# ============================================================================
# encode_rgb_frames_with_jetson_av1()
#
# 【这是当前文件最核心的函数】
#
#
# 定义来源：
#
#   tiangong_recorder/video_progress.py
#
#
# 调用来源：
#
#   tiangong_recorder/direct_lerobot_dataset.py
#
# 中的：
#
#   DirectLeRobotDataset._encode_direct_video()
#
#
# 再上一层：
#
#   DirectLeRobotDataset.save_episode()
#
# 会遍历：
#
#   self.meta.video_keys
#
# 对每个摄像头分别调用一次。
#
#
# ---------------------------------------------------------------------------
# 【输入变量的完整来源】
# ---------------------------------------------------------------------------
#
# frames
#
#   来源：
#
#       DirectLeRobotDataset.save_episode()
#           │
#           ▼
#       prepared[image_key]
#           │
#           ▼
#       _encode_direct_video(frames, ...)
#
#   实际内容：
#
#       list[np.ndarray]
#
#   每个 ndarray 是已经转换成目标 RGB 顺序的图像。
#
#
# video_path
#
#   来源：
#
#       self.root
#           +
#       self.meta.get_video_file_path(
#           episode_index,
#           image_key,
#       )
#
#   表示当前 camera 最终应该写出的 MP4 文件路径。
#
#
# fps
#
#   来源：
#
#       DirectLeRobotDataset.self.fps
#
#   再向上来自 LeRobotV21Writer 创建数据集时使用的 fps 配置。
#
#
# label
#
#   来源：
#
#       image_key.rsplit(".", 1)[-1]
#
#   通常类似：
#
#       front
#       left_wrist
#       right_wrist
#
#   这里只用于进度和日志展示。
#
#
# width / height
#
#   来源：
#
#       self.features[image_key]["shape"]
#
#   用来：
#
#       1. 校验 NumPy frame shape；
#       2. 告诉 GStreamer raw RGB 输入尺寸。
#
#
# bitrate
#
#   来源：
#
#       self._direct_video_bitrates[label]
#
#   其默认值在 lerobot_writer.py 中按三路 camera 配置。
#
#   最终传给：
#
#       nvv4l2av1enc
#
#   作为编码 bitrate 参数。
#
#
# ---------------------------------------------------------------------------
# 【函数内部总数据流】
# ---------------------------------------------------------------------------
#
# frames: Sequence[np.ndarray]
#       │
#       │ 每帧
#       ▼
# _rgb_frame()
#       │
#       │ shape/dtype/连续内存检查
#       ▼
# ndarray.tobytes()
#       │
#       ▼
# GStreamer stdin
#       │
#       ▼
# fdsrc
#       │
#       ▼
# rawvideoparse RGB
#       │
#       ▼
# videoconvert → I420
#       │
#       ▼
# nvvidconv → NVMM/NV12
#       │
#       ▼
# nvv4l2av1enc
#       │
#       ▼
# ivf_path
#       │
#       │ FFmpeg -c:v copy
#       ▼
# remux_path
#       │
#       │ replace()
#       ▼
# video_path
#
#
# 最终返回：
#
#   None
#
# 实际“输出”是文件系统中的：
#
#   video_path
#
# 即 LeRobot episode 的一个 MP4 视频文件。
# ============================================================================
def encode_rgb_frames_with_jetson_av1(
    frames: Sequence[np.ndarray],
    video_path: Path | str,
    *,
    fps: int,
    label: str,
    width: int,
    height: int,
    bitrate: int,
) -> None:
    """Encode seekable AV1 with Jetson NVENC, then losslessly remux IVF to MP4."""

    # ========================================================================
    # 局部功能块：拒绝空视频
    #
    # 输入：
    #
    #   frames
    #       ← DirectLeRobotDataset 为当前 camera 收集的全部图像。
    #
    #
    # if not frames：
    #
    #   当前 camera 一帧图像都没有。
    #
    #
    # 这种情况下没有可编码视频，
    # 直接：
    #
    #   raise ValueError
    #
    #
    # 异常去向：
    #
    #   返回上层：
    #
    #       _encode_direct_video()
    #           ↓
    #       save_episode()
    #
    # 从而阻止生成一个看似成功但实际为空的 MP4。
    # ========================================================================
    if not frames:
        raise ValueError("cannot encode an empty video")

    # ========================================================================
    # 局部功能块：把目标路径统一转换成 Path
    #
    # video_path 来源：
    #
    #   DirectLeRobotDataset._encode_direct_video()
    #
    # 上游根据：
    #
    #   dataset root
    #   episode_index
    #   image_key
    #
    # 计算出来。
    #
    #
    # 调用者允许传：
    #
    #   Path
    #   或 str
    #
    #
    # Path(video_path)：
    #
    #   统一转换为 Path，
    #   后面才能方便调用：
    #
    #       .parent
    #       .mkdir()
    #       .with_name()
    #       .unlink()
    #       .replace()
    #
    #
    # 输出：
    #
    #   video_path: Path
    # ========================================================================
    video_path = Path(video_path)

    # ========================================================================
    # 局部功能块：确保最终视频父目录存在
    #
    # 输入：
    #
    #   video_path.parent
    #
    #
    # mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )
    #
    # parents=True：
    #   缺少多层父目录时一并创建。
    #
    # exist_ok=True：
    #   目录已经存在也不报错。
    #
    #
    # 输出：
    #
    #   文件系统中的 video parent directory。
    #
    # 后面 GStreamer / FFmpeg 可以在其中写文件。
    # ========================================================================
    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # 局部功能块：删除可能残留的旧目标 MP4
    #
    # 输入：
    #
    #   video_path
    #
    #
    # missing_ok=True：
    #
    #   文件不存在时也不报错。
    #
    #
    # 用途：
    #
    #   确保本次编码不会误把之前残留的 MP4
    #   当作当前任务成功产生的结果。
    #
    #
    # video_path 后面最终会由：
    #
    #   remux_path.replace(video_path)
    #
    # 重新生成。
    # ========================================================================
    video_path.unlink(missing_ok=True)

    # ========================================================================
    # 局部功能块：为两阶段编码创建临时文件路径
    #
    # 输入：
    #
    #   video_path
    #
    #
    # ivf_path：
    #
    #   Jetson/GStreamer 编码阶段的临时输出。
    #
    # 例如最终：
    #
    #   front.mp4
    #
    # 临时可能是：
    #
    #   .front.mp4.encoding.ivf
    #
    #
    # remux_path：
    #
    #   FFmpeg remux 阶段的临时 MP4。
    #
    # 例如：
    #
    #   .front.remux.tmp.mp4
    #
    #
    # 为什么不让 FFmpeg 直接写最终 video_path：
    #
    #   当前代码使用临时文件，
    #   只有 FFmpeg 成功后才：
    #
    #       remux_path.replace(video_path)
    #
    #   从而减少失败时留下半成品目标文件的风险。
    #
    #
    # 输出：
    #
    #   ivf_path
    #       → 传给 GStreamer filesink
    #
    #   remux_path
    #       → 传给 FFmpeg
    # ========================================================================
    ivf_path = video_path.with_name(
        f".{video_path.name}.encoding.ivf"
    )

    remux_path = video_path.with_name(
        f".{video_path.stem}.remux.tmp.mp4"
    )

    # ========================================================================
    # 局部功能块：清除上一次异常可能留下的临时文件
    #
    # 输入：
    #
    #   ivf_path
    #   remux_path
    #
    #
    # 输出：
    #
    #   一个干净的编码工作目录状态。
    #
    # 后续：
    #
    #   GStreamer 创建 ivf_path
    #
    #   FFmpeg 创建 remux_path
    # ========================================================================
    ivf_path.unlink(missing_ok=True)
    remux_path.unlink(missing_ok=True)

    # ========================================================================
    # 局部功能块：构造 GStreamer AV1 编码命令
    #
    # command 最终会传给：
    #
    #   subprocess.Popen(command, ...)
    #
    #
    # 外部程序：
    #
    #   gst-launch-1.0
    #
    # 是 GStreamer 提供的命令行 pipeline 启动工具。
    #
    #
    # ------------------------------------------------------------------------
    # pipeline 数据流
    # ------------------------------------------------------------------------
    #
    # Python
    #
    #   process.stdin.write(
    #       RGB ndarray bytes
    #   )
    #
    #       │
    #       ▼
    #
    # fdsrc fd=0
    #
    #   从子进程 stdin 文件描述符读取原始字节。
    #
    #       │
    #       ▼
    #
    # rawvideoparse
    #
    #   Python 传入的是没有容器、没有帧头的 raw bytes。
    #
    #   因此这里明确告诉 GStreamer：
    #
    #       format=rgb
    #       width=width
    #       height=height
    #       framerate=fps/1
    #
    #       │
    #       ▼
    #
    # videoconvert
    #
    #       │
    #       ▼
    #
    # video/x-raw,format=I420
    #
    #   要求这一阶段得到 I420。
    #
    #       │
    #       ▼
    #
    # nvvidconv
    #
    #   NVIDIA/Jetson 视频转换环节。
    #
    #       │
    #       ▼
    #
    # video/x-raw(memory:NVMM),format=NV12
    #
    #   把下一阶段要求的数据放到 NVIDIA 视频内存路径，
    #   并指定 NV12 格式。
    #
    #       │
    #       ▼
    #
    # nvv4l2av1enc
    #
    #   Jetson AV1 编码器。
    #
    #   bitrate 等参数都在这里传入。
    #
    #       │
    #       ▼
    #
    # filesink
    #
    #       location=ivf_path
    #
    #   将编码输出写到临时 IVF 路径。
    #
    #
    # command 使用的外部变量：
    #
    #   width
    #       ← feature shape
    #
    #   height
    #       ← feature shape
    #
    #   fps
    #       ← dataset fps
    #
    #   bitrate
    #       ← camera 对应编码 bitrate
    #
    #   ivf_path
    #       ← 前面从 video_path 构造
    #
    #
    # 输出：
    #
    #   command: list[str]
    #
    # 下一步传入：
    #
    #   subprocess.Popen(command)
    # ========================================================================
    command = [
        "gst-launch-1.0",
        "-q",
        "fdsrc",
        "fd=0",
        "!",
        "rawvideoparse",
        "format=rgb",
        f"width={width}",
        f"height={height}",
        f"framerate={fps}/1",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=I420",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "nvv4l2av1enc",
        f"bitrate={bitrate}",
        "control-rate=1",
        "preset-level=2",
        "maxperf-enable=true",
        "enable-headers=true",
        "insert-seq-hdr=true",
        "iframeinterval=2",
        "idrinterval=2",
        "!",
        "filesink",
        f"location={ivf_path}",
    ]

    # ========================================================================
    # 局部功能块：创建当前 camera 的编码进度条
    #
    # label 来源：
    #
    #   DirectLeRobotDataset._encode_direct_video()
    #
    # 通常：
    #
    #   front
    #   left_wrist
    #   right_wrist
    #
    #
    # 输出：
    #
    #   progress
    #
    # 后续：
    #
    #   编码开始
    #       → update(0, ...)
    #
    #   每发送一帧
    #       → update(index, ...)
    #
    #   异常
    #       → break_line()
    # ========================================================================
    progress = TerminalProgressBar(label)

    # ========================================================================
    # 局部功能块：输出初始 0% 进度
    #
    # current：
    #   0
    #
    # total：
    #   len(frames)
    #
    # force=True：
    #
    #   无论之前状态如何都强制打印初始状态。
    #
    #
    # 这里只产生 UI 输出，
    # frames 本身还没有开始送入 GStreamer。
    # ========================================================================
    progress.update(
        0,
        len(frames),
        force=True,
    )

    # ========================================================================
    # 局部功能块：启动 GStreamer 子进程
    #
    # command：
    #   ← 上面构造的完整 pipeline。
    #
    #
    # stdin=subprocess.PIPE
    #
    #   关键数据入口。
    #
    #   Python 后面把每个 NumPy frame 的 bytes 写到：
    #
    #       process.stdin
    #
    #   GStreamer 的：
    #
    #       fdsrc fd=0
    #
    #   从这里读取。
    #
    #
    # stdout=subprocess.PIPE
    # stderr=subprocess.PIPE
    #
    #   捕获 GStreamer 标准输出和错误输出。
    #
    #   正常完成或异常时会读取，
    #   用于构造错误 details。
    #
    #
    # bufsize=0
    #
    #   为 subprocess pipe 使用无额外 Python 缓冲模式。
    #
    #
    # 输出：
    #
    #   process: subprocess.Popen
    #
    # 后续用途：
    #
    #   process.stdin
    #       → 写图像
    #
    #   process.wait()
    #       → 等编码完成
    #
    #   process.stdout/stderr
    #       → 错误诊断
    #
    #   _stop_subprocess(process)
    #       → 异常清理
    # ========================================================================
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    try:
        # ====================================================================
        # 局部功能块：确认 GStreamer stdin Pipe 存在
        #
        # process.stdin 来源：
        #
        #   subprocess.Popen(
        #       ...,
        #       stdin=subprocess.PIPE,
        #   )
        #
        #
        # 类型系统中 stdin 仍可能标记为 Optional，
        # 所以这里 assert：
        #
        #   process.stdin is not None
        #
        #
        # 后续：
        #
        #   process.stdin.write(...)
        #
        # 依赖这个对象。
        # ====================================================================
        assert process.stdin is not None

        # ====================================================================
        # 局部功能块：逐帧把 NumPy RGB 图像送给 GStreamer
        #
        # frames 来源：
        #
        #   DirectLeRobotDataset
        #       ↓
        #   当前 camera 的全部内存图像。
        #
        #
        # enumerate(..., start=1)：
        #
        #   frame
        #       → 当前 NumPy 图像
        #
        #   index
        #       → 已经处理到第几帧
        #
        # 从 1 开始是为了直接作为：
        #
        #   progress current
        #
        # 使用。
        #
        #
        # 每帧的数据链：
        #
        #   frame
        #     │
        #     ▼
        #   _rgb_frame()
        #     │
        #     │ 校验 uint8 / HWC / contiguous
        #     ▼
        #   ndarray
        #     │
        #     ▼
        #   .tobytes()
        #     │
        #     ▼
        #   process.stdin.write()
        #     │
        #     ▼
        #   gst-launch stdin
        #     │
        #     ▼
        #   fdsrc
        #     │
        #     ▼
        #   AV1 pipeline
        #
        #
        # 写完每一帧后：
        #
        #   progress.update(index, len(frames))
        #
        # 更新用户看到的编码进度。
        # ====================================================================
        for index, frame in enumerate(
            frames,
            start=1,
        ):
            process.stdin.write(
                _rgb_frame(
                    frame,
                    width,
                    height,
                ).tobytes()
            )

            progress.update(
                index,
                len(frames),
            )

        # ====================================================================
        # 局部功能块：所有图像发送完毕，关闭 GStreamer stdin
        #
        # 输入：
        #
        #   process.stdin
        #
        #
        # 到达这里代表：
        #
        #   frames 中所有 NumPy 图像都已经写入 pipe。
        #
        #
        # close()：
        #
        #   告诉 GStreamer：
        #
        #       stdin 不会再有新的 raw RGB 数据。
        #
        # 这会让 pipeline 获得输入结束信号，
        # 从而完成剩余编码并退出。
        #
        #
        # 下一步：
        #
        #   process.wait()
        #
        # 等待 GStreamer 真正完成。
        # ====================================================================
        process.stdin.close()

        # ====================================================================
        # 局部功能块：等待 GStreamer AV1 编码进程结束
        #
        # process：
        #   ← 前面启动的 gst-launch-1.0。
        #
        #
        # process.wait()：
        #
        #   阻塞等待编码完成。
        #
        #
        # 输出：
        #
        #   return_code
        #
        # 正常通常为：
        #
        #   0
        #
        # 非零表示 GStreamer pipeline 执行失败。
        #
        #
        # return_code 后续会传给：
        #
        #   if return_code != 0 ...
        #
        # 做最终成功判断。
        # ====================================================================
        return_code = process.wait()

        # ====================================================================
        # 局部功能块：读取 GStreamer stdout
        #
        # process.stdout 来源：
        #
        #   Popen(stdout=subprocess.PIPE)
        #
        #
        # read()：
        #   得到 bytes。
        #
        # decode(..., errors="replace")：
        #
        #   转成 Python str；
        #   无法解码的字节用替代字符，
        #   避免错误日志本身因为 UnicodeDecodeError 丢失。
        #
        #
        # 输出：
        #
        #   stdout: str
        #
        # 后续仅在失败时组成：
        #
        #   details
        #
        # 给 HardwareVideoEncodingError。
        # ====================================================================
        stdout = (
            process.stdout.read().decode(
                "utf-8",
                errors="replace",
            )
            if process.stdout
            else ""
        )

        # ====================================================================
        # 局部功能块：读取 GStreamer stderr
        #
        # 数据来源和 stdout 相同，
        # 但来自：
        #
        #   Popen(stderr=subprocess.PIPE)
        #
        #
        # 输出：
        #
        #   stderr: str
        #
        # 后续：
        #
        #   stdout + stderr
        #       ↓
        #   details
        #       ↓
        #   HardwareVideoEncodingError
        # ====================================================================
        stderr = (
            process.stderr.read().decode(
                "utf-8",
                errors="replace",
            )
            if process.stderr
            else ""
        )

    except KeyboardInterrupt:
        # ====================================================================
        # 局部功能块：用户在 GStreamer 编码过程中 Ctrl+C
        #
        # KeyboardInterrupt 来源：
        #
        #   通常用户按 Ctrl+C。
        #
        #
        # 当前处理顺序：
        #
        #   1. progress.break_line()
        #
        #      如果进度条正在同一行刷新，
        #      先补换行。
        #
        #
        #   2. _stop_subprocess(process)
        #
        #      停止 GStreamer 子进程。
        #
        #
        #   3. 删除：
        #
        #      ivf_path
        #      remux_path
        #      video_path
        #
        #      防止留下不完整文件。
        #
        #
        #   4. raise
        #
        #      不是吞掉 Ctrl+C，
        #      而是把原 KeyboardInterrupt 继续抛给上层。
        #
        #
        # 异常去向：
        #
        #   DirectLeRobotDataset
        #       ↓
        #   converter
        #
        # 由更上层决定整个转换任务如何停止。
        # ====================================================================
        progress.break_line()
        _stop_subprocess(process)

        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        raise

    except Exception as exc:
        # ====================================================================
        # 局部功能块：处理 GStreamer 数据写入/运行阶段的其他异常
        #
        # exc 可能来自：
        #
        #   _rgb_frame()
        #       → frame 格式不合法
        #
        #   process.stdin.write()
        #       → pipe/进程异常
        #
        #   process.wait()
        #
        #   stdout/stderr 读取
        #
        #   等其他 Python 运行时问题。
        #
        #
        # 当前处理：
        #
        #   1. 结束进度条行
        #
        #   2. 停止 GStreamer
        #
        #   3. 尽量读取 stdout/stderr
        #
        #   4. 删除所有可能的中间文件
        #
        #   5. 将底层异常统一包装成
        #      HardwareVideoEncodingError
        # ====================================================================
        progress.break_line()
        _stop_subprocess(process)

        # ====================================================================
        # GStreamer 即使异常，
        # 仍然尽量读取它已经产生的 stdout，
        # 为上层提供更完整的诊断信息。
        # ====================================================================
        stdout = (
            process.stdout.read().decode(
                "utf-8",
                errors="replace",
            )
            if process.stdout
            else ""
        )

        # ====================================================================
        # 同上，读取 stderr。
        #
        # stderr 往往是外部视频工具错误原因的重要来源。
        # ====================================================================
        stderr = (
            process.stderr.read().decode(
                "utf-8",
                errors="replace",
            )
            if process.stderr
            else ""
        )

        # ====================================================================
        # 局部功能块：清理失败任务产生的所有文件
        #
        # ivf_path：
        #   可能是只写了一部分的 AV1/IVF。
        #
        # remux_path：
        #   理论上此阶段通常还没有进入 FFmpeg，
        #   但为了保证状态干净仍统一清除。
        #
        # video_path：
        #   防止出现旧文件或不完整最终文件。
        #
        #
        # missing_ok=True：
        #   没有产生对应文件也不影响错误处理。
        # ====================================================================
        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        # ====================================================================
        # 局部功能块：合并 GStreamer 输出信息
        #
        # 输入：
        #
        #   stdout
        #   stderr
        #
        #
        # part.strip()：
        #   去掉首尾空白。
        #
        # if part.strip()：
        #   跳过空字符串。
        #
        # "\n".join(...)：
        #
        #   把两路诊断输出合并成一个 details。
        #
        #
        # 输出：
        #
        #   details
        #
        # 下一步：
        #
        #   details[-2000:]
        #
        # 只把末尾最多 2000 个字符放进最终异常，
        # 防止非常长的外部程序日志把错误消息无限放大。
        # ====================================================================
        details = "\n".join(
            part.strip()
            for part in (stdout, stderr)
            if part.strip()
        )

        # ====================================================================
        # 局部功能块：把底层异常转换为统一的视频编码异常
        #
        # 输入：
        #
        #   type(exc).__name__
        #       → 原异常类型。
        #
        #   exc
        #       → 原异常消息。
        #
        #   details[-2000:]
        #       → GStreamer 最后部分日志。
        #
        #
        # raise ... from exc：
        #
        #   保留 Python exception chaining。
        #
        # 上层既可以看到：
        #
        #   HardwareVideoEncodingError
        #
        # 也可以通过异常链追到原始 exc。
        #
        #
        # 异常去向：
        #
        #   _encode_direct_video()
        #       ↓
        #   save_episode()
        #       ↓
        #   converter 上层错误处理
        # ====================================================================
        raise HardwareVideoEncodingError(
            f"Jetson AV1 pipeline failed: "
            f"{type(exc).__name__}: {exc}; "
            f"{details[-2000:]}"
        ) from exc

    # ========================================================================
    # 局部功能块：检查 GStreamer 是否真正成功生成 IVF
    #
    # 到达这里表示：
    #
    #   Python try 块本身没有抛异常。
    #
    # 但外部进程仍可能：
    #
    #   return_code != 0
    #
    # 或虽然退出码正常，但：
    #
    #   ivf_path 根本不存在。
    #
    #
    # 输入：
    #
    #   return_code
    #       ← process.wait()
    #
    #   ivf_path
    #       ← 编码临时输出路径
    #
    #
    # 成功要求：
    #
    #   return_code == 0
    #
    # 并且：
    #
    #   ivf_path.is_file() == True
    #
    #
    # 失败：
    #
    #   清理文件
    #       ↓
    #   stdout/stderr → details
    #       ↓
    #   HardwareVideoEncodingError
    #
    #
    # 成功：
    #
    #   ivf_path 继续传给下一阶段 FFmpeg remux。
    # ========================================================================
    if (
        return_code != 0
        or not ivf_path.is_file()
    ):
        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        # ====================================================================
        # 和前面的 Python 异常路径一样，
        # 把 GStreamer stdout/stderr 合并成诊断信息。
        # ====================================================================
        details = "\n".join(
            part.strip()
            for part in (stdout, stderr)
            if part.strip()
        )

        # ====================================================================
        # 此处不是 Python API 调用失败，
        # 而是：
        #
        #   GStreamer 自己以失败状态结束
        #
        # 或：
        #
        #   没有创建预期 IVF 文件。
        #
        # 所以错误消息明确包含：
        #
        #   exit code
        #
        # 方便区分失败阶段。
        # ====================================================================
        raise HardwareVideoEncodingError(
            f"Jetson AV1 encoder failed with exit code "
            f"{return_code}: {details[-2000:]}"
        )

    # ========================================================================
    # 到达这里：
    #
    #   GStreamer AV1 编码阶段成功。
    #
   # 已得到：
    #
    #   ivf_path
    #
    # 但最终 LeRobot 需要的是：
    #
    #   video_path → .mp4
    #
    # 所以下一步进入 FFmpeg remux。
    # ========================================================================
    print(
        f"[LeRobot转换][阶段 7/8][MP4封装][{label}] "
        "硬件AV1编码完成，正在无损remux IVF → MP4",
        flush=True,
    )

    # ========================================================================
    # 局部功能块：构造 FFmpeg remux 命令
    #
    # 外部程序：
    #
    #   ffmpeg
    #
    #
    # 输入：
    #
    #   ivf_path
    #       ← GStreamer 硬件 AV1 编码结果。
    #
    #
    # 输出：
    #
    #   remux_path
    #       ← 临时 MP4。
    #
    #
    # 参数含义：
    #
    #   -hide_banner
    #       减少 FFmpeg 启动 banner 输出。
    #
    #   -loglevel error
    #       只保留 error 级别日志。
    #
    #   -f ivf
    #       明确告诉 FFmpeg 输入按 IVF 读取。
    #
    #   -i ivf_path
    #       输入临时编码文件。
    #
    #   -c:v copy
    #
    #       关键：
    #
    #       不重新编码 AV1 视频，
    #       直接复制已经编码好的 video bitstream。
    #
    #       因此这里做的是容器封装转换，
    #       而不是第二次有损视频压缩。
    #
    #   -y
    #       允许覆盖输出文件。
    #
    #
    # 最终：
    #
    #   remux_command
    #
    # 传给：
    #
    #   subprocess.Popen()
    # ========================================================================
    remux_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "ivf",
        "-i",
        str(ivf_path),
        "-c:v",
        "copy",
        "-y",
        str(remux_path),
    ]

    # ========================================================================
    # 局部功能块：启动 FFmpeg remux 子进程
    #
    # 输入：
    #
    #   remux_command
    #       ← 上一步构造。
    #
    #
    # stdout/stderr 使用 PIPE：
    #
    #   之后 communicate() 会一次性等待进程并取得输出。
    #
    #
    # 输出：
    #
    #   remux_process
    #
    # 后续：
    #
    #   communicate()
    #       → 等待并读取输出
    #
    #   returncode
    #       → 判断成功
    #
    #   _stop_subprocess()
    #       → 异常终止
    # ========================================================================
    remux_process = subprocess.Popen(
        remux_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # ====================================================================
        # 局部功能块：等待 FFmpeg 完成并收集 stdout/stderr
        #
        # communicate()：
        #
        #   等待 FFmpeg 进程结束，
        #   同时读取它的 stdout 和 stderr。
        #
        #
        # 输出：
        #
        #   remux_stdout: bytes
        #
        #   remux_stderr: bytes
        #
        #
        # 后续：
        #
        #   如果 remux_process.returncode != 0，
        #
        #   两者会 decode 后组成 details。
        # ====================================================================
        remux_stdout, remux_stderr = (
            remux_process.communicate()
        )

    except KeyboardInterrupt:
        # ====================================================================
        # 局部功能块：用户在 FFmpeg remux 阶段 Ctrl+C
        #
        # 此时 GStreamer 阶段已经完成，
        # ivf_path 通常已经存在。
        #
        #
        # 处理：
        #
        #   _stop_subprocess(remux_process)
        #       → 停止 FFmpeg
        #
        #   删除：
        #       ivf_path
        #       remux_path
        #       video_path
        #
        #   raise
        #       → 继续把 KeyboardInterrupt 传给上层。
        #
        #
        # 这样不会留下：
        #
        #   编了一半的 MP4
        #   或本次转换的 IVF 临时文件。
        # ====================================================================
        _stop_subprocess(remux_process)

        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        raise

    except Exception as exc:
        # ====================================================================
        # 局部功能块：FFmpeg Python 调用阶段发生其他异常
        #
        # exc 来源可能是：
        #
        #   communicate()
        #   subprocess 通信
        #   其他 Python 运行异常。
        #
        #
        # 处理：
        #
        #   1. 停止 FFmpeg
        #   2. 删除所有临时/目标文件
        #   3. 包装成 HardwareVideoEncodingError
        #
        #
        # 和 GStreamer 异常路径相比：
        #
        #   这里错误类型明确写：
        #
        #       AV1 IVF remux failed
        #
        # 表示 AV1 编码本身已经经过，
        # 失败发生在 MP4 封装阶段。
        # ====================================================================
        _stop_subprocess(remux_process)

        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        raise HardwareVideoEncodingError(
            f"AV1 IVF remux failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # ========================================================================
    # 局部功能块：验证 FFmpeg remux 是否成功
    #
    # 输入：
    #
    #   remux_process.returncode
    #       ← FFmpeg 进程退出码。
    #
    #   remux_path
    #       ← 预期生成的临时 MP4。
    #
    #
    # 成功要求：
    #
    #   returncode == 0
    #
    # 且：
    #
    #   remux_path.is_file()
    #
    #
    # 如果失败：
    #
    #   remux_stdout / remux_stderr
    #       ↓
    #   decode
    #       ↓
    #   details
    #       ↓
    #   删除文件
    #       ↓
    #   HardwareVideoEncodingError
    # ========================================================================
    if (
        remux_process.returncode != 0
        or not remux_path.is_file()
    ):
        # ====================================================================
        # 局部功能块：解码 FFmpeg 诊断输出
        #
        # remux_stdout / remux_stderr：
        #
        #   communicate() 返回的是 bytes。
        #
        #
        # decode(
        #     "utf-8",
        #     errors="replace"
        # )
        #
        #   转成字符串。
        #
        #
        # strip()：
        #   去掉首尾空白。
        #
        # if value.strip()：
        #   跳过空输出。
        #
        #
        # 输出：
        #
        #   details
        #
        # 下一步放入异常消息，
        # 同样只保留最后 2000 字符。
        # ====================================================================
        details = "\n".join(
            value.decode(
                "utf-8",
                errors="replace",
            ).strip()
            for value in (
                remux_stdout,
                remux_stderr,
            )
            if value.strip()
        )

        # ====================================================================
        # 局部功能块：清理失败的 remux 结果
        #
        # 删除：
        #
        #   ivf_path
        #       → 硬件编码中间文件
        #
        #   remux_path
        #       → 不完整 MP4
        #
        #   video_path
        #       → 防止旧目标文件残留
        #
        # 清理后再向上抛异常。
        # ====================================================================
        ivf_path.unlink(missing_ok=True)
        remux_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)

        # ====================================================================
        # 局部功能块：报告 FFmpeg 非零退出/未产生文件
        #
        # 输入：
        #
        #   remux_process.returncode
        #
        #   details[-2000:]
        #
        #
        # 输出：
        #
        #   HardwareVideoEncodingError
        #
        # 向上：
        #
        #   DirectLeRobotDataset._encode_direct_video()
        #       ↓
        #   save_episode()
        #       ↓
        #   converter
        # ====================================================================
        raise HardwareVideoEncodingError(
            f"AV1 IVF remux failed with exit code "
            f"{remux_process.returncode}: "
            f"{details[-2000:]}"
        )

    # ========================================================================
    # 局部功能块：把成功的临时 MP4 原子式切换为最终目标路径
    #
    # 到达这里已经确认：
    #
    #   FFmpeg returncode == 0
    #
    # 且：
    #
    #   remux_path 是实际文件。
    #
    #
    # 输入：
    #
    #   remux_path
    #       ← FFmpeg 生成。
    #
    #   video_path
    #       ← 上游 DirectLeRobotDataset 指定的最终视频位置。
    #
    #
    # Path.replace(video_path)：
    #
    #   把：
    #
    #       .xxx.remux.tmp.mp4
    #
    #   替换/移动为：
    #
    #       最终 xxx.mp4
    #
    #
    # 完成后：
    #
    #   remux_path 不再作为单独临时文件存在，
    #
    #   最终用户/LeRobot 数据集看到的是：
    #
    #       video_path
    #
    #
    # video_path 之后会被 LeRobot metadata 引用，
    # 成为当前 episode 的正式视频文件。
    # ========================================================================
    remux_path.replace(video_path)

    # ========================================================================
    # 局部功能块：最终成功后删除 IVF 中间文件
    #
    # ivf_path：
    #
    #   只是在：
    #
    #       Jetson AV1 encoder
    #           ↓
    #       FFmpeg MP4 remux
    #
    #   两阶段之间使用。
    #
    #
    # 此时：
    #
    #   remux 已成功
    #   video_path 已正式生成
    #
    # 所以 IVF 已经没有继续保留的必要。
    #
    #
    # 最终文件状态：
    #
    #   ivf_path
    #       → 删除
    #
    #   remux_path
    #       → 已 replace 成 video_path
    #
    #   video_path
    #       → 保留，作为最终输出 MP4
    #
    #
    # 函数随后自然返回 None。
    #
    #
    # 整条视频数据链到此完成：
    #
    #   NumPy RGB frames
    #          │
    #          ▼
    #     raw RGB bytes
    #          │
    #          ▼
    #      GStreamer
    #          │
    #          ▼
    #    Jetson AV1 编码
    #          │
    #          ▼
    #      临时 IVF
    #          │
    #          ▼
    #    FFmpeg -c:v copy
    #          │
    #          ▼
    #      临时 MP4
    #          │
    #          ▼
    #      video_path
    #          │
    #          ▼
    #   LeRobot episode 视频
    # ========================================================================
    ivf_path.unlink(missing_ok=True)
