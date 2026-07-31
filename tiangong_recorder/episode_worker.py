from __future__ import annotations

import time
import traceback
from multiprocessing.connection import Connection
from typing import Dict

import zmq

from .config import RecorderConfig
from .dataset_writer import write_episode_pickle
from .image_decoder import decode_ros_image
from .synchronizer import LiveEpisodeSynchronizer


def run_episode_worker(
    episode_id: str,
    config: RecorderConfig,
    control_connection: Connection,
) -> None:
    """Record, align, and save one episode in a dedicated process."""

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    synchronizer = LiveEpisodeSynchronizer(config.camera_topics.keys())
    last_frame_monotonic: Dict[str, float | None] = {
        name: None for name in config.camera_topics
    }
    camera_errors: Dict[str, str] = {}

    class EpisodeCameraNode(Node):
        def __init__(self):
            super().__init__(f"tiangong_episode_{episode_id[:32]}")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._camera_subscriptions = []
            for camera_name, topic in config.camera_topics.items():
                subscription = self.create_subscription(
                    Image,
                    topic,
                    lambda message, name=camera_name: self._camera_callback(
                        name,
                        message,
                    ),
                    qos,
                )
                self._camera_subscriptions.append(subscription)

        def _camera_callback(self, camera_name: str, message: Image) -> None:
            try:
                frame = decode_ros_image(
                    message,
                    expected_width=config.image_width,
                    expected_height=config.image_height,
                    fallback_timestamp_ns=time.time_ns(),
                )
                synchronizer.add_camera_frame(camera_name, frame)
                last_frame_monotonic[camera_name] = time.monotonic()
                camera_errors.pop(camera_name, None)
            except Exception as exc:
                camera_errors[camera_name] = str(exc)

    context = zmq.Context()
    state_socket = context.socket(zmq.PULL)
    state_socket.setsockopt(zmq.LINGER, 0)
    state_socket.setsockopt(zmq.RCVHWM, config.state_receive_hwm)
    state_port = state_socket.bind_to_random_port(
        "tcp://0.0.0.0",
        min_port=config.state_port_min,
        max_port=config.state_port_max,
        max_tries=100,
    )

    rclpy.init(args=None)
    node = EpisodeCameraNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    ready_sent = False
    startup_deadline = time.monotonic() + config.startup_timeout_s
    stop_expected_count: int | None = None
    stop_received_monotonic: float | None = None
    tail_deadline: float | None = None
    discard_requested = False
    received_state_count = 0
    runtime_closed = False

    try:
        while True:
            executor.spin_once(timeout_sec=0.002)

            while state_socket.poll(timeout=0):
                state = state_socket.recv_pyobj()
                if state.get("_episode_id") != episode_id:
                    raise RuntimeError("received state for a different episode")
                frame_index = int(state["_frame_index"])
                if frame_index != received_state_count:
                    raise RuntimeError(
                        f"state frame discontinuity: expected "
                        f"{received_state_count}, got {frame_index}"
                    )
                synchronizer.add_state(state)
                received_state_count += 1

            while control_connection.poll():
                command = control_connection.recv()
                command_type = command.get("type")
                if command_type == "STOP_SAVE":
                    stop_expected_count = int(command["expected_state_count"])
                    stop_received_monotonic = time.monotonic()
                elif command_type == "DISCARD":
                    discard_requested = True
                else:
                    raise RuntimeError(f"unsupported worker command: {command_type}")

            if discard_requested:
                control_connection.send(
                    {
                        "type": "discarded",
                        "episode_id": episode_id,
                    }
                )
                return

            if synchronizer.cameras_ready() and not ready_sent:
                control_connection.send(
                    {
                        "type": "ready",
                        "episode_id": episode_id,
                        "state_addr": f"tcp://{config.advertise_host}:{state_port}",
                    }
                )
                ready_sent = True

            if not ready_sent:
                if time.monotonic() >= startup_deadline:
                    details = camera_errors or {
                        name: "no frame received"
                        for name, value in last_frame_monotonic.items()
                        if value is None
                    }
                    raise TimeoutError(
                        f"camera startup timed out: {details}"
                    )
                continue

            synchronizer.merge_ready(force=False)

            if stop_expected_count is None:
                now = time.monotonic()
                stale_cameras = [
                    name
                    for name, last_time in last_frame_monotonic.items()
                    if last_time is None
                    or now - last_time > config.camera_stale_timeout_s
                ]
                if stale_cameras:
                    raise RuntimeError(
                        f"camera stream stale: {', '.join(stale_cameras)}"
                    )
                continue

            if received_state_count > stop_expected_count:
                raise RuntimeError(
                    f"received {received_state_count} states, "
                    f"but STOP_SAVE declared {stop_expected_count}"
                )

            if received_state_count < stop_expected_count:
                assert stop_received_monotonic is not None
                if (
                    time.monotonic() - stop_received_monotonic
                    > config.state_drain_timeout_s
                ):
                    raise TimeoutError(
                        f"state drain timed out: received "
                        f"{received_state_count}/{stop_expected_count}"
                    )
                continue

            if not synchronizer.pending_states:
                break

            if tail_deadline is None:
                tail_deadline = time.monotonic() + config.tail_wait_timeout_s
            if time.monotonic() >= tail_deadline:
                synchronizer.merge_ready(force=True)
                if synchronizer.pending_states:
                    raise RuntimeError(
                        "unable to match remaining states with camera frames"
                    )

        control_connection.send(
            {
                "type": "saving",
                "episode_id": episode_id,
                "count": len(synchronizer.episode_data),
            }
        )

        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
        state_socket.close()
        context.term()
        runtime_closed = True

        final_path = write_episode_pickle(
            episode_id,
            synchronizer.episode_data,
            config.output_dir,
        )
        control_connection.send(
            {
                "type": "saved",
                "episode_id": episode_id,
                "count": len(synchronizer.episode_data),
                "path": str(final_path),
                "alignment": synchronizer.alignment_summary(),
            }
        )
    except Exception as exc:
        try:
            control_connection.send(
                {
                    "type": "error",
                    "episode_id": episode_id,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        if not runtime_closed:
            state_socket.close()
            context.term()
        control_connection.close()
