from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Deque, Dict

import zmq

from .config import RecorderConfig
from .episode_worker import run_episode_worker
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


@dataclass
class WorkerHandle:
    episode_id: str
    process: mp.Process
    connection: Connection
    status: str
    state_addr: str | None = None
    last_message: dict | None = None


class RecorderServer:
    def __init__(self, config: RecorderConfig):
        config.validate()
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(config.control_bind_addr)
        self.active: WorkerHandle | None = None
        self.finishing: Dict[str, WorkerHandle] = {}
        self.recent: Deque[dict] = deque(maxlen=20)
        self.running = True
        self.mp_context = mp.get_context("spawn")

    def _start(self, request: dict) -> dict:
        if self.active is not None and self.active.process.is_alive():
            return error(
                "another episode is already recording",
                status="busy",
                active_episode_id=self.active.episode_id,
            )

        episode_id = validate_episode_id(request.get("episode_id"))
        parent_connection, child_connection = self.mp_context.Pipe(duplex=True)
        process = self.mp_context.Process(
            target=run_episode_worker,
            args=(episode_id, self.config, child_connection),
            name=f"episode-{episode_id}",
        )
        process.start()
        child_connection.close()
        handle = WorkerHandle(
            episode_id=episode_id,
            process=process,
            connection=parent_connection,
            status="starting",
        )

        deadline = time.monotonic() + self.config.startup_timeout_s + 2.0
        while time.monotonic() < deadline:
            if parent_connection.poll(0.1):
                message = parent_connection.recv()
                handle.last_message = message
                if message.get("type") == "ready":
                    handle.status = "recording"
                    handle.state_addr = message["state_addr"]
                    self.active = handle
                    print(
                        f"[READY] {episode_id} state={handle.state_addr}",
                        flush=True,
                    )
                    return ok(
                        status="ready",
                        episode_id=episode_id,
                        state_addr=handle.state_addr,
                    )
                if message.get("type") == "error":
                    process.join(timeout=1.0)
                    parent_connection.close()
                    return error(
                        message.get("message", "worker startup failed"),
                        status="error",
                        episode_id=episode_id,
                    )
            if not process.is_alive():
                process.join(timeout=0.1)
                parent_connection.close()
                return error(
                    "episode worker exited during startup",
                    status="error",
                    episode_id=episode_id,
                )

        process.terminate()
        process.join(timeout=2.0)
        parent_connection.close()
        return error(
            "episode worker startup timed out",
            status="error",
            episode_id=episode_id,
        )

    def _stop_save(self, request: dict) -> dict:
        episode_id = validate_episode_id(request.get("episode_id"))
        if self.active is None or self.active.episode_id != episode_id:
            return error(
                "episode is not currently recording",
                status="not_recording",
                episode_id=episode_id,
            )
        expected_state_count = int(request.get("expected_state_count", -1))
        if expected_state_count < 0:
            return error("expected_state_count must be non-negative")

        handle = self.active
        handle.connection.send(
            {
                "type": STOP_SAVE,
                "expected_state_count": expected_state_count,
            }
        )
        handle.status = "finalizing"
        self.finishing[episode_id] = handle
        self.active = None
        print(
            f"[STOP_SAVE] {episode_id} expected={expected_state_count}",
            flush=True,
        )
        return ok(
            status="finalizing",
            episode_id=episode_id,
            expected_state_count=expected_state_count,
        )

    def _discard(self, request: dict) -> dict:
        episode_id = validate_episode_id(request.get("episode_id"))
        if self.active is None or self.active.episode_id != episode_id:
            return error(
                "episode is not currently recording",
                status="not_recording",
                episode_id=episode_id,
            )

        handle = self.active
        handle.connection.send({"type": DISCARD})
        handle.status = "discarding"
        self.finishing[episode_id] = handle
        self.active = None
        print(f"[DISCARD] {episode_id}", flush=True)
        return ok(status="discarding", episode_id=episode_id)

    def _status(self, request: dict) -> dict:
        episode_id = request.get("episode_id")
        if episode_id:
            episode_id = validate_episode_id(episode_id)
            if self.active is not None and self.active.episode_id == episode_id:
                return ok(
                    status=self.active.status,
                    episode_id=episode_id,
                    state_addr=self.active.state_addr,
                )
            handle = self.finishing.get(episode_id)
            if handle is not None:
                return ok(
                    status=handle.status,
                    episode_id=episode_id,
                    details=handle.last_message,
                )
            for result in reversed(self.recent):
                if result.get("episode_id") == episode_id:
                    return ok(
                        status=result.get("type", "unknown"),
                        episode_id=episode_id,
                        details=result,
                    )
            return error(
                "episode is unknown",
                status="unknown",
                episode_id=episode_id,
            )

        return ok(
            status="recording" if self.active else "idle",
            active_episode_id=self.active.episode_id if self.active else None,
            finalizing_episode_ids=list(self.finishing),
        )

    def _handle_request(self, request: dict) -> dict:
        if int(request.get("protocol_version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
            return error("unsupported protocol version")
        request_type = request.get("type")
        if request_type == START:
            return self._start(request)
        if request_type == STOP_SAVE:
            return self._stop_save(request)
        if request_type == DISCARD:
            return self._discard(request)
        if request_type == STATUS:
            return self._status(request)
        if request_type == PING:
            return ok(
                status="alive",
                server_time_ns=time.time_ns(),
            )
        if request_type == SHUTDOWN:
            self.running = False
            return ok(status="shutting_down")
        return error(f"unsupported request type: {request_type}")

    def _poll_workers(self) -> None:
        handles = list(self.finishing.values())
        if self.active is not None:
            handles.append(self.active)

        for handle in handles:
            try:
                while handle.connection.poll():
                    message = handle.connection.recv()
                    handle.last_message = message
                    message_type = message.get("type", "unknown")
                    handle.status = message_type
                    if message_type in {"saved", "discarded", "error"}:
                        self.recent.append(message)
                        print(f"[{message_type.upper()}] {message}", flush=True)
                        if (
                            self.active is not None
                            and self.active.episode_id == handle.episode_id
                        ):
                            self.finishing[handle.episode_id] = handle
                            self.active = None
            except EOFError:
                pass

            if not handle.process.is_alive():
                handle.process.join(timeout=0.1)
                if handle.last_message is None:
                    handle.last_message = {
                        "type": "error",
                        "episode_id": handle.episode_id,
                        "message": (
                            f"worker exited with code {handle.process.exitcode}"
                        ),
                    }
                    self.recent.append(handle.last_message)
                self.finishing.pop(handle.episode_id, None)
                if (
                    self.active is not None
                    and self.active.episode_id == handle.episode_id
                ):
                    self.active = None
                handle.connection.close()

    def serve_forever(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        print(
            f"Recorder server listening on {self.config.control_bind_addr}",
            flush=True,
        )
        while self.running:
            self._poll_workers()
            events = dict(poller.poll(timeout=100))
            if self.socket not in events:
                continue
            try:
                request = self.socket.recv_json()
                reply = self._handle_request(request)
            except Exception as exc:
                reply = error(str(exc))
            self.socket.send_json(reply)

    def close(self) -> None:
        if self.active is not None:
            try:
                self.active.connection.send({"type": DISCARD})
            except Exception:
                pass
        deadline = time.monotonic() + 3.0
        handles = list(self.finishing.values())
        if self.active is not None:
            handles.append(self.active)
        for handle in handles:
            remaining = max(0.0, deadline - time.monotonic())
            handle.process.join(timeout=remaining)
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=1.0)
            handle.connection.close()
        self.socket.close()
        self.context.term()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/recorder.yaml",
        help="Path to recorder YAML configuration",
    )
    args = parser.parse_args()

    config = RecorderConfig.from_yaml(args.config)
    server = RecorderServer(config)

    def stop_server(_signum, _frame):
        server.running = False

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
