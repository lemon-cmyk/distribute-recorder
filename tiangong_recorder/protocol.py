from __future__ import annotations

import re


PROTOCOL_VERSION = 1

START = "START"
STOP_SAVE = "STOP_SAVE"
DISCARD = "DISCARD"
STATUS = "STATUS"
PING = "PING"
SHUTDOWN = "SHUTDOWN"

_EPISODE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def validate_episode_id(value: object) -> str:
    episode_id = str(value or "")
    if not _EPISODE_ID_RE.fullmatch(episode_id):
        raise ValueError("episode_id must contain only letters, digits, '.', '_' or '-'")
    return episode_id


def ok(**payload):
    return {"ok": True, "protocol_version": PROTOCOL_VERSION, **payload}


def error(message: str, **payload):
    return {
        "ok": False,
        "protocol_version": PROTOCOL_VERSION,
        "message": str(message),
        **payload,
    }

