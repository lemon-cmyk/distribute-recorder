from __future__ import annotations

import os
import pickle
from pathlib import Path

from .protocol import validate_episode_id


def write_episode_pickle(
    episode_id: str,
    episode_data: list[dict],
    output_dir: str | Path,
) -> Path:
    validate_episode_id(episode_id)
    if not episode_data:
        raise ValueError("refusing to write an empty episode")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_path = output_path / f"{episode_id}.pkl"
    temporary_path = output_path / f"{episode_id}.pkl.tmp"

    if final_path.exists():
        raise FileExistsError(f"dataset already exists: {final_path}")

    with temporary_path.open("wb") as stream:
        pickle.dump(episode_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, final_path)
    return final_path

