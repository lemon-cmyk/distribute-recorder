#!/usr/bin/env python3
"""Merge multiple compatible LeRobot v2.1 datasets into one dataset.

The source datasets are never modified.  The destination is assembled in a
temporary sibling directory and renamed into place only after validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise SystemExit(
        "Missing dependency. Run this script with an environment containing "
        "numpy and pyarrow (for this project: .venv-lerobot/bin/python)."
    ) from exc


META_FILENAMES = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
COMPATIBILITY_FIELDS = (
    "robot_type",
    "fps",
    "chunks_size",
    "data_path",
    "video_path",
    "features",
)


class MergeError(RuntimeError):
    """Raised when a source dataset is invalid or incompatible."""


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict[str, Any]
    tasks: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    episode_stats: dict[int, dict[str, Any]]

    @property
    def video_keys(self) -> list[str]:
        return sorted(
            key
            for key, feature in self.info["features"].items()
            if feature.get("dtype") == "video"
        )


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MergeError(f"Cannot read JSONL file {path}: {exc}") from exc
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def format_episode_path(template: str, episode_index: int, chunks_size: int, **extra: str) -> Path:
    fields: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // chunks_size,
    }
    fields.update(extra)
    try:
        return Path(template.format(**fields))
    except (KeyError, ValueError) as exc:
        raise MergeError(f"Unsupported LeRobot path template {template!r}: {exc}") from exc


def load_source(root: Path, *, check_files: bool = True) -> SourceDataset:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise MergeError(f"Source is not a directory: {root}")
    for filename in META_FILENAMES:
        if not (root / "meta" / filename).is_file():
            raise MergeError(f"Missing metadata file: {root / 'meta' / filename}")

    info = read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise MergeError(
            f"{root} has codebase_version={info.get('codebase_version')!r}; expected 'v2.1'"
        )
    missing_info = [
        key
        for key in (*COMPATIBILITY_FIELDS, "total_episodes", "total_frames", "total_tasks")
        if key not in info
    ]
    if missing_info:
        raise MergeError(f"{root}/meta/info.json is missing fields: {', '.join(missing_info)}")
    if not isinstance(info["chunks_size"], int) or info["chunks_size"] <= 0:
        raise MergeError(f"Invalid chunks_size in {root}/meta/info.json")

    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    stats_rows = read_jsonl(root / "meta" / "episodes_stats.jsonl")

    try:
        task_indices = [int(row["task_index"]) for row in tasks]
        episode_indices = [int(row["episode_index"]) for row in episodes]
        stats_by_index = {int(row["episode_index"]): row for row in stats_rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise MergeError(f"Malformed metadata indices in {root}/meta: {exc}") from exc

    expected_tasks = list(range(len(tasks)))
    if sorted(task_indices) != expected_tasks or len(set(task_indices)) != len(tasks):
        raise MergeError(f"Task indices in {root} must be contiguous from 0")
    expected_episodes = list(range(len(episodes)))
    if sorted(episode_indices) != expected_episodes or len(set(episode_indices)) != len(episodes):
        raise MergeError(f"Episode indices in {root} must be contiguous from 0")
    if sorted(stats_by_index) != expected_episodes or len(stats_rows) != len(episodes):
        raise MergeError(f"Episode stats in {root} do not match episodes.jsonl")
    if info["total_tasks"] != len(tasks) or info["total_episodes"] != len(episodes):
        raise MergeError(f"Totals in {root}/meta/info.json do not match metadata rows")

    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    episodes = [episodes_by_index[index] for index in expected_episodes]
    total_frames = 0
    for episode in episodes:
        try:
            length = int(episode["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MergeError(f"Invalid episode length in {root}: {episode!r}") from exc
        if length < 0:
            raise MergeError(f"Negative episode length in {root}: {episode!r}")
        total_frames += length
    if info["total_frames"] != total_frames:
        raise MergeError(
            f"total_frames mismatch in {root}: info={info['total_frames']}, episodes={total_frames}"
        )

    source = SourceDataset(root, info, tasks, episodes, stats_by_index)
    if check_files:
        validate_episode_files(source)
    return source


def validate_episode_files(source: SourceDataset) -> None:
    chunks_size = source.info["chunks_size"]
    for episode in source.episodes:
        episode_index = int(episode["episode_index"])
        data_rel = format_episode_path(source.info["data_path"], episode_index, chunks_size)
        data_path = source.root / data_rel
        if not data_path.is_file():
            raise MergeError(f"Missing Parquet file: {data_path}")
        try:
            row_count = pq.ParquetFile(data_path).metadata.num_rows
        except Exception as exc:
            raise MergeError(f"Cannot read Parquet metadata {data_path}: {exc}") from exc
        if row_count != int(episode["length"]):
            raise MergeError(
                f"Row count mismatch in {data_path}: parquet={row_count}, "
                f"metadata={episode['length']}"
            )
        for video_key in source.video_keys:
            video_rel = format_episode_path(
                source.info["video_path"], episode_index, chunks_size, video_key=video_key
            )
            video_path = source.root / video_rel
            if not video_path.is_file():
                raise MergeError(f"Missing video file: {video_path}")


def ensure_compatible(sources: list[SourceDataset]) -> None:
    reference = sources[0]
    for source in sources[1:]:
        differences = [
            field
            for field in COMPATIBILITY_FIELDS
            if source.info.get(field) != reference.info.get(field)
        ]
        if differences:
            raise MergeError(
                f"Dataset {source.root} is incompatible with {reference.root}; "
                f"different fields: {', '.join(differences)}"
            )


def build_task_maps(
    sources: list[SourceDataset],
) -> tuple[list[dict[str, Any]], list[dict[int, int]]]:
    merged_tasks: list[dict[str, Any]] = []
    task_to_new_index: dict[str, int] = {}
    source_maps: list[dict[int, int]] = []
    for source in sources:
        source_map: dict[int, int] = {}
        for task_row in sorted(source.tasks, key=lambda row: int(row["task_index"])):
            task = task_row.get("task")
            if not isinstance(task, str):
                raise MergeError(f"Invalid task row in {source.root}: {task_row!r}")
            if task not in task_to_new_index:
                new_index = len(merged_tasks)
                task_to_new_index[task] = new_index
                merged_tasks.append({"task_index": new_index, "task": task})
            source_map[int(task_row["task_index"])] = task_to_new_index[task]
        source_maps.append(source_map)
    return merged_tasks, source_maps


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise MergeError(f"Parquet table is missing required column {name!r}")
    return table.set_column(column_index, table.schema.field(column_index), values)


def rewrite_parquet(
    source_path: Path,
    destination_path: Path,
    new_episode_index: int,
    frame_offset: int,
    task_map: dict[int, int],
) -> tuple[int, np.ndarray]:
    try:
        table = pq.read_table(source_path)
    except Exception as exc:
        raise MergeError(f"Cannot read Parquet file {source_path}: {exc}") from exc
    row_count = table.num_rows

    episode_field = table.schema.field("episode_index")
    index_field = table.schema.field("index")
    task_field = table.schema.field("task_index")
    old_task_values = table["task_index"].combine_chunks().to_numpy(zero_copy_only=False)
    try:
        new_task_values_np = np.fromiter(
            (task_map[int(value)] for value in old_task_values),
            dtype=np.int64,
            count=row_count,
        )
    except KeyError as exc:
        raise MergeError(f"Unknown task_index {exc.args[0]} in {source_path}") from exc

    table = replace_column(
        table,
        "episode_index",
        pa.array(np.full(row_count, new_episode_index, dtype=np.int64), type=episode_field.type),
    )
    table = replace_column(
        table,
        "index",
        pa.array(np.arange(frame_offset, frame_offset + row_count, dtype=np.int64), type=index_field.type),
    )
    table = replace_column(
        table,
        "task_index",
        pa.array(new_task_values_np, type=task_field.type),
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, destination_path, compression="zstd")
    except Exception as exc:
        raise MergeError(f"Cannot write Parquet file {destination_path}: {exc}") from exc
    return row_count, new_task_values_np


def set_constant_stats(stats: dict[str, Any], key: str, value: int, count: int) -> None:
    if key not in stats:
        return
    stats[key]["min"] = [value]
    stats[key]["max"] = [value]
    stats[key]["mean"] = [float(value)]
    stats[key]["std"] = [0.0]
    stats[key]["count"] = [count]


def shift_stats(stats: dict[str, Any], key: str, offset: int, count: int) -> None:
    if key not in stats:
        return
    for statistic in ("min", "max", "mean"):
        if statistic in stats[key]:
            stats[key][statistic] = [value + offset for value in stats[key][statistic]]
    stats[key]["count"] = [count]


def set_task_stats(stats: dict[str, Any], values: np.ndarray) -> None:
    if "task_index" not in stats or len(values) == 0:
        return
    values64 = values.astype(np.float64, copy=False)
    stats["task_index"] = {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values64.mean())],
        "std": [float(values64.std())],
        "count": [int(len(values))],
    }


def copy_video(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link_mode == "hardlink":
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
    except OSError as exc:
        raise MergeError(f"Cannot {link_mode} video {source} -> {destination}: {exc}") from exc


def merge_datasets(
    sources: list[SourceDataset], output: Path, link_mode: str, *, dry_run: bool
) -> None:
    ensure_compatible(sources)
    merged_tasks, task_maps = build_task_maps(sources)
    total_episodes = sum(len(source.episodes) for source in sources)
    total_frames = sum(int(source.info["total_frames"]) for source in sources)
    video_keys = sources[0].video_keys
    chunks_size = int(sources[0].info["chunks_size"])

    print(f"Sources: {len(sources)}")
    for source in sources:
        print(
            f"  {source.root}: {len(source.episodes)} episodes, "
            f"{source.info['total_frames']} frames"
        )
    print(
        f"Plan: {total_episodes} episodes, {total_frames} frames, "
        f"{len(merged_tasks)} tasks, {total_episodes * len(video_keys)} videos"
    )
    print(f"Output: {output}")
    if dry_run:
        print("Dry run complete; no files were written.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    new_episode_index = 0
    frame_offset = 0
    try:
        for source_number, source in enumerate(sources):
            task_map = task_maps[source_number]
            for old_episode in source.episodes:
                old_episode_index = int(old_episode["episode_index"])
                source_data_rel = format_episode_path(
                    source.info["data_path"], old_episode_index, chunks_size
                )
                destination_data_rel = format_episode_path(
                    source.info["data_path"], new_episode_index, chunks_size
                )
                row_count, new_task_values = rewrite_parquet(
                    source.root / source_data_rel,
                    temporary / destination_data_rel,
                    new_episode_index,
                    frame_offset,
                    task_map,
                )
                expected_length = int(old_episode["length"])
                if row_count != expected_length:
                    raise MergeError(
                        f"Row count changed while reading {source.root / source_data_rel}: "
                        f"expected {expected_length}, got {row_count}"
                    )

                for video_key in video_keys:
                    source_video_rel = format_episode_path(
                        source.info["video_path"],
                        old_episode_index,
                        chunks_size,
                        video_key=video_key,
                    )
                    destination_video_rel = format_episode_path(
                        source.info["video_path"],
                        new_episode_index,
                        chunks_size,
                        video_key=video_key,
                    )
                    copy_video(
                        source.root / source_video_rel,
                        temporary / destination_video_rel,
                        link_mode,
                    )

                episode_row = copy.deepcopy(old_episode)
                episode_row["episode_index"] = new_episode_index
                merged_episodes.append(episode_row)

                stats_row = copy.deepcopy(source.episode_stats[old_episode_index])
                stats_row["episode_index"] = new_episode_index
                stats = stats_row.get("stats")
                if not isinstance(stats, dict):
                    raise MergeError(
                        f"Invalid stats for episode {old_episode_index} in {source.root}"
                    )
                set_constant_stats(stats, "episode_index", new_episode_index, row_count)
                shift_stats(stats, "index", frame_offset, row_count)
                set_task_stats(stats, new_task_values)
                merged_stats.append(stats_row)

                frame_offset += row_count
                new_episode_index += 1
                print(
                    f"[{new_episode_index}/{total_episodes}] {source.root.name} "
                    f"episode {old_episode_index} -> {new_episode_index - 1}"
                )

        merged_info = copy.deepcopy(sources[0].info)
        merged_info.update(
            {
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "total_tasks": len(merged_tasks),
                "total_videos": total_episodes * len(video_keys),
                "total_chunks": math.ceil(total_episodes / chunks_size),
                "splits": {"train": f"0:{total_episodes}"},
            }
        )
        write_json(temporary / "meta" / "info.json", merged_info)
        write_jsonl(temporary / "meta" / "tasks.jsonl", merged_tasks)
        write_jsonl(temporary / "meta" / "episodes.jsonl", merged_episodes)
        write_jsonl(temporary / "meta" / "episodes_stats.jsonl", merged_stats)

        generated = load_source(temporary, check_files=True)
        if generated.info["total_frames"] != frame_offset:
            raise MergeError(
                f"Generated frame count mismatch: metadata={generated.info['total_frames']}, "
                f"written={frame_offset}"
            )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Merge complete: {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge compatible LeRobot v2.1 datasets without modifying the sources."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Source dataset directories, in the desired episode order",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="New dataset directory")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place videos in the output (default: copy; hardlink saves disk space)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the merge plan without writing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    if len(set(input_paths)) != len(input_paths):
        raise MergeError("The same source dataset was provided more than once")
    if output in input_paths:
        raise MergeError("Output directory cannot also be an input dataset")
    if output.exists():
        raise MergeError(f"Output already exists; refusing to overwrite it: {output}")
    sources = [load_source(path, check_files=True) for path in input_paths]
    merge_datasets(sources, output, args.link_mode, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
