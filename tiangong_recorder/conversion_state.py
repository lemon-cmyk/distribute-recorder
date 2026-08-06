from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConversionRecord:
    source_path: str
    source_sha256: str
    status: str
    frame_count: int | None
    target_episode_index: int | None
    episode_index: int | None
    dataset_root: str | None
    error_message: str | None
    source_deleted: bool


class ConversionState:
    """Durable conversion state used to make source deletion crash-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversions (
                source_path TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                frame_count INTEGER,
                target_episode_index INTEGER,
                episode_index INTEGER,
                dataset_root TEXT,
                error_message TEXT,
                source_deleted INTEGER NOT NULL DEFAULT 0,
                updated_at_ns INTEGER NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row | None) -> ConversionRecord | None:
        if row is None:
            return None
        return ConversionRecord(
            source_path=row["source_path"],
            source_sha256=row["source_sha256"],
            status=row["status"],
            frame_count=row["frame_count"],
            target_episode_index=row["target_episode_index"],
            episode_index=row["episode_index"],
            dataset_root=row["dataset_root"],
            error_message=row["error_message"],
            source_deleted=bool(row["source_deleted"]),
        )

    def get(self, source_path: str | Path) -> ConversionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM conversions WHERE source_path = ?", (str(source_path),)
        ).fetchone()
        return self._record(row)

    def begin(
        self,
        source_path: str | Path,
        source_sha256: str,
        frame_count: int,
        target_episode_index: int,
        dataset_root: str | Path,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO conversions (
                source_path, source_sha256, status, frame_count,
                target_episode_index, episode_index, dataset_root,
                error_message, source_deleted, updated_at_ns
            ) VALUES (?, ?, 'converting', ?, ?, NULL, ?, NULL, 0, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_sha256 = excluded.source_sha256,
                status = 'converting',
                frame_count = excluded.frame_count,
                target_episode_index = excluded.target_episode_index,
                episode_index = NULL,
                dataset_root = excluded.dataset_root,
                error_message = NULL,
                source_deleted = 0,
                updated_at_ns = excluded.updated_at_ns
            """,
            (
                str(source_path),
                source_sha256,
                int(frame_count),
                int(target_episode_index),
                str(dataset_root),
                time.time_ns(),
            ),
        )
        self.connection.commit()

    def mark_done(
        self, source_path: str | Path, episode_index: int, frame_count: int
    ) -> None:
        self.connection.execute(
            """
            UPDATE conversions
            SET status = 'done', episode_index = ?, frame_count = ?,
                error_message = NULL, updated_at_ns = ?
            WHERE source_path = ?
            """,
            (int(episode_index), int(frame_count), time.time_ns(), str(source_path)),
        )
        self.connection.commit()

    def mark_deleted(self, source_path: str | Path) -> None:
        self.connection.execute(
            """
            UPDATE conversions
            SET source_deleted = 1, updated_at_ns = ?
            WHERE source_path = ?
            """,
            (time.time_ns(), str(source_path)),
        )
        self.connection.commit()

    def mark_error(
        self,
        source_path: str | Path,
        source_sha256: str,
        error_message: str,
        *,
        release_reservation: bool = False,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO conversions (
                source_path, source_sha256, status, error_message,
                source_deleted, updated_at_ns
            ) VALUES (?, ?, 'error', ?, 0, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_sha256 = excluded.source_sha256,
                status = 'error',
                error_message = excluded.error_message,
                source_deleted = 0,
                target_episode_index = CASE
                    WHEN ? THEN NULL ELSE target_episode_index
                END,
                episode_index = CASE WHEN ? THEN NULL ELSE episode_index END,
                updated_at_ns = excluded.updated_at_ns
            """,
            (
                str(source_path),
                source_sha256,
                error_message,
                time.time_ns(),
                int(release_reservation),
                int(release_reservation),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
