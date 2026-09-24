from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .analyzer import FrameStats, summarize


class CsvReport:
    """Append-only CSV per-frame report."""

    FIELDNAMES = [
        "timestamp_iso",
        "frame_idx",
        "timestamp",
        "total",
        "cracked",
        "intact",
        "dirty",
        "other",
        "cracked_ratio",
        "avg_crack_area_ratio",
        "max_crack_area_ratio",
    ]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def log(self, stats: FrameStats) -> None:
        row = stats.as_row()
        row["timestamp_iso"] = datetime.now().isoformat(timespec="seconds")
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class JsonSummary:
    """Single-file JSON summary written on close."""

    def __init__(self, path: str, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta or {})
        self._stats: list = []

    def add(self, stats: FrameStats) -> None:
        self._stats.append(stats)

    def close(self, extra: dict | None = None) -> None:
        payload = {"meta": self._meta, "summary": summarize(self._stats)}
        if extra:
            payload["meta"].update(extra)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)