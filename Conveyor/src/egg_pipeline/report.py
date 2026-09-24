from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .associator import EggVerdict, VerdictSource


CROSSING_FIELDS = [
    "timestamp_iso",
    "frame_idx",
    "timestamp",
    "track_id",
    "direction",
    "color",
    "cls",
    "cracked",
    "dirty",
    "intact",
    "is_defective",
    "mask_area_ratio",
    "conf",
    "iou",
    "source",
]


FRAME_FIELDS = [
    "timestamp_iso",
    "frame_idx",
    "timestamp",
    "in_frame_eggs",
    "cracked_in_frame",
    "intact_in_frame",
    "dirty_in_frame",
    "cracked_ratio_in_frame",
    "crossings_this_frame",
]


class FramesCsv:
    """Per-frame roll-up CSV."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FRAME_FIELDS)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def log(self, stats: dict) -> None:
        row = dict(stats)
        row["timestamp_iso"] = datetime.now().isoformat(timespec="seconds")
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class CrossingsCsv:
    """Per-egg verdict written when an egg crosses the line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CROSSING_FIELDS)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def log(self, frame_idx: int, timestamp: float, track_id: int, direction: str, verdict: EggVerdict) -> None:
        row = verdict.as_row()
        row["timestamp_iso"] = datetime.now().isoformat(timespec="seconds")
        row["frame_idx"] = frame_idx
        row["timestamp"] = round(timestamp, 3)
        row["track_id"] = track_id
        row["direction"] = direction
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class JsonSummary:
    """Single JSON file aggregating the run."""

    def __init__(self, path: Path, meta: Optional[dict] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta or {})
        self._crossings = 0
        self._defective = 0
        self._color_counts: dict = {}

    def record_crossing(self, verdict: EggVerdict) -> None:
        self._crossings += 1
        if verdict.source == VerdictSource.INSPECTOR:
            self._color_counts[verdict.color] = self._color_counts.get(verdict.color, 0) + 1
            if verdict.is_defective:
                self._defective += 1

    def close(self, extra: Optional[dict] = None) -> None:
        payload = {
            "meta": self._meta,
            "summary": {
                "total_crossings": self._crossings,
                "defective_count": self._defective,
                "defective_rate": (
                    round(self._defective / self._crossings, 4) if self._crossings else 0.0
                ),
                "color_split": self._color_counts,
            },
        }
        if extra:
            payload["meta"].update(extra)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)