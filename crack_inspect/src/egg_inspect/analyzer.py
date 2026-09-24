from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import CRACKED_SUFFIX, DIRTY_SUFFIX, INTACT_NAMES
from .detector import CrackFinding


def is_crack(name: str) -> bool:
    return name.endswith(CRACKED_SUFFIX)


def is_dirty(name: str) -> bool:
    return name.endswith(DIRTY_SUFFIX)


def is_intact(name: str) -> bool:
    return name in INTACT_NAMES


@dataclass
class FrameStats:
    """Per-frame roll-up of the crack / quality inspection."""
    frame_idx: int = 0
    timestamp: float = 0.0
    total: int = 0
    cracked: int = 0
    intact: int = 0
    dirty: int = 0
    other: int = 0
    cracked_ratio: float = 0.0
    avg_crack_area_ratio: float = 0.0  # mean of mask_area_ratio over cracked eggs
    max_crack_area_ratio: float = 0.0
    by_class: dict = field(default_factory=dict)
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "total": self.total,
            "cracked": self.cracked,
            "intact": self.intact,
            "dirty": self.dirty,
            "other": self.other,
            "cracked_ratio": round(self.cracked_ratio, 4),
            "avg_crack_area_ratio": round(self.avg_crack_area_ratio, 4),
            "max_crack_area_ratio": round(self.max_crack_area_ratio, 4),
        }


def analyze_frame(
    findings: List[CrackFinding],
    frame_idx: int = 0,
    timestamp: float = 0.0,
    min_mask_pixels: int = 0,
    min_bbox_area: int = 0,
    crack_only: bool = False,
) -> FrameStats:
    """Compute per-frame stats from raw findings.

    Filters out very small masks/bboxes and (optionally) non-crack classes
    before counting.
    """
    cracked_ratios: List[float] = []
    by_class: dict = {}
    cracked = intact = dirty = other = 0
    filtered_total = 0

    for f in findings:
        if f.mask_area < min_mask_pixels:
            continue
        if f.bbox_area < min_bbox_area:
            continue
        if crack_only and not is_crack(f.name):
            continue
        filtered_total += 1
        by_class[f.name] = by_class.get(f.name, 0) + 1
        if is_crack(f.name):
            cracked += 1
            cracked_ratios.append(f.mask_area_ratio)
        elif is_intact(f.name):
            intact += 1
        elif is_dirty(f.name):
            dirty += 1
        else:
            other += 1

    ratio = (cracked / filtered_total) if filtered_total > 0 else 0.0
    avg_crack_ratio = sum(cracked_ratios) / len(cracked_ratios) if cracked_ratios else 0.0
    max_crack_ratio = max(cracked_ratios) if cracked_ratios else 0.0

    return FrameStats(
        frame_idx=frame_idx,
        timestamp=timestamp,
        total=filtered_total,
        cracked=cracked,
        intact=intact,
        dirty=dirty,
        other=other,
        cracked_ratio=ratio,
        avg_crack_area_ratio=avg_crack_ratio,
        max_crack_area_ratio=max_crack_ratio,
        by_class=by_class,
    )


def summarize(stats: List[FrameStats]) -> dict:
    """Aggregate per-frame stats into a final report dictionary."""
    if not stats:
        return {
            "frames": 0,
            "total_eggs": 0,
            "total_cracked": 0,
            "total_intact": 0,
            "total_dirty": 0,
            "total_other": 0,
            "overall_crack_rate": 0.0,
            "avg_crack_area_ratio": 0.0,
            "max_crack_area_ratio": 0.0,
            "by_class": {},
        }
    total = sum(s.total for s in stats)
    cracked = sum(s.cracked for s in stats)
    intact = sum(s.intact for s in stats)
    dirty = sum(s.dirty for s in stats)
    other = sum(s.other for s in stats)
    rate = (cracked / total) if total > 0 else 0.0
    cracked_ratios = [r for s in stats for r in [s.avg_crack_area_ratio] if r > 0]
    avg_crack = sum(cracked_ratios) / len(cracked_ratios) if cracked_ratios else 0.0
    max_crack = max((s.max_crack_area_ratio for s in stats), default=0.0)
    by_class: dict = {}
    for s in stats:
        for k, v in s.by_class.items():
            by_class[k] = by_class.get(k, 0) + v
    return {
        "frames": len(stats),
        "total_eggs": total,
        "total_cracked": cracked,
        "total_intact": intact,
        "total_dirty": dirty,
        "total_other": other,
        "overall_crack_rate": round(rate, 4),
        "avg_crack_area_ratio": round(avg_crack, 4),
        "max_crack_area_ratio": round(max_crack, 4),
        "by_class": by_class,
    }