from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from egg_inspect.analyzer import is_crack, is_dirty, is_intact  # type: ignore  # noqa: E402


class VerdictSource(str, Enum):
    INSPECTOR = "inspector"
    UNKNOWN = "unknown"


@dataclass
class EggVerdict:
    """Per-egg quality verdict at the line-crossing moment."""

    source: VerdictSource = VerdictSource.UNKNOWN
    color: str = "unknown"           # white | brown | unknown
    cls_name: str = "unknown"
    cracked: bool = False
    dirty: bool = False
    intact: bool = False
    is_defective: bool = False       # cracked or dirty
    mask_area_ratio: float = 0.0
    conf: float = 0.0
    iou: float = 0.0                 # IoU between track bbox and finding bbox

    def as_row(self) -> dict:
        return {
            "color": self.color,
            "cls": self.cls_name,
            "cracked": self.cracked,
            "dirty": self.dirty,
            "intact": self.intact,
            "is_defective": self.is_defective,
            "mask_area_ratio": round(self.mask_area_ratio, 4),
            "conf": round(self.conf, 4),
            "iou": round(self.iou, 4),
            "source": self.source.value,
        }


def _color_from_name(name: str) -> str:
    if name.startswith("white"):
        return "white"
    if name.startswith("brown") or name.startswith("brow"):
        return "brown"
    return "unknown"


def _verdict_from_finding(finding, iou: float) -> EggVerdict:
    cracked = is_crack(finding.name)
    dirty = is_dirty(finding.name)
    intact = is_intact(finding.name)
    return EggVerdict(
        source=VerdictSource.INSPECTOR,
        color=_color_from_name(finding.name),
        cls_name=finding.name,
        cracked=cracked,
        dirty=dirty,
        intact=intact,
        is_defective=cracked or dirty,
        mask_area_ratio=float(getattr(finding, "mask_area_ratio", 0.0)),
        conf=float(getattr(finding, "conf", 0.0)),
        iou=float(iou),
    )


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate(
    track_bbox,
    crack_findings: List,
    min_iou: float = 0.3,
) -> EggVerdict:
    """Pick the crack finding that best overlaps a tracked egg."""
    if not crack_findings:
        return EggVerdict()
    best_iou = -1.0
    best = None
    for f in crack_findings:
        b = (f.x_min, f.y_min, f.x_max, f.y_max)
        iou = _iou_xyxy(track_bbox, b)
        if iou > best_iou:
            best_iou = iou
            best = f
    if best is None or best_iou < min_iou:
        return EggVerdict()
    return _verdict_from_finding(best, best_iou)