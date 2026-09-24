from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


def _color_for(name: str) -> Tuple[int, int, int]:
    """Deterministic BGR color per class name."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _find_pil_font(size: int):
    from PIL import ImageFont

    for c in ("simsun.ttc", "msyh.ttc", "simhei.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


@dataclass
class OverlayStats:
    """Lightweight HUD state."""
    total: int = 0
    in_frame: int = 0
    cracked: int = 0
    intact: int = 0
    dirty: int = 0
    defective_count: int = 0
    total_crossings: int = 0


class CombinedOverlay:
    """Draws the counter line, counter bboxes, crack masks, and a unified HUD."""

    def __init__(
        self,
        line_start: Tuple[int, int] = (100, 240),
        line_end: Tuple[int, int] = (540, 240),
        line_direction: str = "down",
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
        show_mask: bool = True,
        show_box: bool = True,
    ) -> None:
        self.line_start = tuple(line_start)
        self.line_end = tuple(line_end)
        self.line_direction = line_direction.lower()
        self.mask_alpha = float(mask_alpha)
        self.box_thickness = int(box_thickness)
        self.show_mask = bool(show_mask)
        self.show_box = bool(show_box)

    def draw(
        self,
        frame: np.ndarray,
        counter_tracks: Iterable,
        crack_findings: Iterable,
        stats: OverlayStats,
        show_window: bool = True,
    ) -> np.ndarray:
        canvas = frame

        if self.show_mask:
            canvas = self._overlay_masks(canvas, crack_findings)

        if show_window:
            canvas = self._draw_counter(canvas, counter_tracks, stats)
            canvas = self._draw_crack_bboxes(canvas, crack_findings)
            canvas = self._draw_hud(canvas, stats)

        return canvas

    def _overlay_masks(self, frame: np.ndarray, findings: Iterable) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for f in findings:
            mask = getattr(f, "mask", None)
            if mask is None or mask.size == 0 or not mask.any():
                continue
            if mask.shape != (h, w):
                m = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                m = mask
            color = np.array(_color_for(getattr(f, "name", "egg")), dtype=np.uint8)
            overlay[m] = (self.mask_alpha * color + (1 - self.mask_alpha) * overlay[m]).astype(np.uint8)
        return overlay

    def _draw_counter(self, canvas: np.ndarray, tracks: Iterable, stats: OverlayStats) -> np.ndarray:
        p1 = (int(self.line_start[0]), int(self.line_start[1]))
        p2 = (int(self.line_end[0]), int(self.line_end[1]))
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"Line: {self.line_direction.upper()}",
            (p1[0], max(0, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for tr in tracks:
            boxes = getattr(tr, "boxes", None)
            if not boxes:
                continue
            x1, y1, x2, y2 = [int(v) for v in boxes[-1]]
            crossed = getattr(tr, "crossed", False)
            color = (0, 255, 0) if crossed else (255, 128, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)
            cv2.putText(
                canvas,
                f"id={tr.track_id}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_crack_bboxes(self, canvas: np.ndarray, findings: Iterable) -> np.ndarray:
        h = canvas.shape[0]
        for f in findings:
            x1, y1, x2, y2 = [int(v) for v in (f.x_min, f.y_min, f.x_max, f.y_max)]
            color = _color_for(f.name)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            label_y = min(h - 4, y2 + 14)
            cv2.putText(
                canvas,
                f"{f.name} {f.conf:.2f}",
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_hud(self, canvas: np.ndarray, stats: OverlayStats) -> np.ndarray:
        lines = [
            (f"Crossings: {stats.total_crossings}", (0, 255, 0)),
            (f"In-frame eggs: {stats.in_frame}", (255, 255, 255)),
            (
                f"Cracked: {stats.cracked}  Intact: {stats.intact}  Dirty: {stats.dirty}",
                (255, 255, 255),
            ),
            (
                f"Defective (track history): {stats.defective_count}",
                (0, 165, 255) if stats.defective_count else (255, 255, 255),
            ),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (12, 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return canvas