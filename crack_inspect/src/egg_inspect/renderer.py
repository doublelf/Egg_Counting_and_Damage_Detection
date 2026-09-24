from __future__ import annotations

import hashlib
from typing import Iterable

import cv2
import numpy as np

from .analyzer import FrameStats
from .config import VisualConfig
from .detector import CrackFinding


def _color_for(name: str) -> tuple:
    """Deterministic BGR color per class name."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _find_pil_font(size: int):
    from PIL import ImageFont

    candidates = [
        "simsun.ttc",
        "msyh.ttc",
        "simhei.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


class Renderer:
    """Draws masks + boxes + labels + a top-of-frame HUD."""

    def __init__(self, visual: VisualConfig):
        self.visual = visual

    def draw(
        self,
        frame: np.ndarray,
        findings: Iterable[CrackFinding],
        stats: FrameStats | None = None,
    ) -> np.ndarray:
        canvas = frame
        vis = self.visual

        if vis.show_mask:
            canvas = self._overlay_masks(canvas, findings, vis.mask_alpha)

        if vis.show_box:
            for f in findings:
                color = _color_for(f.name)
                p1 = (int(f.x_min), int(f.y_min))
                p2 = (int(f.x_max), int(f.y_max))
                cv2.rectangle(canvas, p1, p2, color, vis.box_thickness)
                label = f"{f.name} {f.conf:.2f} mask={f.mask_area_ratio:.2f}"
                self._label(canvas, label, p1, color, vis.label_font_scale)

        if stats is not None:
            self._hud(canvas, stats, vis)

        return canvas

    def _overlay_masks(
        self,
        frame: np.ndarray,
        findings: Iterable[CrackFinding],
        alpha: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for f in findings:
            if f.mask is None or f.mask.size == 0 or not f.mask.any():
                continue
            if f.mask.shape != (h, w):
                m = cv2.resize(
                    f.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                m = f.mask
            color = np.array(_color_for(f.name), dtype=np.uint8)
            overlay[m] = (alpha * color + (1 - alpha) * overlay[m]).astype(np.uint8)
        return overlay

    def _label(self, canvas, text, p1, color, font_scale) -> None:
        x, y = p1
        try:
            font = _find_pil_font(int(round(14 * font_scale)))
            from PIL import ImageDraw, ImageFont  # noqa: F401

            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil = __import__("PIL").Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil)
            # White shadow + color text for legibility
            draw.text((x + 1, max(0, y - 16 + 1)), text, fill=(255, 255, 255), font=font)
            draw.text((x, max(0, y - 16)), text, fill=color[::-1], font=font)
            canvas[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(
                canvas,
                text,
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    def _hud(self, canvas, stats: FrameStats, vis: VisualConfig) -> None:
        hud_lines = [
            f"Eggs: {stats.total}",
            f"Cracked: {stats.cracked} ({stats.cracked_ratio * 100:.1f}%)",
            f"Intact: {stats.intact}   Dirty: {stats.dirty}",
        ]
        y0 = 24
        for i, line in enumerate(hud_lines):
            cv2.putText(
                canvas,
                line,
                (12, y0 + i * int(22 * vis.hud_font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                vis.hud_font_scale,
                (0, 255, 0) if i == 0 else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )