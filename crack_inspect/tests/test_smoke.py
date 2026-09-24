"""Smoke test for the Egg Crack / Quality Inspection System.

Validates analyzer + renderer + report without requiring a YOLOv8 model.

    python tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from egg_inspect import (  # noqa: E402
    FrameStats,
    Renderer,
    analyze_frame,
    is_crack,
    is_dirty,
    is_intact,
    summarize,
)
from egg_inspect.detector import CrackFinding  # noqa: E402
from egg_inspect.config import VisualConfig  # noqa: E402


def make_finding(name: str, x: int, y: int, w: int = 60, h: int = 80, mask_ratio: float = 0.3) -> CrackFinding:
    mask = np.zeros((480, 640), dtype=bool)
    mask[y:y + h, x:x + w] = True
    total_pixels = w * h
    crack_pixels = int(total_pixels * mask_ratio)
    coords = np.argwhere(mask)[:crack_pixels]
    mask[:] = False
    if len(coords) > 0:
        mask[coords[:, 0], coords[:, 1]] = True
    cls_map = {"brown-egg": 1, "white-egg": 3, "brown-egg-crack": 2, "white-egg-crack": 4}
    return CrackFinding(
        cls_id=cls_map.get(name, 0),
        name=name,
        conf=0.9,
        x_min=x,
        y_min=y,
        x_max=x + w,
        y_max=y + h,
        mask=mask,
        bbox_area=total_pixels,
        mask_area=crack_pixels,
        mask_area_ratio=mask_ratio,
    )


def test_class_helpers() -> None:
    assert is_crack("brown-egg-crack")
    assert is_crack("white-egg-crack")
    assert not is_crack("brown-egg")
    assert is_intact("brown-egg")
    assert is_intact("white-egg")
    assert is_dirty("brow-egg-dirty")
    assert is_dirty("white-egg-dirty")
    print("[smoke] class helpers OK")


def test_analyzer() -> None:
    findings = [
        make_finding("brown-egg", 50, 100, mask_ratio=0.0),
        make_finding("brown-egg-crack", 200, 100, mask_ratio=0.4),
        make_finding("white-egg-crack", 350, 100, mask_ratio=0.6),
        make_finding("white-egg-dirty", 500, 100, mask_ratio=0.05),
    ]
    stats = analyze_frame(findings, frame_idx=1, timestamp=0.1)
    assert stats.total == 4
    assert stats.cracked == 2
    assert stats.intact == 1
    assert stats.dirty == 1
    assert abs(stats.cracked_ratio - 0.5) < 1e-9
    assert stats.avg_crack_area_ratio == (0.4 + 0.6) / 2
    assert stats.max_crack_area_ratio == 0.6
    print(f"[smoke] analyzer OK: cracked={stats.cracked}/{stats.total} ratio={stats.cracked_ratio:.2%}")

    cracked_only = analyze_frame(findings, crack_only=True)
    assert cracked_only.total == 2 and cracked_only.cracked == 2
    print(f"[smoke] crack_only filter OK: kept {cracked_only.total} findings")

    filtered = analyze_frame(
        findings,
        min_mask_pixels=5000,
        min_bbox_area=0,
    )
    # brown-egg mask is 0 pixels -> filtered out; dirty has only 0.05 ratio -> small
    assert filtered.total < stats.total
    print(f"[smoke] mask filter OK: kept {filtered.total} findings")


def test_summarize() -> None:
    s1 = analyze_frame([make_finding("brown-egg-crack", 100, 100, mask_ratio=0.5)], frame_idx=1)
    s2 = analyze_frame(
        [
            make_finding("brown-egg", 50, 100, mask_ratio=0.0),
            make_finding("white-egg-crack", 300, 100, mask_ratio=0.4),
        ],
        frame_idx=2,
    )
    out = summarize([s1, s2])
    assert out["frames"] == 2
    assert out["total_eggs"] == 3
    assert out["total_cracked"] == 2
    assert abs(out["overall_crack_rate"] - 2 / 3) < 1e-3
    assert out["by_class"]["brown-egg-crack"] == 1
    print(f"[smoke] summarize OK: {json.dumps(out)}")


def test_renderer() -> None:
    findings = [
        make_finding("brown-egg", 50, 100),
        make_finding("brown-egg-crack", 200, 100, mask_ratio=0.5),
        make_finding("white-egg-crack", 350, 100, mask_ratio=0.3),
    ]
    stats = analyze_frame(findings, frame_idx=10, timestamp=0.5)
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    out = Renderer(VisualConfig()).draw(frame, findings, stats)
    assert out.shape == frame.shape
    assert (out != frame).any(), "Renderer should modify the canvas"
    print("[smoke] renderer OK")


def main() -> int:
    test_class_helpers()
    test_analyzer()
    test_summarize()
    test_renderer()
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())