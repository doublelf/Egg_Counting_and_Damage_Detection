"""Test the classical CV crack detector on real egg images.

Runs :class:`ClassicalCrackDetector` (vendored YOLOv5 + OpenCV) on every
PNG in ``../crackedChickenEggs161/`` and exports annotated debug images to
``output/classical_debug/``.

The demo images are not labelled, so this test is qualitative: it checks
that the pipeline runs end-to-end and prints per-image stats. Visual
inspection of the debug PNGs is the real acceptance test.

Run::

    python tests/test_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ED = ROOT.parent

# Make sibling packages importable BEFORE importing egg_inspect / egg_common
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ED / "egg_line" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from egg_common.paths import setup_sibling_paths  # noqa: E402

setup_sibling_paths()

from egg_inspect import (  # noqa: E402
    ClassicalConfig,
    ClassicalCrackDetector,
    Renderer,
    analyze_frame,
)
from egg_inspect.config import VisualConfig  # noqa: E402


def main() -> int:
    weights = (ED / "Automated-Egg-Counting-System" / "best.pt")
    if not weights.exists():
        print(f"[classical-test] FAIL: counter weights missing at {weights}")
        return 1

    demo_dir = ED / "crackedChickenEggs161"
    if not demo_dir.exists():
        print(f"[classical-test] FAIL: demo dir missing at {demo_dir}")
        return 1

    out_dir = ROOT / "output" / "classical_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[classical-test] writing annotated images to {out_dir}")

    detector = ClassicalCrackDetector(
        weights=str(weights),
        classical_config=ClassicalConfig(),
        device="cpu",
        img_size=640,
        conf_threshold=0.35,
        iou_threshold=0.45,
        project_root=weights.parent,
    )

    renderer = Renderer(VisualConfig())
    images = sorted(p for p in demo_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"[classical-test] no images in {demo_dir}")
        return 1
    print(f"[classical-test] {len(images)} demo images")

    summary = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        findings = detector.detect(frame)
        stats = analyze_frame(findings)
        canvas = renderer.draw(frame, findings, stats)
        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), canvas)
        summary.append((img_path.name, len(findings), stats.cracked, stats.intact, stats.dirty))
        print(
            f"[classical-test] {img_path.name}: "
            f"eggs={len(findings)} cracked={stats.cracked} "
            f"intact={stats.intact} dirty={stats.dirty}"
        )

    if not summary:
        print("[classical-test] FAIL: no detections produced")
        return 1

    n_imgs = len(summary)
    n_with_eggs = sum(1 for _, n, *_ in summary if n > 0)
    total_eggs = sum(n for _, n, *_ in summary)
    total_cracked = sum(c for _, _, c, _, _ in summary)
    total_intact = sum(i for _, _, _, i, _ in summary)
    total_dirty = sum(d for _, _, _, _, d in summary)
    print(
        f"[classical-test] images={n_imgs} images_with_eggs={n_with_eggs} "
        f"eggs={total_eggs} cracked={total_cracked} intact={total_intact} dirty={total_dirty}"
    )

    # If we never see any eggs, the pipeline is misconfigured.
    if n_with_eggs == 0:
        print("[classical-test] WARN: no eggs detected across all images")
        print("[classical-test] the vendored YOLOv5 may be missing or returning empty")

    print("[classical-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())