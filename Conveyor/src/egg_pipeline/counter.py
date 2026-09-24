from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import CounterConfig
from .paths import setup_sibling_paths

setup_sibling_paths()

from egg_counter import CentroidTracker, LineCrossingCounter, YoloDetector  # type: ignore  # noqa: E402


def build_counter(
    cfg: CounterConfig,
    weights_path: Path,
) -> Tuple[YoloDetector, CentroidTracker, LineCrossingCounter]:
    """Instantiate the vendored YOLOv5 counter + tracker + line counter."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Counter weights not found: {weights_path}\n"
            "Conveyor/ expects the vendored YOLOv5 best.pt at\n"
            "  ../Automated-Egg-Counting-System/best.pt\n"
            "Clone or symlink that project into the egg_detection/ tree."
        )
    det = YoloDetector(
        weights=str(weights_path),
        device=cfg.model.device,
        img_size=cfg.model.img_size,
        conf_threshold=cfg.model.conf_threshold,
        iou_threshold=cfg.model.iou_threshold,
        project_root=weights_path.parent,
    )
    det.set_conf_threshold(cfg.model.conf_threshold)
    det.set_iou_threshold(cfg.model.iou_threshold)

    tracker = CentroidTracker(
        max_lost_frames=cfg.tracker.max_lost_frames,
        iou_match_threshold=cfg.tracker.iou_match_threshold,
    )
    counter = LineCrossingCounter(
        line_start=tuple(cfg.line.start),
        line_end=tuple(cfg.line.end),
        direction=cfg.line.direction,
        min_track_length=cfg.line.min_track_length,
    )
    return det, tracker, counter