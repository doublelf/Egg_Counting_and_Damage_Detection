"""Classical-CV crack / dirty classifier and hybrid YOLOv5 + OpenCV detector.

This module lets the cracked-egg pipeline run **without a trained
YOLOv8-seg weights file**. The :class:`ClassicalCrackDetector`:

1. reuses the vendored YOLOv5 (in ``Automated-Egg-Counting-System/``) to
   find eggs as bounding boxes;
2. for every egg crop, applies a classical computer-vision pipeline
   (CLAHE → bilateral filter → black-hat morphology → adaptive
   threshold → connected components → shape-based classification) to
   decide whether the egg is *intact*, *cracked*, or *dirty*;
3. returns the same :class:`egg_inspect.CrackFinding` shape used by the
   rest of the application, so the analyzer / renderer / report
   pipeline is unchanged.

The module exposes a single configuration class,
:class:`ClassicalConfig`, with sensible defaults that can be tuned from
``config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DEFAULT_NAMES
from .detector import CrackFinding

# Make vendored yolov5 + sibling projects importable.
def _ensure_sibling_paths():
    here = Path(__file__).resolve()
    # crack_inspect/src/egg_inspect/classical.py -> egg_inspect/src/egg_inspect
    pkg_root = here.parents[1]   # .../egg_inspect/src
    project_root = pkg_root.parent  # .../egg_inspect
    egg_detection = project_root.parent  # .../egg_detection
    siblings = [
        str(egg_detection / "Automated-Egg-Counting-System" / "src"),
        str(egg_detection / "Automated-Egg-Counting-System" / "vendor_yolov5"),
        str(egg_detection / "egg_line" / "src"),
    ]
    import sys as _sys
    for p in siblings:
        if p not in _sys.path:
            _sys.path.insert(0, p)


CLASS_TO_ID: dict = {n: i for i, n in enumerate(DEFAULT_NAMES)}


@dataclass
class ClassicalConfig:
    """Tunable parameters for the classical crack classifier."""

    color_a_threshold: int = 135            # Lab a* mean; > this => brown
    clahe_clip: float = 2.0                 # CLAHE clip limit
    clahe_tile: int = 8                     # CLAHE tile grid size
    bilateral_d: int = 9                    # bilateral filter diameter
    bilateral_sigma: float = 75.0           # bilateral filter sigma
    blackhat_kernel: List[int] = field(default_factory=lambda: [15, 15])
    threshold_block_size: int = 51          # adaptive threshold block size (odd)
    threshold_c: int = -10                  # adaptive threshold constant
    cleanup_kernel: List[int] = field(default_factory=lambda: [3, 3])
    min_component_area: int = 12            # ignore tiny connected components
    crack_min_aspect: float = 3.0           # elongated structures
    crack_max_solidity: float = 0.5
    dirt_min_aspect: float = 0.7
    dirt_max_aspect: float = 1.5
    dirt_max_solidity: float = 0.7
    dirt_min_area_ratio: float = 0.01
    crack_area_threshold: float = 0.01       # crack_pixels / egg_pixels
    dirt_area_threshold: float = 0.02
    ellipse_inset: float = 0.45             # ROI mask is an ellipse of 0.45 * bbox


class ClassicalCrackClassifier:
    """Per-egg crop classifier using only OpenCV."""

    def __init__(self, config: Optional[ClassicalConfig] = None):
        self.config = config or ClassicalConfig()

    def classify(self, roi_bgr: np.ndarray, ellipse_mask: Optional[np.ndarray] = None) -> Tuple[str, np.ndarray, float]:
        """Return ``(cls_name, mask_bool, conf)`` for a single egg crop."""
        cfg = self.config
        if roi_bgr is None or roi_bgr.size == 0:
            return "brown-egg", np.zeros((0, 0), dtype=bool), 0.0

        # 1. Color: Lab a* channel mean
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        a_chan = lab[:, :, 1]
        if ellipse_mask is not None and ellipse_mask.any():
            mean_a = float(a_chan[ellipse_mask].mean())
        else:
            mean_a = float(a_chan.mean())
        is_brown = mean_a > cfg.color_a_threshold

        # 2. Preprocess L channel with CLAHE + bilateral
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
        l_eq = clahe.apply(lab[:, :, 0])
        l_filtered = cv2.bilateralFilter(l_eq, cfg.bilateral_d, cfg.bilateral_sigma, cfg.bilateral_sigma)

        # 3. Black-hat morphology: dark structures on light background
        bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(cfg.blackhat_kernel))
        blackhat = cv2.morphologyEx(l_filtered, cv2.MORPH_BLACKHAT, bh_kernel)

        # 4. Adaptive threshold
        bs = cfg.threshold_block_size
        if bs % 2 == 0:
            bs += 1
        binary = cv2.adaptiveThreshold(
            blackhat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            cfg.threshold_c,
        )

        # 5. Constrain to ellipse (exclude egg shell edge and background)
        if ellipse_mask is not None:
            binary = cv2.bitwise_and(binary, (ellipse_mask.astype(np.uint8) * 255))

        # 6. Morphological open + close cleanup
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(cfg.cleanup_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ck)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ck)

        # 7. Connected components + shape classification
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        crack_mask = np.zeros_like(binary, dtype=np.uint8)
        dirt_mask = np.zeros_like(binary, dtype=np.uint8)
        egg_area = int(ellipse_mask.sum()) if ellipse_mask is not None else binary.size
        egg_area = max(egg_area, 1)

        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < cfg.min_component_area:
                continue
            short = max(min(w, h), 1)
            long = max(w, h)
            aspect = long / short
            solidity = area / max(w * h, 1)
            if aspect >= cfg.crack_min_aspect and solidity <= cfg.crack_max_solidity:
                crack_mask[labels == i] = 255
            elif (
                cfg.dirt_min_aspect <= aspect <= cfg.dirt_max_aspect
                and solidity <= cfg.dirt_max_solidity
                and (area / egg_area) >= cfg.dirt_min_area_ratio
            ):
                dirt_mask[labels == i] = 255

        # 8. Decide class
        crack_ratio = float(crack_mask.sum()) / egg_area
        dirt_ratio = float(dirt_mask.sum()) / egg_area
        if crack_ratio > cfg.crack_area_threshold:
            cls = "brown-egg-crack" if is_brown else "white-egg-crack"
            mask = crack_mask > 0
            conf = float(min(1.0, 0.5 + crack_ratio * 10.0))
        elif dirt_ratio > cfg.dirt_area_threshold:
            cls = "brown-egg-dirty" if is_brown else "white-egg-dirty"
            mask = dirt_mask > 0
            conf = float(min(1.0, 0.5 + dirt_ratio * 5.0))
        else:
            cls = "brown-egg" if is_brown else "white-egg"
            mask = np.zeros_like(binary, dtype=bool)
            conf = float(1.0 - min(0.99, crack_ratio + dirt_ratio))
        return cls, mask, conf


class ClassicalCrackDetector:
    """Hybrid YOLOv5 + OpenCV detector; returns :class:`CrackFinding` list."""

    def __init__(
        self,
        weights: str,
        classical_config: Optional[ClassicalConfig] = None,
        device: str = "auto",
        img_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.weights = weights
        self._classifier = ClassicalCrackClassifier(classical_config)
        self._yolo = None
        self._yolo_error: Optional[str] = None
        self.device = device
        self.img_size = int(img_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.classes = classes

        weights_path = Path(weights)
        if project_root is not None and not weights_path.is_absolute():
            weights_path = Path(project_root) / weights_path
        if not weights_path.exists():
            self._yolo_error = f"weights not found at {weights_path}"
            return

        _ensure_sibling_paths()
        try:
            from egg_counter import YoloDetector  # type: ignore

            yolo_inst = YoloDetector(
                weights=str(weights_path),
                device=device,
                img_size=self.img_size,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                classes=self.classes,
                project_root=weights_path.parent,
            )
            self._yolo = yolo_inst
        except Exception as exc:
            self._yolo_error = f"{type(exc).__name__}: {exc}"

    @property
    def egg_detector_ready(self) -> bool:
        return self._yolo is not None

    def set_conf_threshold(self, value: float) -> None:
        self.conf_threshold = float(value)
        if self._yolo is not None:
            try:
                self._yolo.set_conf_threshold(value)
            except Exception:
                pass

    def set_iou_threshold(self, value: float) -> None:
        self.iou_threshold = float(value)
        if self._yolo is not None:
            try:
                self._yolo.set_iou_threshold(value)
            except Exception:
                pass

    @property
    def class_names(self) -> dict:
        return dict(CLASS_TO_ID)

    def detect(self, frame: np.ndarray) -> List[CrackFinding]:
        if frame is None or frame.size == 0:
            return []
        if self._yolo is None:
            return []
        try:
            detections = self._yolo.detect(frame)
        except Exception:
            return []

        findings: List[CrackFinding] = []
        h, w = frame.shape[:2]
        cfg = self._classifier.config

        for d in detections:
            x1 = max(0, int(d.x_min))
            y1 = max(0, int(d.y_min))
            x2 = min(w, int(d.x_max))
            y2 = min(h, int(d.y_max))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            ellipse_mask = self._make_ellipse_mask(roi.shape, cfg.ellipse_inset)
            cls_name, mask_local, conf = self._classifier.classify(roi, ellipse_mask)
            cls_id = CLASS_TO_ID.get(cls_name, 0)
            mask_full = np.zeros((h, w), dtype=bool)
            mask_full[y1:y2, x1:x2] = mask_local
            bbox_area = (x2 - x1) * (y2 - y1)
            mask_area = int(mask_full.sum())
            findings.append(
                CrackFinding(
                    cls_id=cls_id,
                    name=cls_name,
                    conf=float(conf),
                    x_min=float(x1),
                    y_min=float(y1),
                    x_max=float(x2),
                    y_max=float(y2),
                    mask=mask_full,
                    bbox_area=bbox_area,
                    mask_area=mask_area,
                    mask_area_ratio=(mask_area / bbox_area) if bbox_area > 0 else 0.0,
                )
            )
        return findings

    @staticmethod
    def _make_ellipse_mask(shape: Tuple[int, int, int] | Tuple[int, int], inset: float) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        cx, cy = w // 2, h // 2
        rx = max(1, int(w * inset))
        ry = max(1, int(h * inset))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, thickness=-1)
        return mask