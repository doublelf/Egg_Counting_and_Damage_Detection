from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover
    YOLO = None  # type: ignore
    _YOLO_IMPORT_ERROR = _exc


@dataclass
class CrackFinding:
    """Single detected egg region with optional segmentation mask."""
    cls_id: int
    name: str
    conf: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    # H x W bool ndarray, full-frame sized. May be all-False for detect-only models.
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    bbox_area: int = 0
    mask_area: int = 0
    mask_area_ratio: float = 0.0  # mask_area / bbox_area

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0


def _mask_to_full(masks_data, box_xyxy, img_h: int, img_w: int) -> np.ndarray:
    """Crop + resize the per-finding prototype mask to a full-frame sized bool mask.

    ``masks_data`` is the ``M x H_proto x W_proto`` tensor returned by ultralytics
    for one image. ``box_xyxy`` selects the row corresponding to this finding.
    Falls back to a bbox-fill mask if the mask tensor is unavailable.
    """
    out = np.zeros((img_h, img_w), dtype=bool)
    if masks_data is None or len(masks_data) == 0:
        return out
    try:
        m = masks_data[int(box_xyxy[6])] if masks_data.shape[-1] != masks_data.shape[1] else masks_data[int(box_xyxy[6])]
    except Exception:
        return out
    m_arr = np.asarray(m)
    if m_arr.ndim != 2:
        return out
    if m_arr.shape != (img_h, img_w):
        import cv2
        m_arr = cv2.resize(m_arr.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    out = m_arr.astype(bool)
    return out


def _bbox_mask(img_h: int, img_w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    out = np.zeros((img_h, img_w), dtype=bool)
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = True
    return out


class YoloCrackDetector:
    """Ultralytics YOLO wrapper producing :class:`CrackFinding` objects."""

    def __init__(
        self,
        weights: str,
        device: str = "auto",
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        task: str = "segment",
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is required for YoloCrackDetector. "
                f"Original import error: {_YOLO_IMPORT_ERROR!r}"
            )
        self.weights = weights
        self.device = self._resolve_device(device)
        self.img_size = int(img_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.classes = classes
        self.task = task.lower()

        self._model = YOLO(weights, task=self.task if self.task in ("detect", "segment") else None)
        self.class_names: dict = dict(self._model.names) if hasattr(self._model, "names") else {}

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def set_conf_threshold(self, value: float) -> None:
        v = float(value)
        self.conf_threshold = v
        try:
            self._model.overrides["conf"] = v
        except Exception:
            pass

    def set_iou_threshold(self, value: float) -> None:
        v = float(value)
        self.iou_threshold = v
        try:
            self._model.overrides["iou"] = v
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> List[CrackFinding]:
        if frame is None or frame.size == 0:
            return []
        results = self._model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )
        findings: List[CrackFinding] = []
        if not results:
            return findings
        r = results[0]
        names = getattr(r, "names", None) or self.class_names or {}
        img_h, img_w = frame.shape[:2]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return findings

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
        masks_obj = getattr(r, "masks", None)
        masks_data = getattr(masks_obj, "data", None) if masks_obj is not None else None
        if masks_data is not None:
            try:
                masks_data = masks_data.cpu().numpy()
            except Exception:
                masks_data = None

        if xyxy is None or conf is None or cls is None:
            return findings

        for i in range(xyxy.shape[0]):
            cid = int(cls[i])
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            if masks_data is not None and masks_data.ndim == 3:
                mask = _mask_to_full(masks_data, xyxy[i], img_h, img_w)
                # _mask_to_full expects index in [6]; if xyxy row has no cls col we fall back to i
                if not mask.any():
                    mask = _bbox_mask(img_h, img_w, int(x1), int(y1), int(x2), int(y2))
            else:
                mask = _bbox_mask(img_h, img_w, int(x1), int(y1), int(x2), int(y2))

            bbox_area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
            mask_area = int(mask.sum())
            ratio = (mask_area / bbox_area) if bbox_area > 0 else 0.0

            findings.append(
                CrackFinding(
                    cls_id=cid,
                    name=str(names.get(cid, str(cid))),
                    conf=float(conf[i]),
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    mask=mask,
                    bbox_area=bbox_area,
                    mask_area=mask_area,
                    mask_area_ratio=ratio,
                )
            )
        return findings