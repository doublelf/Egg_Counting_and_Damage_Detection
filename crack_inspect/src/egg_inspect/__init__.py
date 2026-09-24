"""Egg crack / quality inspection package (YOLOv8-seg).

Local inference wrapper around ultralytics.YOLO with our
:class:`CrackFinding` and :class:`FrameStats` data model so the rest of the
application does not depend on ultralytics internals.

Also ships a hybrid YOLOv5 + OpenCV detector (:class:`ClassicalCrackDetector`)
that requires no crack-egg training data.
"""

from __future__ import annotations

from .config import AppConfig, ClassicalConfig, load_config
from .detector import CrackFinding, YoloCrackDetector
from .classical import ClassicalCrackClassifier, ClassicalCrackDetector
from .analyzer import FrameStats, analyze_frame, is_crack, is_dirty, is_intact, summarize
from .renderer import Renderer
from .report import CsvReport, JsonSummary
from .video import VideoSource, open_source

__version__ = "0.1.0"
__all__ = [
    "AppConfig",
    "ClassicalConfig",
    "load_config",
    "CrackFinding",
    "YoloCrackDetector",
    "ClassicalCrackClassifier",
    "ClassicalCrackDetector",
    "FrameStats",
    "analyze_frame",
    "is_crack",
    "is_dirty",
    "is_intact",
    "summarize",
    "Renderer",
    "CsvReport",
    "JsonSummary",
    "VideoSource",
    "open_source",
]