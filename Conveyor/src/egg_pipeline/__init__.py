"""Conveyor belt egg counter + crack detector.

Combines the vendored YOLOv5-based counter (``Automated-Egg-Counting-System/``)
with the vendored classical-CV crack classifier (``crack_inspect/``) into a
single per-egg-quality report. The two upstream projects are not modified;
this package only adds orchestration glue.
"""

from __future__ import annotations

from .paths import setup_sibling_paths
from .config import (
    AssociationConfig,
    ConveyorConfig,
    CounterConfig,
    CounterLineConfig,
    CounterModelConfig,
    CounterTrackerConfig,
    InspectorConfig,
    InspectorFilterConfig,
    InspectorModelConfig,
    OutputConfig,
    VideoConfig,
    config_paths,
    load_config,
)
from .counter import build_counter
from .inspector import build_inspector
from .associator import EggVerdict, VerdictSource, associate
from .overlay import CombinedOverlay, OverlayStats
from .report import CrossingsCsv, FramesCsv, JsonSummary
from .pipeline import PipelineResult, run

__all__ = [
    "setup_sibling_paths",
    "ConveyorConfig",
    "load_config",
    "config_paths",
    "CounterConfig",
    "CounterModelConfig",
    "CounterLineConfig",
    "CounterTrackerConfig",
    "InspectorConfig",
    "InspectorModelConfig",
    "InspectorFilterConfig",
    "AssociationConfig",
    "VideoConfig",
    "OutputConfig",
    "build_counter",
    "build_inspector",
    "EggVerdict",
    "VerdictSource",
    "associate",
    "CombinedOverlay",
    "OverlayStats",
    "CrossingsCsv",
    "FramesCsv",
    "JsonSummary",
    "PipelineResult",
    "run",
]