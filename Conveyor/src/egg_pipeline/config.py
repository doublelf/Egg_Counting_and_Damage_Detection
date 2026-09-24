from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CounterModelConfig:
    weights: str = "../Automated-Egg-Counting-System/best.pt"
    device: str = "auto"
    img_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45


@dataclass
class CounterLineConfig:
    start: List[int] = field(default_factory=lambda: [100, 240])
    end: List[int] = field(default_factory=lambda: [540, 240])
    direction: str = "down"  # down | up | any
    min_track_length: int = 3


@dataclass
class CounterTrackerConfig:
    max_lost_frames: int = 30
    iou_match_threshold: float = 0.2


@dataclass
class CounterConfig:
    model: CounterModelConfig = field(default_factory=CounterModelConfig)
    line: CounterLineConfig = field(default_factory=CounterLineConfig)
    tracker: CounterTrackerConfig = field(default_factory=CounterTrackerConfig)


@dataclass
class InspectorModelConfig:
    weights: str = "../Automated-Egg-Counting-System/best.pt"   # vendored YOLOv5, classical pipeline
    device: str = "auto"
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    task: str = "segment"  # only used by mode=yolo
    mode: str = "classical"  # auto | classical | yolo


@dataclass
class InspectorFilterConfig:
    crack_only: bool = False
    min_mask_pixels: int = 0       # 0 = disabled
    min_bbox_area: int = 16


@dataclass
class InspectorConfig:
    model: InspectorModelConfig = field(default_factory=InspectorModelConfig)
    filter: InspectorFilterConfig = field(default_factory=InspectorFilterConfig)


@dataclass
class AssociationConfig:
    min_iou: float = 0.3  # crossing event ↔ crack finding match threshold


@dataclass
class VideoConfig:
    source: str = "0"
    loop: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass
class OutputConfig:
    frames_csv: str = "logs/frames.csv"
    crossings_csv: str = "logs/crossings.csv"
    summary_json: str = "logs/summary.json"
    save_video: Optional[str] = None
    show_window: bool = True


@dataclass
class ConveyorConfig:
    counter: CounterConfig = field(default_factory=CounterConfig)
    inspector: InspectorConfig = field(default_factory=InspectorConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counter": {
                "model": asdict(self.counter.model),
                "line": asdict(self.counter.line),
                "tracker": asdict(self.counter.tracker),
            },
            "inspector": {
                "model": asdict(self.inspector.model),
                "filter": asdict(self.inspector.filter),
            },
            "association": asdict(self.association),
            "video": asdict(self.video),
            "output": asdict(self.output),
        }


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> ConveyorConfig:
    pkg_root = Path(__file__).resolve().parents[2]
    default_yaml = pkg_root / "config.yaml"

    data: Dict[str, Any] = {}
    if default_yaml.exists():
        with default_yaml.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    if path:
        user_path = Path(path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {user_path}")
        with user_path.open("r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
        data = _merge(data, user_data)

    cfg = ConveyorConfig()

    def assign(cls, target_dict: Dict[str, Any]):
        return cls(**{k: v for k, v in target_dict.items() if k in cls.__dataclass_fields__})

    if "counter" in data:
        c = data["counter"]
        cfg.counter.model = assign(CounterModelConfig, c.get("model", {}))
        cfg.counter.line = assign(CounterLineConfig, c.get("line", {}))
        cfg.counter.tracker = assign(CounterTrackerConfig, c.get("tracker", {}))
    if "inspector" in data:
        i = data["inspector"]
        cfg.inspector.model = assign(InspectorModelConfig, i.get("model", {}))
        cfg.inspector.filter = assign(InspectorFilterConfig, i.get("filter", {}))
    if "association" in data:
        cfg.association = assign(AssociationConfig, data["association"])
    if "video" in data:
        cfg.video = assign(VideoConfig, data["video"])
    if "output" in data:
        cfg.output = assign(OutputConfig, data["output"])
    return cfg


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def config_paths(cfg: ConveyorConfig) -> Dict[str, Path]:
    here = Path(__file__).resolve().parents[1]
    ed = here.parent  # Conveyor/
    return {
        "counter_weights": resolve_path(cfg.counter.model.weights, ed),
        "inspector_weights": resolve_path(cfg.inspector.model.weights, ed),
        "frames_csv": resolve_path(cfg.output.frames_csv, ed),
        "crossings_csv": resolve_path(cfg.output.crossings_csv, ed),
        "summary_json": resolve_path(cfg.output.summary_json, ed),
        "save_video": resolve_path(cfg.output.save_video, ed) if cfg.output.save_video else None,
    }