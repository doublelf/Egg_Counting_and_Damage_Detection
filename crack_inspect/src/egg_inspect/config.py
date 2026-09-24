from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


CRACKED_SUFFIX = "-crack"
DIRTY_SUFFIX = "-dirty"
INTACT_NAMES = {"brown-egg", "white-egg"}

DEFAULT_NAMES: List[str] = [
    "brow-egg-dirty",
    "brown-egg",
    "brown-egg-crack",
    "white-egg",
    "white-egg-crack",
    "white-egg-dirty",
]


@dataclass
class ModelConfig:
    weights: str = "best.pt"
    device: str = "auto"
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: Optional[List[int]] = None  # None = all
    task: str = "segment"  # "segment" or "detect"


@dataclass
class VisualConfig:
    mask_alpha: float = 0.45
    box_thickness: int = 2
    label_font_scale: float = 0.5
    hud_font_scale: float = 0.9
    show_mask: bool = True
    show_box: bool = True


@dataclass
class FilterConfig:
    crack_only: bool = False         # drop non-crack findings before reporting
    min_mask_pixels: int = 8         # ignore masks with too few pixels
    min_bbox_area: int = 16          # ignore tiny bboxes (px^2)


@dataclass
class ClassicalConfig:
    """Tunable parameters for :class:`ClassicalCrackClassifier`."""
    color_a_threshold: int = 135
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    bilateral_d: int = 9
    bilateral_sigma: float = 75.0
    blackhat_kernel: List[int] = field(default_factory=lambda: [15, 15])
    threshold_block_size: int = 51
    threshold_c: int = -10
    cleanup_kernel: List[int] = field(default_factory=lambda: [3, 3])
    min_component_area: int = 12
    crack_min_aspect: float = 3.0
    crack_max_solidity: float = 0.5
    dirt_min_aspect: float = 0.7
    dirt_max_aspect: float = 1.5
    dirt_max_solidity: float = 0.7
    dirt_min_area_ratio: float = 0.01
    crack_area_threshold: float = 0.01
    dirt_area_threshold: float = 0.02
    ellipse_inset: float = 0.45


@dataclass
class VideoConfig:
    source: str = "0"
    loop: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass
class OutputConfig:
    csv_path: str = "logs/inspect.csv"
    summary_path: str = "logs/summary.json"
    save_video: Optional[str] = None
    show_window: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_NAMES))
    mode: str = "auto"  # auto | yolo | classical | hybrid
    classical_egg_weights: str = "../Automated-Egg-Counting-System/best.pt"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model),
            "visual": asdict(self.visual),
            "filter": asdict(self.filter),
            "video": asdict(self.video),
            "output": asdict(self.output),
            "class_names": list(self.class_names),
        }


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load YAML config and merge into dataclass defaults.

    Resolution order (later wins):
    1. dataclass defaults
    2. packaged default ``config.yaml`` if present
    3. user-provided ``path`` if present
    """
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

    cfg = AppConfig()
    if "model" in data:
        cfg.model = ModelConfig(**{k: v for k, v in data["model"].items() if k in ModelConfig.__dataclass_fields__})
    if "visual" in data:
        cfg.visual = VisualConfig(**{k: v for k, v in data["visual"].items() if k in VisualConfig.__dataclass_fields__})
    if "filter" in data:
        cfg.filter = FilterConfig(**{k: v for k, v in data["filter"].items() if k in FilterConfig.__dataclass_fields__})
    if "classical" in data:
        cfg.classical = ClassicalConfig(**{k: v for k, v in data["classical"].items() if k in ClassicalConfig.__dataclass_fields__})
    if "video" in data:
        cfg.video = VideoConfig(**{k: v for k, v in data["video"].items() if k in VideoConfig.__dataclass_fields__})
    if "output" in data:
        cfg.output = OutputConfig(**{k: v for k, v in data["output"].items() if k in OutputConfig.__dataclass_fields__})
    if "class_names" in data:
        cfg.class_names = list(data["class_names"])
    if "mode" in data:
        cfg.mode = str(data["mode"]).lower()
    if "classical_egg_weights" in data:
        cfg.classical_egg_weights = str(data["classical_egg_weights"])
    return cfg