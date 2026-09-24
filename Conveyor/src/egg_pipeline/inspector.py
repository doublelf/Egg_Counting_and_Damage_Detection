from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import InspectorConfig
from .paths import setup_sibling_paths

setup_sibling_paths()

from egg_inspect import (  # type: ignore  # noqa: E402
    ClassicalConfig,
    ClassicalCrackDetector,
    YoloCrackDetector,
)


VALID_MODES = ("auto", "yolo", "classical")


def build_inspector(
    cfg: InspectorConfig,
    weights_path: Path,
) -> Tuple[object, str]:
    """Construct the crack inspector according to ``cfg.inspector.model.mode``.

    Returns ``(detector, mode_label)`` where ``mode_label`` is one of
    ``"yolo"`` or ``"classical"``.
    """
    mode = (cfg.model.mode or "classical").lower()
    if mode not in VALID_MODES:
        raise ValueError(f"unknown inspector mode: {mode}")

    if mode == "yolo":
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Inspector (yolo) weights not found: {weights_path}"
            )
        return (
            YoloCrackDetector(
                weights=str(weights_path),
                device=cfg.model.device,
                img_size=cfg.model.img_size,
                conf_threshold=cfg.model.conf_threshold,
                iou_threshold=cfg.model.iou_threshold,
                task=cfg.model.task,
            ),
            "yolo",
        )

    if mode == "auto":
        # fall back to classical if the configured weights are missing
        if weights_path.exists():
            try:
                return (
                    YoloCrackDetector(
                        weights=str(weights_path),
                        device=cfg.model.device,
                        img_size=cfg.model.img_size,
                        conf_threshold=cfg.model.conf_threshold,
                        iou_threshold=cfg.model.iou_threshold,
                        task=cfg.model.task,
                    ),
                    "yolo",
                )
            except Exception:
                pass

    # default: classical
    return (
        ClassicalCrackDetector(
            weights=str(weights_path),
            classical_config=ClassicalConfig(),
            device=cfg.model.device,
            img_size=cfg.model.img_size,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            project_root=weights_path.parent,
        ),
        "classical",
    )