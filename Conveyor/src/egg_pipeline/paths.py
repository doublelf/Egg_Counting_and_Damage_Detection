from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def egg_detection_root() -> Path:
    return project_root().parent


def setup_sibling_paths() -> None:
    """Add the two upstream projects to ``sys.path`` for in-process reuse.

    * ``Automated-Egg-Counting-System/src`` — for ``egg_counter``.
    * ``Automated-Egg-Counting-System/vendor_yolov5`` — required by the
      vendored YOLOv5 model class at unpickle time.
    * ``crack_inspect/src`` — for ``egg_inspect``.
    * ``Conveyor/src`` — for the local package.
    """
    root = project_root()
    ed = egg_detection_root()
    candidates = [
        str(root / "src"),
        str(ed / "Automated-Egg-Counting-System" / "src"),
        str(ed / "Automated-Egg-Counting-System" / "vendor_yolov5"),
        str(ed / "crack_inspect" / "src"),
    ]
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)