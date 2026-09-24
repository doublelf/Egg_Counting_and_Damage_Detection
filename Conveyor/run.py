"""Conveyor belt counter + crack inspector (independent project).

Examples
--------
# Local video
python run.py --video path/to/run.mp4

# Webcam (default index 0)
python run.py --source 0
python run.py --webcam

# Custom line + thresholds
python run.py --video run.mp4 --line-start 100 240 --line-end 540 240 \
              --conf-counter 0.35 --conf-inspector 0.25 \
              --associate-iou 0.3

# Headless with annotated mp4 + per-egg CSV
python run.py --video run.mp4 --no-display --save-video output/annotated.mp4
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
sys_path_inserted = False

import sys  # noqa: E402

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from egg_pipeline import (  # noqa: E402
    ConveyorConfig,
    CrossingsCsv,
    FramesCsv,
    JsonSummary,
    load_config,
    run,
)


VALID_MODES = ("auto", "yolo", "classical")  # noqa: F811


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conveyor belt counter + crack inspector")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--source", type=str, default=None, help="Override video.source")
    p.add_argument("--video", type=str, default=None, help="Local video file")
    p.add_argument("--webcam", type=int, nargs="?", const=0, default=None, metavar="INDEX",
                   help="Use webcam INDEX (default 0).")
    p.add_argument("--counter-weights", type=str, default=None, help="Override counter model.weights")
    p.add_argument("--inspector-weights", type=str, default=None, help="Override inspector model.weights")
    p.add_argument(
        "--inspector-mode",
        type=str,
        choices=VALID_MODES,
        default=None,
        help="Inspector mode (default: classical).",
    )
    p.add_argument("--conf-counter", type=float, default=None, help="Override counter conf")
    p.add_argument("--conf-inspector", type=float, default=None, help="Override inspector conf")
    p.add_argument("--line-start", type=int, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--line-end", type=int, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument(
        "--line-direction",
        type=str,
        choices=["down", "up", "any"],
        default=None,
        help="Override counter line direction",
    )
    p.add_argument("--associate-iou", type=float, default=None, help="Override association.min_iou")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-video", type=str, default=None)
    p.add_argument("--frames-csv", type=str, default=None)
    p.add_argument("--crossings-csv", type=str, default=None)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--loop", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: ConveyorConfig, args: argparse.Namespace) -> ConveyorConfig:
    # source precedence: --source > --video > --webcam
    if args.source is not None:
        cfg.video.source = args.source
    elif args.video is not None:
        cfg.video.source = args.video
    elif args.webcam is not None:
        cfg.video.source = str(args.webcam)
    if args.counter_weights is not None:
        cfg.counter.model.weights = args.counter_weights
    if args.inspector_weights is not None:
        cfg.inspector.model.weights = args.inspector_weights
    if args.inspector_mode is not None:
        cfg.inspector.model.mode = args.inspector_mode
    if args.conf_counter is not None:
        cfg.counter.model.conf_threshold = args.conf_counter
    if args.conf_inspector is not None:
        cfg.inspector.model.conf_threshold = args.conf_inspector
    if args.line_start is not None:
        cfg.counter.line.start = list(args.line_start)
    if args.line_end is not None:
        cfg.counter.line.end = list(args.line_end)
    if args.line_direction is not None:
        cfg.counter.line.direction = args.line_direction
    if args.associate_iou is not None:
        cfg.association.min_iou = args.associate_iou
    if args.no_display:
        cfg.output.show_window = False
    if args.save_video is not None:
        cfg.output.save_video = args.save_video
    if args.frames_csv is not None:
        cfg.output.frames_csv = args.frames_csv
    if args.crossings_csv is not None:
        cfg.output.crossings_csv = args.crossings_csv
    if args.summary is not None:
        cfg.output.summary_json = args.summary
    if args.loop:
        cfg.video.loop = True
    return cfg


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    try:
        result = run(cfg)
    except FileNotFoundError as exc:
        print(f"[conveyor] ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"[conveyor] FATAL: {exc!r}")
        raise

    print(
        f"[conveyor] Done. crossings={result.total_crossings} "
        f"defective={result.defective_count} frames={result.frames_seen}"
    )
    print(
        f"[conveyor] Artifacts: "
        f"frames={result.csv_paths['frames_csv']} "
        f"crossings={result.csv_paths['crossings_csv']} "
        f"summary={result.csv_paths['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())