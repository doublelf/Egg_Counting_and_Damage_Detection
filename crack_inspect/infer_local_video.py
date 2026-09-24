"""Convenience wrapper for running ``crack_inspect.py`` against a local
video file (or webcam) without having to remember every flag.

The wrapper delegates to ``crack_inspect.py`` via :func:`runpy.run_path`,
so the full feature set of the inspection pipeline (mode selection,
``--conf-list`` sweeps, CSV / JSON reports, mask overlay, save-video)
is preserved.

Examples
--------
# Pick a local video via Tkinter file dialog, run in classical mode
python infer_local_video.py --picker

# Explicit local file
python infer_local_video.py --video path/to/run.mp4

# Webcam shortcut (same as ``python crack_inspect.py --source 0``)
python infer_local_video.py --webcam
python infer_local_video.py --webcam 1     # secondary camera

# Force a YOLO-seg run if you have a trained best.pt
python infer_local_video.py --video run.mp4 --mode yolo

# Save annotated output without the GUI
python infer_local_video.py --video run.mp4 --no-display \
                            --save-video annotated.mp4 \
                            --csv counts.csv --summary summary.json

# Dry-run: print the crack_inspect.py command that would run
python infer_local_video.py --video run.mp4 --dry-run
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import List, Optional


VALID_MODES = ("auto", "yolo", "classical", "hybrid")
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run crack_inspect.py against a local video (or webcam) with sensible defaults."
    )
    p.add_argument("--video", type=str, default=None,
                   help="Path to a local video file (.mp4/.avi/.mov/...).")
    p.add_argument("--webcam", type=int, nargs="?", const=0, default=None, metavar="INDEX",
                   help="Use webcam INDEX (default 0). Shorthand for --source 0.")
    p.add_argument("--picker", action="store_true",
                   help="Open a Tkinter file dialog to choose a video interactively.")
    p.add_argument("--mode", type=str, choices=VALID_MODES, default="classical",
                   help="Detector mode passed to crack_inspect.py (default: classical).")
    p.add_argument("--output-dir", type=str, default="output/local_infer",
                   help="Directory for annotated mp4 + CSV/JSON outputs.")
    p.add_argument("--save-video", type=str, default=None,
                   help="Annotated mp4 filename (default: <output-dir>/annotated.mp4).")
    p.add_argument("--csv", type=str, default=None, help="Per-frame CSV filename.")
    p.add_argument("--summary", type=str, default=None, help="JSON summary filename.")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode (no OpenCV preview window).")
    p.add_argument("--conf", type=float, default=None, help="Override model.conf_threshold.")
    p.add_argument("--iou", type=float, default=None, help="Override model.iou_threshold.")
    p.add_argument("--conf-list", type=str, default=None,
                   help="Comma-separated conf thresholds to sweep, e.g. '0.2,0.35,0.5'.")
    p.add_argument("--loop", action="store_true", help="Loop the input file.")
    p.add_argument("--crack-only", action="store_true",
                   help="Drop non-crack findings before stats.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the crack_inspect.py command instead of executing it.")
    p.add_argument("--crack-inspect", type=str, default=str(HERE / "crack_inspect.py"),
                   help="Path to crack_inspect.py (default: ./crack_inspect.py).")
    return p.parse_args()


def _resolve_source(args: argparse.Namespace) -> str:
    """Resolve the video source string for crack_inspect.py's --source."""
    if args.webcam is not None:
        return str(args.webcam)
    if args.video:
        path = Path(args.video)
        if not path.is_absolute():
            path = (HERE / args.video).resolve()
        if not path.exists():
            raise FileNotFoundError(f"video file not found: {path}")
        return str(path)
    if args.picker:
        return _pick_file_via_tk()
    raise SystemExit(
        "ERROR: must supply one of --video PATH, --webcam [INDEX], or --picker"
    )


def _pick_file_via_tk() -> str:
    """Open a Tkinter file dialog and return the selected path, or sys.exit(0) if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[infer-local] tkinter unavailable: {exc!r}")
        raise SystemExit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.flv *.wmv"),
        ("All files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Pick a local video to inspect", filetypes=filetypes)
    root.destroy()
    if not path:
        print("[infer-local] cancelled.")
        sys.exit(0)
    return path


def _resolve_output_paths(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (HERE / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "save_video": str(out_dir / "annotated.mp4") if not args.save_video else str(out_dir / args.save_video),
        "csv": str(out_dir / "counts.csv") if not args.csv else str(out_dir / args.csv),
        "summary": str(out_dir / "summary.json") if not args.summary else str(out_dir / args.summary),
    }
    if args.no_display and not args.save_video:
        # Headless without --save-video: skip writing an mp4.
        paths["save_video"] = None
    return paths


def _build_argv(source: str, args: argparse.Namespace, paths: dict) -> List[str]:
    argv = [str(Path(args.crack_inspect))]
    argv += ["--source", source]
    argv += ["--mode", args.mode]
    if args.no_display:
        argv += ["--no-display"]
    if args.conf is not None:
        argv += ["--conf", str(args.conf)]
    if args.iou is not None:
        argv += ["--iou", str(args.iou)]
    if args.conf_list:
        argv += ["--conf-list", args.conf_list]
    if args.loop:
        argv += ["--loop"]
    if args.crack_only:
        argv += ["--crack-only"]
    if paths["save_video"]:
        argv += ["--save-video", paths["save_video"]]
    argv += ["--csv", paths["csv"]]
    argv += ["--summary", paths["summary"]]
    return argv


def main() -> int:
    args = parse_args()

    # Sanity: at least one source flag must be set.
    if not (args.video or args.webcam is not None or args.picker):
        print("ERROR: must supply one of --video PATH, --webcam [INDEX], or --picker")
        return 2

    try:
        source = _resolve_source(args)
    except FileNotFoundError as exc:
        print(f"[infer-local] {exc}")
        return 2

    paths = _resolve_output_paths(args)
    argv = _build_argv(source, args, paths)

    print(f"[infer-local] source     : {source}")
    print(f"[infer-local] mode       : {args.mode}")
    print(f"[infer-local] output dir : {args.output_dir}")
    if paths["save_video"]:
        print(f"[infer-local] annotated  -> {paths['save_video']}")
    print(f"[infer-local] csv        -> {paths['csv']}")
    print(f"[infer-local] summary    -> {paths['summary']}")
    print(f"[infer-local] command    : {' '.join(argv)}")

    if args.dry_run:
        return 0

    print(f"[infer-local] launching crack_inspect.py ...")
    # Delegate to crack_inspect.py: replaces sys.argv and runs as __main__.
    saved_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(args.crack_inspect, run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except FileNotFoundError as exc:
        print(f"[infer-local] ERROR: {exc}")
        return 2
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    sys.exit(main())