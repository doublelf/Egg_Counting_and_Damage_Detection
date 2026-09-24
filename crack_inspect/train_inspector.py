"""Train a YOLOv8-seg cracked-egg detector.

Examples
--------
# Use the bundled demo dataset (created by prepare_data.py --make-demo)
python train_inspector.py --data data/cracked_eggs_demo/data.yaml

# Use your own labelled dataset
python train_inspector.py --data /path/to/cracked_eggs/data.yaml --epochs 100

# Fine-tune from a custom checkpoint
python train_inspector.py --data data.yaml --weights runs/segment/train/weights/last.pt --resume

# After training, best.pt is auto-copied to ./best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:
    print(f"[train] FATAL: ultralytics is required ({exc!r})")
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the YOLOv8-seg cracked-egg model")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--weights", type=str, default="yolov8s-seg.pt",
                   help="Starting weights (default: yolov8s-seg.pt — downloaded automatically)")
    p.add_argument("--model-cfg", type=str, default=None,
                   help="Optional model yaml (e.g. yolov8-seg-C2f-Faster.yaml) for arch tweaks")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | 0 | cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", type=str, default="runs_seg")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--copy-best-to", type=str, default="best.pt",
                   help="After training, copy best.pt to this path (default: ./best.pt)")
    p.add_argument("--patience", type=int, default=20, help="EarlyStopping patience")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every N epochs (-1 to disable)")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[train] ERROR: data.yaml not found: {data_yaml}")
        return 2

    device = resolve_device(args.device)
    print(f"[train] device: {device}")
    print(f"[train] data:   {data_yaml}")
    print(f"[train] weights: {args.weights}")

    if args.resume:
        # Resume requires a checkpoint path
        if not Path(args.weights).exists():
            print(f"[train] ERROR: --resume needs an existing checkpoint at {args.weights}")
            return 2
        model = YOLO(args.weights)
        train_kwargs = {"resume": True}
        print(f"[train] resuming from {args.weights}")
    else:
        if args.model_cfg:
            model = YOLO(args.model_cfg)
            if args.weights and Path(args.weights).exists():
                model.load(args.weights)
        else:
            model = YOLO(args.weights)
        train_kwargs = dict(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            patience=args.patience,
            seed=args.seed,
            exist_ok=True,
            plots=False,
            verbose=True,
        )
        if args.save_period > 0:
            train_kwargs["save_period"] = args.save_period

    try:
        results = model.train(**train_kwargs)
    except Exception as exc:
        print(f"[train] FATAL: training failed: {exc!r}")
        return 3

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.exists():
        print(f"[train] WARNING: best.pt not found at {best_path}")
        return 4

    if args.copy_best_to:
        target = Path(args.copy_best_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target)
        print(f"[train] copied best.pt -> {target}")

    print(f"[train] Done.")
    print(f"[train] best.pt  = {best_path}")
    print(f"[train] last.pt  = {last_path}")
    print(f"[train] results  = {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())