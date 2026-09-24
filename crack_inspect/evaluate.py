"""Evaluate a trained YOLOv8-seg cracked-egg model.

Examples
--------
python evaluate.py --weights best.pt --data data/cracked_eggs_demo/data.yaml
python evaluate.py --weights best.pt --data data.yaml --save-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:
    print(f"[evaluate] FATAL: ultralytics required ({exc!r})")
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the cracked-egg model")
    p.add_argument("--weights", type=str, default="best.pt")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-json", action="store_true", help="Save per-image metrics JSON")
    p.add_argument("--plots", action="store_true", help="Render confusion / PR curves")
    p.add_argument("--save-dir", type=str, default="runs_seg/val")
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
    if not Path(args.weights).exists():
        print(f"[evaluate] ERROR: weights not found: {args.weights}")
        return 2
    if not Path(args.data).exists():
        print(f"[evaluate] ERROR: data.yaml not found: {args.data}")
        return 2

    device = resolve_device(args.device)
    print(f"[evaluate] loading {args.weights} on {device}")
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=args.plots,
        save_json=args.save_json,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "weights": args.weights,
        "data": args.data,
        "save_dir": str(metrics.save_dir) if hasattr(metrics, "save_dir") else args.save_dir,
    }
    for k in ("box_map50", "box_map50_95", "mask_map50", "mask_map50_95", "mp", "mr", "f1"):
        v = getattr(metrics, k, None)
        if v is not None:
            summary[k] = float(v) if hasattr(v, "__float__") else v

    out_path = Path(args.save_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[evaluate] summary -> {out_path}")
    print(f"[evaluate] metrics: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())