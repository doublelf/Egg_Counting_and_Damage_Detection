"""Run inference on the demo PNGs from `crackedChickenEggs161/`.

These demo images were shipped with the original upstream repository and
have **no labels**, so this script is purely a visual smoke check — it
shows what a trained model produces on real-looking photographs.

Examples
--------
python infer_demo.py                           # uses ./best.pt and ../crackedChickenEggs161/*.png
python infer_demo.py --weights path/to/best.pt
python infer_demo.py --source /path/to/images/ --out output/demo_infer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except Exception as exc:
    print(f"[infer-demo] FATAL: ultralytics required ({exc!r})")
    sys.exit(2)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from egg_inspect.renderer import Renderer  # noqa: E402
from egg_inspect.config import VisualConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on demo PNGs")
    p.add_argument("--weights", type=str, default="best.pt")
    p.add_argument("--source", type=str, default=None,
                   help="Image folder or single image; default: ../crackedChickenEggs161")
    p.add_argument("--out", type=str, default="output/demo_infer")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-render", action="store_true", help="Skip renderer; just save raw YOLO outputs")
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
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[infer-demo] ERROR: weights not found: {weights}")
        return 2

    source = Path(args.source) if args.source else (_HERE.parent / "crackedChickenEggs161")
    if not source.exists():
        print(f"[infer-demo] ERROR: source not found: {source}")
        return 2

    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    else:
        images = [source]
    if not images:
        print(f"[infer-demo] no images found at {source}")
        return 2
    print(f"[infer-demo] {len(images)} images from {source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[infer-demo] device: {device}, weights: {weights}, conf: {args.conf}")
    model = YOLO(str(weights))
    model.overrides["conf"] = args.conf
    model.overrides["iou"] = args.iou

    renderer = Renderer(VisualConfig()) if not args.no_render else None

    saved = 0
    for img_path in images:
        results = model.predict(source=str(img_path), conf=args.conf, iou=args.iou, device=device, verbose=False)
        if not results:
            continue
        r = results[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        findings = []
        if hasattr(r, "boxes") and r.boxes is not None:
            boxes = r.boxes
            masks_obj = getattr(r, "masks", None)
            masks_data = getattr(masks_obj, "data", None) if masks_obj is not None else None
            if masks_data is not None:
                try:
                    masks_data = masks_data.cpu().numpy()
                except Exception:
                    masks_data = None
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
            cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
            names = getattr(r, "names", {}) or {}
            if xyxy is not None and conf is not None and cls is not None:
                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    mask = np.zeros((h, w), dtype=bool)
                    if masks_data is not None and masks_data.ndim == 3 and i < masks_data.shape[0]:
                        m_arr = np.asarray(masks_data[i])
                        if m_arr.shape != (h, w):
                            mask = cv2.resize(m_arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        else:
                            mask = m_arr.astype(bool)
                    bbox_area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
                    mask_area = int(mask.sum())
                    from egg_inspect.detector import CrackFinding
                    findings.append(CrackFinding(
                        cls_id=int(cls[i]),
                        name=str(names.get(int(cls[i]), str(int(cls[i])))),
                        conf=float(conf[i]),
                        x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                        mask=mask, bbox_area=bbox_area, mask_area=mask_area,
                        mask_area_ratio=(mask_area / bbox_area) if bbox_area > 0 else 0.0,
                    ))

        if renderer is not None:
            from egg_inspect.analyzer import analyze_frame
            stats = analyze_frame(findings)
            canvas = renderer.draw(img, findings, stats)
        else:
            canvas = img

        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), canvas)
        saved += 1
        print(f"[infer-demo] {img_path.name} -> {out_path.name} ({len(findings)} finding(s))")

    print(f"[infer-demo] saved {saved} annotated image(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())