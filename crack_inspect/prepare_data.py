"""Data preparation for the cracked-egg segmentation model.

Subcommands
-----------
--make-demo      Generate a small synthetic YOLO-seg dataset under
                 data/cracked_eggs_demo/ so the training pipeline can be
                 exercised end-to-end without a real labelled dataset.
--validate PATH  Check that PATH follows the YOLO-seg layout
                 (images/{train,val,test} + labels/...) and report class
                 distribution.
--from-roboflow  (placeholder) Pull a Roboflow dataset using the
                 ``roboflow`` SDK if it is installed.
--info           Print the expected dataset layout and class list.

Real dataset workflow
---------------------
The crackedChickenEggs dataset is ~4900 images across 6 classes. Export it
from Roboflow / CVAT in YOLOv8-seg format and place the resulting
``data.yaml`` next to a folder containing ``images/{train,val,test}`` and
``labels/{train,val,test}`` files. Then run::

    python train_inspector.py --data data/cracked_eggs/data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import yaml


CLASS_NAMES: List[str] = [
    "brow-egg-dirty",
    "brown-egg",
    "brown-egg-crack",
    "white-egg",
    "white-egg-crack",
    "white-egg-dirty",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# --make-demo
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    out_dir: Path
    n_train: int = 60
    n_val: int = 12
    img_w: int = 640
    img_h: int = 480
    seed: int = 42
    min_eggs: int = 1
    max_eggs: int = 3


def _draw_egg(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int, color: Tuple[int, int, int]) -> np.ndarray:
    out = canvas.copy()
    cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)
    # Soft shading
    shade = (int(color[0] * 0.92), int(color[1] * 0.92), int(color[2] * 0.92))
    cv2.ellipse(out, (cx - rx // 4, cy - ry // 4), (rx // 2, ry // 2), 0, 0, 360, shade, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_crack(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    """Draw a curved dark line inside an egg."""
    out = canvas.copy()
    pts: List[Tuple[int, int]] = []
    n = 18
    for i in range(n + 1):
        t = i / n
        ang = -np.pi / 2 + t * np.pi * 0.9
        wobble = np.sin(t * 6.28 * 1.5) * 0.15
        x = int(cx + (rx * 0.7) * np.cos(ang) * (1 + wobble * 0.2))
        y = int(cy + (ry * 0.7) * np.sin(ang) * (1 + wobble * 0.2))
        pts.append((x, y))
    cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=(40, 30, 20), thickness=2, lineType=cv2.LINE_AA)
    # Slightly thinner inner line for realism
    cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=(80, 60, 40), thickness=1, lineType=cv2.LINE_AA)
    return out


def _draw_dirt(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    out = canvas.copy()
    n_blobs = random.randint(2, 5)
    for _ in range(n_blobs):
        bx = int(cx + random.uniform(-rx * 0.6, rx * 0.6))
        by = int(cy + random.uniform(-ry * 0.6, ry * 0.6))
        br = random.randint(3, 8)
        cv2.circle(out, (bx, by), br, (60, 50, 40), thickness=-1, lineType=cv2.LINE_AA)
    return out


def _egg_polygon(cx: int, cy: int, rx: int, ry: int, n: int = 32) -> np.ndarray:
    pts = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        pts.append([cx + rx * np.cos(ang), cy + ry * np.sin(ang)])
    return np.array(pts, dtype=np.float32)


def _crack_polygon(cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    pts: List[List[float]] = []
    n = 18
    for i in range(n + 1):
        t = i / n
        ang = -np.pi / 2 + t * np.pi * 0.9
        wobble = np.sin(t * 6.28 * 1.5) * 0.15
        x = cx + (rx * 0.7) * np.cos(ang) * (1 + wobble * 0.2)
        y = cy + (ry * 0.7) * np.sin(ang) * (1 + wobble * 0.2)
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def _normalize_polygon(poly: np.ndarray, w: int, h: int) -> List[float]:
    flat: List[float] = []
    for x, y in poly:
        flat.append(round(float(x) / w, 6))
        flat.append(round(float(y) / h, 6))
    return flat


def _make_one(out_dir: Path, idx: int, cfg: DemoConfig, split: str) -> None:
    rng = random.Random(cfg.seed + idx)
    random.seed(cfg.seed + idx)
    np.random.seed(cfg.seed + idx)

    img = np.full((cfg.img_h, cfg.img_w, 3), random.randint(35, 65), dtype=np.uint8)
    # Subtle background texture
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    label_lines: List[str] = []
    n_eggs = random.randint(cfg.min_eggs, cfg.max_eggs)
    for _ in range(n_eggs):
        rx = random.randint(28, 50)
        ry = random.randint(36, 60)
        cx = random.randint(rx + 10, cfg.img_w - rx - 10)
        cy = random.randint(ry + 10, cfg.img_h - ry - 10)
        is_brown = random.random() < 0.55
        base = (random.randint(140, 175), random.randint(95, 130), random.randint(60, 95)) if is_brown else (
            random.randint(220, 245), random.randint(215, 240), random.randint(200, 230)
        )

        kind = random.choices(
            ["clean", "crack", "dirty"],
            weights=[0.55, 0.30, 0.15],
            k=1,
        )[0]
        if kind == "clean":
            cls = "brown-egg" if is_brown else "white-egg"
        elif kind == "crack":
            cls = "brown-egg-crack" if is_brown else "white-egg-crack"
            img = _draw_crack(img, cx, cy, rx, ry)
        else:
            cls = "brow-egg-dirty" if is_brown else "white-egg-dirty"
            img = _draw_dirt(img, cx, cy, rx, ry)

        img = _draw_egg(img, cx, cy, rx, ry, base)

        egg_poly = _egg_polygon(cx, cy, rx, ry)
        flat = _normalize_polygon(egg_poly, cfg.img_w, cfg.img_h)
        cls_id = CLASS_TO_ID[cls]
        label_lines.append(f"{cls_id} " + " ".join(str(v) for v in flat))

        if "crack" in cls:
            crack_poly = _crack_polygon(cx, cy, rx, ry)
            crack_cls_id = CLASS_TO_ID[cls]  # same class for instance seg
            crack_flat = _normalize_polygon(crack_poly, cfg.img_w, cfg.img_h)
            label_lines.append(f"{crack_cls_id} " + " ".join(str(v) for v in crack_flat))

    img_path = out_dir / "images" / split / f"{split}_{idx:04d}.jpg"
    lbl_path = out_dir / "labels" / split / f"{split}_{idx:04d}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def make_demo(cfg: DemoConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare-data] Generating demo dataset at {cfg.out_dir}")
    for split, count in [("train", cfg.n_train), ("val", cfg.n_val)]:
        for i in range(count):
            _make_one(cfg.out_dir, i, cfg, split)
        print(f"[prepare-data]   {split}: {count} images")

    yaml_path = cfg.out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(cfg.out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"[prepare-data] wrote {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# --validate
# ---------------------------------------------------------------------------


def validate(data_dir: Path) -> int:
    if not data_dir.exists():
        print(f"[prepare-data] ERROR: {data_dir} does not exist")
        return 2
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"[prepare-data] ERROR: {yaml_path} missing")
        return 2

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    print(f"[prepare-data] data.yaml keys: {list(data.keys())}")
    print(f"[prepare-data] classes ({data.get('nc')}): {data.get('names')}")

    base = data_dir
    if data.get("path"):
        p = Path(data["path"])
        base = p if p.is_absolute() else (data_dir.parent / p).resolve()

    counts = {}
    for split in ["train", "val", "test"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            print(f"[prepare-data]   {split}: missing ({img_dir})")
            continue
        n_img = sum(1 for _ in img_dir.iterdir() if _.is_file())
        n_lbl = sum(1 for _ in lbl_dir.iterdir() if _.is_file()) if lbl_dir.exists() else 0
        counts[split] = (n_img, n_lbl)
        print(f"[prepare-data]   {split}: {n_img} images, {n_lbl} labels")

    class_hist: dict = {n: 0 for n in data.get("names", [])}
    for split in ["train", "val", "test"]:
        lbl_dir = base / "labels" / split
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.iterdir():
            if not f.is_file():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                names = data.get("names", [])
                if 0 <= cid < len(names):
                    class_hist[names[cid]] += 1
    print("[prepare-data] class histogram:")
    for k, v in class_hist.items():
        print(f"  {k}: {v}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare / inspect / generate datasets")
    p.add_argument("--make-demo", action="store_true", help="Generate a synthetic demo dataset")
    p.add_argument("--validate", type=str, default=None, metavar="PATH", help="Validate an existing dataset directory")
    p.add_argument("--from-roboflow", action="store_true", help="(stub) Use Roboflow SDK if installed")
    p.add_argument("--info", action="store_true", help="Print expected dataset layout")
    p.add_argument("--out", type=str, default="data/cracked_eggs_demo", help="Demo dataset output dir")
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-val", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.info or not (args.make_demo or args.validate or args.from_roboflow):
        print(
            "Expected dataset layout (YOLOv8-seg):\n"
            "  <root>/data.yaml\n"
            "  <root>/images/{train,val,test}/*.jpg\n"
            "  <root>/labels/{train,val,test}/*.txt\n"
            "\n"
            f"Default 6 classes:\n  " + "\n  ".join(CLASS_NAMES) + "\n"
        )
        return 0

    if args.make_demo:
        cfg = DemoConfig(
            out_dir=Path(args.out),
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )
        make_demo(cfg)
        return 0

    if args.validate:
        return validate(Path(args.validate))

    if args.from_roboflow:
        try:
            import roboflow  # noqa: F401
        except Exception as exc:
            print(f"[prepare-data] roboflow SDK not installed ({exc!r}).")
            print("[prepare-data] Install with: pip install roboflow")
            return 2
        print("[prepare-data] Roboflow download path not configured. Set your own API key / workspace / project.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())