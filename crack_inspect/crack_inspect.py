"""Main entry for the Egg Crack / Quality Inspection System.

Supports four detector modes via ``--mode``:

* ``auto``      — YOLOv8-seg when ``best.pt`` is available, otherwise the
                  classical (YOLOv5 + OpenCV) pipeline.
* ``yolo``      — force YOLOv8-seg (requires ``best.pt``).
* ``classical`` — force the OpenCV pipeline; no crack dataset required.
* ``hybrid``    — vendored YOLOv5 finds eggs, OpenCV classifies cracks.

Examples
--------
python crack_inspect.py --source 0
python crack_inspect.py --mode classical --source ../crackedChickenEggs161/
python crack_inspect.py --source path/to/video.mp4 --no-display
python crack_inspect.py --source frames/ --save-video output/run.mp4
python crack_inspect.py --source video.mp4 --conf-list 0.2,0.35,0.5 --no-display
"""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.egg_inspect import (  # type: ignore
        AppConfig,
        ClassicalConfig,
        ClassicalCrackDetector,
        CsvReport,
        JsonSummary,
        Renderer,
        YoloCrackDetector,
        analyze_frame,
        load_config,
        open_source,
    )
else:
    from .src.egg_inspect import (  # type: ignore
        AppConfig,
        ClassicalConfig,
        ClassicalCrackDetector,
        CsvReport,
        JsonSummary,
        Renderer,
        YoloCrackDetector,
        analyze_frame,
        load_config,
        open_source,
    )


VALID_MODES = ("auto", "yolo", "classical", "hybrid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Egg Crack / Quality Inspection System")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--source", type=str, default=None, help="Override video.source")
    p.add_argument("--weights", type=str, default=None, help="Override model.weights (YOLOv8-seg)")
    p.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default=None,
        help="Detector mode: auto|yolo|classical|hybrid",
    )
    p.add_argument(
        "--classical-egg-weights",
        type=str,
        default=None,
        help="Path to vendored YOLOv5 weights used by classical/hybrid modes",
    )
    p.add_argument("--conf", type=float, default=None, help="Override model.conf_threshold")
    p.add_argument("--iou", type=float, default=None, help="Override model.iou_threshold")
    p.add_argument(
        "--crack-only",
        action="store_true",
        help="Drop non-crack findings before stats / rendering",
    )
    p.add_argument(
        "--conf-list",
        type=str,
        default=None,
        help="Comma-separated conf thresholds to sweep",
    )
    p.add_argument(
        "--iou-list",
        type=str,
        default=None,
        help="Comma-separated IoU thresholds to sweep (paired with --conf-list)",
    )
    p.add_argument("--no-display", action="store_true", help="Disable preview window")
    p.add_argument("--save-video", type=str, default=None, help="Path to annotated video")
    p.add_argument("--csv", type=str, default=None, help="Path to per-frame CSV")
    p.add_argument("--summary", type=str, default=None, help="Path to JSON summary")
    p.add_argument("--loop", action="store_true", help="Loop video files (ignored for webcam)")
    return p.parse_args()


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source is not None:
        cfg.video.source = args.source
    if args.weights is not None:
        cfg.model.weights = args.weights
    if args.mode is not None:
        cfg.mode = args.mode
    if args.classical_egg_weights is not None:
        cfg.classical_egg_weights = args.classical_egg_weights
    if args.conf is not None:
        cfg.model.conf_threshold = args.conf
    if args.iou is not None:
        cfg.model.iou_threshold = args.iou
    if args.no_display:
        cfg.output.show_window = False
    if args.save_video is not None:
        cfg.output.save_video = args.save_video
    if args.csv is not None:
        cfg.output.csv_path = args.csv
    if args.summary is not None:
        cfg.output.summary_path = args.summary
    if args.loop:
        cfg.video.loop = True
    if args.crack_only:
        cfg.filter.crack_only = True
    return cfg


def _parse_floats(s: Optional[str]) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _pair_sweeps(conf_list: List[float], iou_list: List[float], cfg: AppConfig):
    if not conf_list and not iou_list:
        return []
    if not conf_list:
        conf_list = [cfg.model.conf_threshold]
    if not iou_list:
        iou_list = [cfg.model.iou_threshold]
    if len(iou_list) == 1 and len(conf_list) > 1:
        iou_list = iou_list * len(conf_list)
    if len(conf_list) == 1 and len(iou_list) > 1:
        conf_list = conf_list * len(iou_list)
    if len(conf_list) != len(iou_list):
        raise ValueError("--conf-list and --iou-list must align in length")
    return list(zip(conf_list, iou_list))


def _resolve_weights(cfg: AppConfig) -> Path:
    """Resolve the configured YOLOv8-seg weights to an absolute path."""
    p = Path(cfg.model.weights)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _resolve_classical_egg_weights(cfg: AppConfig) -> Path:
    p = Path(cfg.classical_egg_weights)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def build_detector(cfg: AppConfig, mode: Optional[str] = None):
    """Construct the detector described by ``mode`` (or ``cfg.mode``)."""
    use_mode = (mode or cfg.mode or "auto").lower()
    if use_mode not in VALID_MODES:
        raise ValueError(f"unknown mode: {use_mode}")

    yolo_weights = _resolve_weights(cfg)
    has_yolo = yolo_weights.exists()

    if use_mode == "yolo":
        if not has_yolo:
            raise FileNotFoundError(f"YOLOv8-seg weights not found: {yolo_weights}")
        print(f"[crack-inspect] mode=yolo -> {yolo_weights}")
        return YoloCrackDetector(
            weights=str(yolo_weights),
            device=cfg.model.device,
            img_size=cfg.model.img_size,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            classes=cfg.model.classes,
            task=cfg.model.task,
        )

    classical_weights = _resolve_classical_egg_weights(cfg)
    has_classical_weights = classical_weights.exists()

    if use_mode == "classical" or use_mode == "hybrid":
        print(
            f"[crack-inspect] mode={use_mode} -> "
            f"YOLOv5 ({classical_weights if has_classical_weights else 'NOT FOUND'}) + OpenCV"
        )
        return ClassicalCrackDetector(
            weights=str(classical_weights) if has_classical_weights else str(classical_weights),
            classical_config=ClassicalConfig(**{k: v for k, v in cfg.classical.__dict__.items() if k in ClassicalConfig.__dataclass_fields__}),
            device=cfg.model.device,
            img_size=cfg.model.img_size,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            classes=cfg.model.classes,
            project_root=classical_weights.parent,
        )

    # auto
    if has_yolo:
        print(f"[crack-inspect] mode=auto -> yolo ({yolo_weights})")
        return YoloCrackDetector(
            weights=str(yolo_weights),
            device=cfg.model.device,
            img_size=cfg.model.img_size,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            classes=cfg.model.classes,
            task=cfg.model.task,
        )
    print(
        f"[crack-inspect] mode=auto -> classical "
        f"(YOLOv5 weights: {classical_weights if has_classical_weights else 'NOT FOUND'})"
    )
    return ClassicalCrackDetector(
        weights=str(classical_weights),
        classical_config=ClassicalConfig(**{k: v for k, v in cfg.classical.__dict__.items() if k in ClassicalConfig.__dataclass_fields__}),
        device=cfg.model.device,
        img_size=cfg.model.img_size,
        conf_threshold=cfg.model.conf_threshold,
        iou_threshold=cfg.model.iou_threshold,
        classes=cfg.model.classes,
        project_root=classical_weights.parent,
    )


def run_once(
    cfg: AppConfig,
    detector,
    conf_value: float,
    iou_value: float,
    csv_path: Path,
    summary_path: Path,
    save_video_path: Optional[Path] = None,
) -> dict:
    if hasattr(detector, "set_conf_threshold"):
        detector.set_conf_threshold(conf_value)
    if hasattr(detector, "set_iou_threshold"):
        detector.set_iou_threshold(iou_value)
    renderer = Renderer(cfg.visual)
    csv_report = CsvReport(str(csv_path))
    summary = JsonSummary(str(summary_path), meta={"conf": conf_value, "iou": iou_value})

    writer = None
    frames_seen = 0
    total_eggs = 0
    total_cracked = 0

    try:
        with open_source(
            cfg.video.source,
            loop=cfg.video.loop,
            resize_width=cfg.video.resize_width,
            resize_height=cfg.video.resize_height,
        ) as src:
            print(
                f"[crack-inspect]   source={src.kind}({cfg.video.source}) "
                f"conf={conf_value:.3f} iou={iou_value:.3f} "
                f"csv={csv_path}"
            )
            while True:
                item = src.read()
                if item is None:
                    break
                frame, meta = item
                frames_seen = meta.index

                findings = detector.detect(frame)
                stats = analyze_frame(
                    findings,
                    frame_idx=meta.index,
                    timestamp=meta.timestamp,
                    min_mask_pixels=cfg.filter.min_mask_pixels,
                    min_bbox_area=cfg.filter.min_bbox_area,
                    crack_only=cfg.filter.crack_only,
                )
                total_eggs += stats.total
                total_cracked += stats.cracked
                csv_report.log(stats)
                summary.add(stats)

                if stats.cracked > 0:
                    print(
                        f"[crack-inspect]   frame={meta.index} "
                        f"cracked={stats.cracked}/{stats.total} "
                        f"rate={stats.cracked_ratio:.2%} "
                        f"avg_mask={stats.avg_crack_area_ratio:.2f}"
                    )

                canvas = renderer.draw(frame, findings, stats)
                if save_video_path is not None:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(save_video_path), fourcc, max(src.fps, 1.0), (w, h)
                        )
                    writer.write(canvas)
                if cfg.output.show_window:
                    cv2.imshow("Crack Inspection", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("r"):
                        print("[crack-inspect]   (reset not applicable to inspection)")
    finally:
        csv_report.close()
        summary.close()
        if writer is not None:
            writer.release()
        if cfg.output.show_window:
            cv2.destroyAllWindows()

    return {
        "conf": conf_value,
        "iou": iou_value,
        "frames": frames_seen,
        "total_eggs": total_eggs,
        "total_cracked": total_cracked,
        "crack_rate": (total_cracked / total_eggs) if total_eggs > 0 else 0.0,
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
    }


def print_summary_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = [
        ("conf", "conf", True),
        ("iou", "iou", True),
        ("total_eggs", "eggs", False),
        ("total_cracked", "cracked", False),
        ("crack_rate", "rate", True),
        ("frames", "frames", False),
    ]
    sep = "  "

    def fmt(col_key: str, is_float: bool, r: dict) -> str:
        v = r[col_key]
        if is_float:
            return f"{float(v):.3f}"
        return str(v)

    headers = [d for _, d, _ in columns]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, (k, _d, f) in enumerate(columns):
            widths[i] = max(widths[i], len(fmt(k, f, r)))
    line = sep.join(f"{{:<{w}}}" for w in widths)
    print("[crack-inspect] Threshold sweep summary")
    print(line.format(*headers))
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    for r in rows:
        print(line.format(*[fmt(k, f, r) for k, _d, f in columns]))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    try:
        detector = build_detector(cfg)
    except FileNotFoundError as exc:
        print(f"[crack-inspect] ERROR: {exc}")
        print(
            "Hint: drop best.pt next to crack_inspect.py, or pass --mode classical "
            "to run the OpenCV pipeline without YOLOv8 weights."
        )
        return 2
    print(f"[crack-inspect] Classes ({len(detector.class_names)}): {detector.class_names}")

    sweeps = _pair_sweeps(
        _parse_floats(args.conf_list), _parse_floats(args.iou_list), cfg
    )

    if not sweeps:
        csv_path = Path(cfg.output.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        save_video = Path(cfg.output.save_video) if cfg.output.save_video else None
        result = run_once(
            cfg,
            detector,
            cfg.model.conf_threshold,
            cfg.model.iou_threshold,
            csv_path=csv_path,
            summary_path=Path(cfg.output.summary_path),
            save_video_path=save_video,
        )
        print(
            f"[crack-inspect] Done. "
            f"eggs={result['total_eggs']} cracked={result['total_cracked']} "
            f"rate={result['crack_rate']:.2%} "
            f"(conf={result['conf']:.3f}, iou={result['iou']:.3f})"
        )
        return 0

    base_csv = Path(cfg.output.csv_path)
    sweep_dir = base_csv.parent / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "summary.csv"
    rows: List[dict] = []
    for conf_v, iou_v in sweeps:
        run_cfg = deepcopy(cfg)
        run_cfg.output.show_window = False
        run_cfg.output.csv_path = str(sweep_dir / f"conf_{conf_v:.3f}_iou_{iou_v:.3f}.csv")
        run_cfg.output.summary_path = str(sweep_dir / f"conf_{conf_v:.3f}_iou_{iou_v:.3f}.json")
        save_video = (
            sweep_dir / f"conf_{conf_v:.3f}_iou_{iou_v:.3f}.mp4"
            if cfg.output.save_video
            else None
        )
        result = run_once(
            run_cfg,
            detector,
            conf_v,
            iou_v,
            csv_path=Path(run_cfg.output.csv_path),
            summary_path=Path(run_cfg.output.summary_path),
            save_video_path=save_video,
        )
        rows.append(result)
        print(
            f"[crack-inspect]   -> conf={result['conf']:.3f} iou={result['iou']:.3f} "
            f"eggs={result['total_eggs']} cracked={result['total_cracked']} "
            f"rate={result['crack_rate']:.2%}"
        )

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["conf", "iou", "frames", "total_eggs", "total_cracked", "crack_rate", "csv_path", "summary_path"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print_summary_table(rows)
    print(f"[crack-inspect] Sweep summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())