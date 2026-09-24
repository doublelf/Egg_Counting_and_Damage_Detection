"""Main pipeline: read frames, run counter + crack inspector in parallel,
associate per-egg verdicts at line-crossing moments, render a unified
view, and write per-frame + per-crossing reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from egg_inspect import open_source

from .associator import EggVerdict, associate
from .config import ConveyorConfig, config_paths
from .counter import build_counter
from .inspector import build_inspector
from .overlay import CombinedOverlay, OverlayStats
from .report import CrossingsCsv, FramesCsv, JsonSummary


@dataclass
class PipelineResult:
    total_crossings: int = 0
    defective_count: int = 0
    frames_seen: int = 0
    csv_paths: dict = None


def _track_bbox(track) -> Optional[tuple]:
    boxes = getattr(track, "boxes", None)
    if not boxes:
        return None
    b = boxes[-1]
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _build_overlay(cfg: ConveyorConfig) -> CombinedOverlay:
    return CombinedOverlay(
        line_start=tuple(cfg.counter.line.start),
        line_end=tuple(cfg.counter.line.end),
        line_direction=cfg.counter.line.direction,
        mask_alpha=0.45,
        box_thickness=2,
        show_mask=True,
        show_box=True,
    )


def _filter_inspector_findings(findings, cfg: ConveyorConfig) -> list:
    out = []
    for f in findings:
        if f.mask_area < cfg.inspector.filter.min_mask_pixels:
            continue
        if f.bbox_area < cfg.inspector.filter.min_bbox_area:
            continue
        if cfg.inspector.filter.crack_only and not _is_crack_name(f.name):
            continue
        out.append(f)
    return out


def _is_crack_name(name: str) -> bool:
    return name.endswith("-crack")


def _frame_inspector_stats(findings) -> dict:
    cracked = sum(1 for f in findings if _is_crack_name(f.name))
    intact = sum(1 for f in findings if f.name in ("brown-egg", "white-egg"))
    dirty = sum(1 for f in findings if f.name.endswith("-dirty"))
    return {
        "cracked": cracked,
        "intact": intact,
        "dirty": dirty,
        "cracked_ratio": (cracked / len(findings)) if findings else 0.0,
    }


def run(
    cfg: ConveyorConfig,
    counter_detector=None,
    inspector_detector=None,
    overlay: Optional[CombinedOverlay] = None,
    counter_tracker=None,
    counter=None,
    frames_csv: Optional[FramesCsv] = None,
    crossings_csv: Optional[CrossingsCsv] = None,
    json_summary: Optional[JsonSummary] = None,
    writer: Optional[cv2.VideoWriter] = None,
    show_window: Optional[bool] = None,
) -> PipelineResult:
    """Execute one full pass over the configured video source."""
    paths = config_paths(cfg)
    show_window = cfg.output.show_window if show_window is None else show_window

    # --- lazy init collaborators ---
    if counter_detector is None or counter_tracker is None or counter is None:
        cd, ct, cn = build_counter(cfg.counter, paths["counter_weights"])
        counter_detector = counter_detector or cd
        counter_tracker = counter_tracker or ct
        counter = counter or cn

    if inspector_detector is None:
        inspector_detector, mode_label = build_inspector(cfg.inspector, paths["inspector_weights"])
    else:
        mode_label = getattr(inspector_detector, "_mode_label", "unknown")

    overlay = overlay or _build_overlay(cfg)
    frames_csv = frames_csv or FramesCsv(paths["frames_csv"])
    crossings_csv = crossings_csv or CrossingsCsv(paths["crossings_csv"])
    json_summary = json_summary or JsonSummary(
        paths["summary_json"],
        meta={
            "counter_weights": str(paths["counter_weights"]),
            "inspector_weights": str(paths["inspector_weights"]),
            "inspector_mode": mode_label,
            "line": {
                "start": list(cfg.counter.line.start),
                "end": list(cfg.counter.line.end),
                "direction": cfg.counter.line.direction,
            },
        },
    )

    writer_local = writer
    save_video_path = paths["save_video"]
    frames_seen = 0
    crossings_total = 0
    defective_total = 0

    try:
        with open_source(
            cfg.video.source,
            loop=cfg.video.loop,
            resize_width=cfg.video.resize_width,
            resize_height=cfg.video.resize_height,
        ) as src:
            print(
                f"[conveyor] source={src.kind}({cfg.video.source}) "
                f"counter={paths['counter_weights'].name} "
                f"inspector[{mode_label}]={paths['inspector_weights'].name} "
                f"min_iou={cfg.association.min_iou}"
            )
            while True:
                item = src.read()
                if item is None:
                    break
                frame, meta = item
                frames_seen = meta.index

                counter_dets = counter_detector.detect(frame)
                crack_raw = inspector_detector.detect(frame)
                crack_filtered = _filter_inspector_findings(crack_raw, cfg)
                ins_stats = _frame_inspector_stats(crack_filtered)

                tracks = counter_tracker.update(counter_dets)
                events = counter.update(tracks, meta.index)

                for ev in events:
                    crossings_total += 1
                    tr = counter_tracker.tracks.get(ev.track_id)
                    bbox = _track_bbox(tr) if tr is not None else None
                    verdict = (
                        associate(bbox, crack_filtered, min_iou=cfg.association.min_iou)
                        if bbox is not None
                        else EggVerdict()
                    )
                    crossings_csv.log(meta.index, meta.timestamp, ev.track_id, ev.direction, verdict)
                    json_summary.record_crossing(verdict)
                    if verdict.is_defective:
                        defective_total += 1
                    print(
                        f"[conveyor] frame={meta.index} track={ev.track_id} "
                        f"{ev.direction} -> {verdict.cls_name} "
                        f"(defective={verdict.is_defective}, iou={verdict.iou:.2f})"
                    )
                   
                frames_csv.log(
                    {
                        "frame_idx": meta.index,
                        "timestamp": round(meta.timestamp, 3),
                        "in_frame_eggs": len(tracks),
                        "cracked_in_frame": ins_stats["cracked"],
                        "intact_in_frame": ins_stats["intact"],
                        "dirty_in_frame": ins_stats["dirty"],
                        "cracked_ratio_in_frame": round(ins_stats["cracked_ratio"], 4),
                        "crossings_this_frame": len(events),
                    }
                )

                overlay_stats = OverlayStats(
                    in_frame=len(tracks),
                    cracked=ins_stats["cracked"],
                    intact=ins_stats["intact"],
                    dirty=ins_stats["dirty"],
                    defective_count=defective_total,
                    total_crossings=crossings_total,
                )
                canvas = overlay.draw(frame, tracks, crack_filtered, overlay_stats, show_window=show_window)

                if save_video_path is not None:
                    if writer_local is None:
                        h, w = canvas.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer_local = cv2.VideoWriter(
                            str(save_video_path), fourcc, max(src.fps, 1.0), (w, h)
                        )
                    writer_local.write(canvas)

                if show_window:
                    cv2.imshow("Conveyor", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("r"):
                        counter.total = 0
                        for tr in counter_tracker.tracks.values():
                            tr.crossed = False
                        print("[conveyor] counter reset")
    finally:
        frames_csv.close()
        crossings_csv.close()
        json_summary.close(extra={"frames_seen": frames_seen})
        if writer_local is not None:
            writer_local.release()
        if show_window:
            cv2.destroyAllWindows()

    return PipelineResult(
        total_crossings=crossings_total,
        defective_count=defective_total,
        frames_seen=frames_seen,
        csv_paths={
            "frames_csv": str(paths["frames_csv"]),
            "crossings_csv": str(paths["crossings_csv"]),
            "summary_json": str(paths["summary_json"]),
        },
    )