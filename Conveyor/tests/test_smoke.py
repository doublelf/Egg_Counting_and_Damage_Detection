"""Smoke test for the Conveyor pipeline.

Uses stub detector + inspector to verify end-to-end flow without any
real YOLO weights. Mirrors the egg_line smoke test pattern.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egg_pipeline import (  # noqa: E402
    AssociationConfig,
    CombinedOverlay,
    ConveyorConfig,
    CounterConfig,
    CounterLineConfig,
    CounterModelConfig,
    CounterTrackerConfig,
    InspectorConfig,
    InspectorFilterConfig,
    InspectorModelConfig,
    JsonSummary,
    OutputConfig,
    VideoConfig,
    associate,
    load_config,
    run,
)


class StubCounterDetector:
    """Emits detections for 3 tracks across the configured line."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        from egg_counter import Detection as CDet

        out = []
        y = {
            1: 260, 2: 245, 3: 230, 4: 220, 5: 240, 6: 260,
            7: 260, 8: 240, 9: 220, 10: 260, 11: 240, 12: 220, 13: 260,
        }.get(self.calls, 250)
        for i, x0 in enumerate([200, 300, 400]):
            if self.calls in (1, 2, 3, 4, 5, 6) and i == 0:
                out.append(CDet(-1, 0, "egg", 0.9, x0 - 25, y - 30, x0 + 25, y + 30))
            if self.calls in (4, 5, 6, 7, 8, 9) and i == 1:
                out.append(CDet(-1, 0, "egg", 0.9, x0 - 25, y - 30, x0 + 25, y + 30))
            if self.calls in (7, 8, 9, 10, 11, 12) and i == 2:
                out.append(CDet(-1, 0, "egg", 0.9, x0 - 25, y - 30, x0 + 25, y + 30))
        return out


class StubInspectorDetector:
    """Returns one crack finding per frame matching track 1 and one intact for track 2."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        from egg_inspect import CrackFinding

        out = []
        h, w = frame.shape[:2]
        if 1 <= self.calls <= 6:
            mask = np.zeros((h, w), dtype=bool)
            mask[220: 300,  190: 250] = True
            coords = np.argwhere(mask)[: int(60 * 80 * 0.45)]
            mask[:] = False
            if len(coords):
                mask[coords[:, 0], coords[:, 1]] = True
            out.append(CrackFinding(
                cls_id=2, name="brown-egg-crack", conf=0.95,
                x_min=190, y_min=220, x_max=250, y_max=300,
                mask=mask, bbox_area=4800, mask_area=len(coords),
                mask_area_ratio=0.17,
            ))
        if 4 <= self.calls <= 9:
            mask = np.zeros((h, w), dtype=bool)
            mask[220: 300,  290: 350] = True
            coords = np.argwhere(mask)[: int(60 * 80 * 0.0)]
            mask[:] = False
            if len(coords):
                mask[coords[:, 0], coords[:, 1]] = True
            out.append(CrackFinding(
                cls_id=3, name="white-egg", conf=0.95,
                x_min=290, y_min=220, x_max=350, y_max=300,
                mask=mask, bbox_area=4800, mask_area=len(coords),
                mask_area_ratio=0.0,
            ))
        return out


def test_associator_direct():
    f = StubInspectorDetector().detect.__self__  # not great; just sanity:
    from egg_inspect import CrackFinding
    findings = [
        CrackFinding(cls_id=2, name="brown-egg-crack", conf=0.95,
                     x_min=190, y_min=220, x_max=250, y_max=300,
                     mask=np.zeros((480, 640), dtype=bool),
                     bbox_area=4800, mask_area=0, mask_area_ratio=0.0),
    ]
    v = associate((180, 210, 260, 300), findings, min_iou=0.3)
    assert v.color == "brown"
    assert v.cracked is True
    print(f"[smoke] associator direct: color={v.color} cracked={v.cracked}")

    far = associate((0, 0, 10, 10), findings, min_iou=0.3)
    assert far.color == "unknown"
    print("[smoke] associator no-match: verdict UNKNOWN OK")


def test_load_config():
    cfg = load_config(None)
    assert cfg.counter.line.direction == "down"
    assert cfg.counter.model.conf_threshold == 0.35
    assert cfg.inspector.model.mode == "classical"
    assert cfg.association.min_iou == 0.3
    print(f"[smoke] load_config OK: counter direction={cfg.counter.line.direction} inspector mode={cfg.inspector.model.mode}")


def test_pipeline_end_to_end(tmp_dir: Path):
    cfg = ConveyorConfig()
    cfg.video.source = str(tmp_dir)
    cfg.counter.line.start = [100, 240]
    cfg.counter.line.end = [540, 240]
    cfg.counter.line.direction = "down"
    cfg.output.frames_csv = str(tmp_dir / "frames.csv")
    cfg.output.crossings_csv = str(tmp_dir / "crossings.csv")
    cfg.output.summary_json = str(tmp_dir / "summary.json")
    cfg.output.show_window = False

    import egg_pipeline.pipeline as pm

    class _FakeMeta:
        def __init__(self, idx, ts):
            self.index = idx
            self.timestamp = ts

    class _FakeSource:
        kind = "synthetic"
        fps = 30.0

        def __init__(self):
            self._i = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            self._i += 1
            if self._i > 13:
                return None
            return np.zeros((480, 640, 3), dtype=np.uint8), _FakeMeta(self._i, self._i / 30.0)

    pm.open_source = lambda *a, **kw: _FakeSource()

    counter_det = StubCounterDetector()
    inspector_det = StubInspectorDetector()

    result = run(
        cfg,
        counter_detector=counter_det,
        inspector_detector=inspector_det,
    )

    assert result.total_crossings == 3, f"expected 3 crossings, got {result.total_crossings}"
    assert result.defective_count == 1, f"expected 1 defective, got {result.defective_count}"
    assert counter_det.calls == 13
    assert inspector_det.calls == 13

    with open(cfg.output.frames_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 13
    cross_counts = [int(r["crossings_this_frame"]) for r in rows]
    assert sum(cross_counts) == 3, f"frame crossings total wrong: {cross_counts}"
    print(f"[smoke] frames CSV OK: {len(rows)} rows, total crossings_this_frame={sum(cross_counts)}")

    with open(cfg.output.crossings_csv, newline="", encoding="utf-8") as fh:
        crows = list(csv.DictReader(fh))
    assert len(crows) == 3
    verdicts = sorted([(r["cls"], r["color"], r["cracked"], r["is_defective"]) for r in crows])
    expected = sorted([
        ("brown-egg-crack", "brown", "True", "True"),
        ("white-egg", "white", "False", "False"),
        ("unknown", "unknown", "False", "False"),
    ])
    assert verdicts == expected, f"verdicts mismatch: {verdicts}"
    print(f"[smoke] crossings CSV OK: verdicts={verdicts}")

    with open(cfg.output.summary_json, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["summary"]["total_crossings"] == 3
    assert summary["summary"]["defective_count"] == 1
    assert summary["summary"]["color_split"]["brown"] == 1
    assert summary["summary"]["color_split"]["white"] == 1
    print(f"[smoke] summary JSON OK: {summary['summary']}")


def main() -> int:
    tmp_dir = Path(__file__).resolve().parent / "_smoke_tmp"
    tmp_dir.mkdir(exist_ok=True)
    test_associator_direct()
    test_load_config()
    test_pipeline_end_to_end(tmp_dir)
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())