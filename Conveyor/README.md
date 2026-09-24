# Conveyor (count + crack detection, independent project)

Combines the vendored YOLOv5-based **egg counter** from
`Automated-Egg-Counting-System/` with the vendored classical-CV crack
classifier from `crack_inspect/` into a single per-egg quality report.

For every frame:

```
   frame
       │
   ┌───┴───────────────────────────────┐
   │                                    │
 vendored YOLOv5                    vendored YOLOv5 + OpenCV
   (counter)                              (crack inspector)
   │                                    │
   Detection[]                       CrackFinding[]
   │                                    │
   CentroidTracker                       │
   │                                    │
   LineCrossingCounter                   │
   │                                    │
   CrossingEvent ──── IoU associator ───┘
   │
 EggVerdict{track_id, cls, cracked, dirty, intact, ...}
   │
   ├── CrossingsCsv (per-egg)
   ├── FramesCsv (per-frame)
   └── JsonSummary
```

The two upstream projects are **not modified**; Conveyor/ only adds
orchestration glue under `src/egg_pipeline/`.

## Features

- Vendored YOLOv5 detects eggs + tracks across frames (centroid tracker).
- Vendored classical CV (CLAHE → black-hat → adaptive threshold →
  connected components) classifies cracks / dirt / intact inside each ROI.
- IoU-based per-egg verdict at the moment of line crossing.
- Per-egg CSV (`crossings.csv`) with `color / cls / cracked / dirty /
  intact / is_defective / mask_area_ratio / iou / source`.
- Per-frame CSV + JSON summary.
- Combined overlay: counter line + counter bboxes + crack masks + HUD.

## Hardware

- CPU is fine for development; NVIDIA GPU accelerates both models.
- USB webcam / Pi Camera + consistent lighting over the belt.

## Software

- Python 3.9+
- `pip install torch torchvision opencv-python numpy Pillow PyYAML ultralytics`
- The vendored YOLOv5 repo at
  `../Automated-Egg-Counting-System/vendor_yolov5/` is required and is
  loaded automatically by `egg_pipeline/paths.py`.

## Quick start

```bash
# 1. Make sure the two upstream projects are present:
#    egg_detection/Automated-Egg-Counting-System/  (with best.pt + vendor_yolov5/)
#    egg_detection/crack_inspect/                   (used via vendored classical module)
#
# 2. Local video (defaults: classical inspector mode, no cracked-egg weights required)
python run.py --video path/to/run.mp4

# 3. Webcam
python run.py --source 0                # equivalent
python run.py --webcam

# 4. Headless with annotated mp4 + per-egg CSV
python run.py --video run.mp4 --no-display --save-video output/annotated.mp4

# 5. Custom line + thresholds
python run.py --video run.mp4 \
              --line-start 100 240 --line-end 540 240 --line-direction down \
              --conf-counter 0.35 --conf-inspector 0.25 \
              --associate-iou 0.3
```

## Outputs

- `output/annotated.mp4` (optional) — overlay with line + boxes + masks.
- `logs/frames.csv` — per-frame roll-up (in-frame counts, crossings).
- `logs/crossings.csv` — per-egg verdict at the moment of crossing.
- `logs/summary.json` — totals + per-color counts + defective rate.

`crossings.csv` columns:

| column | meaning |
|---|---|
| `frame_idx`, `timestamp`, `direction` | when + how the egg crossed |
| `track_id` | counter's stable ID for that egg |
| `color` | `white` / `brown` / `unknown` |
| `cls` | inspector class name (e.g. `brown-egg-crack`) |
| `cracked`, `dirty`, `intact` | derived booleans |
| `is_defective` | `cracked or dirty` |
| `mask_area_ratio` | fraction of the egg bbox covered by the crack mask |
| `iou` | IoU between the counter track bbox and the inspector bbox |
| `source` | `inspector` (matched) or `unknown` (no overlap) |

## Configuration

All defaults live in `config.yaml`. CLI flags override individual keys.
most
Most useful:
```yaml
counter:
  line:    start: [100, 240]
    end:   [540, 240]
    direction: down            # down | up | any
inspector:
  model:    mode: classical          # auto | classical | yolo
    weights: best.pt (YOLOv5)   # only classical mode requires this
association:
  min_iou: 0.3                   # cross-model bbox match threshold
```

The `min_iou` threshold filters weak associations; if the counter
track and the crack finding don't overlap at least this much at the
crossing moment, the egg is recorded as `unknown`. Lower it if your
two models produce systematically offset bboxes; raise it if you see
false-positive matches.

## Smoke test

```bash
python tests/test_smoke.py
```

Runs the full pipeline end-to-end with stub detector + stub inspector
(no weights required). Validates `associate`, `load_config`,
CSV/JSON outputs.

## CLI flags

```
python run.py --help
```

Useful ones:

```
--source / --video / --webcam     input source
--counter-weights                 path to YOLOv5 best.pt
--inspector-weights               path to YOLOv5 best.pt (classical) or YOLOv8 best.pt (yolo)
--inspector-mode                  auto | yolo | classical
--conf-counter / --conf-inspector  override confidence thresholds
--line-start X Y / --line-end X Y / --line-direction
--associate-iou                   cross-model match threshold
--no-display / --save-video / --frames-csv / --crossings-csv / --summary
--loop
```

## Differences from `egg_line/`

`egg_line/` is a similar but separate project that was created earlier
with the same goal. Conveyor/ is a cleaner reimplementation: smaller,
single package, no compatibility shim for the older `egg_line` API.
If you only need one of them, pick the one with the cleaner codebase
(Conveyor/).

## Limitations

- Egg detection still relies on the vendored YOLOv5 weights
  (`Automated-Egg-Counting-System/best.pt`).
- Crack classification uses heuristics (Black-hat + connected
  components) rather than a learned model — sensitivity depends on
  lighting and camera stability.
- 6-class colour discrimination (brown vs white) is done via Lab a*
  threshold, accurate for typical eggshell tones.