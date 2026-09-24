# Egg Crack / Quality Inspection (YOLOv8-seg + classical CV, local)

Two interchangeable detectors, one CLI:

- **YOLOv8-seg** (`mode=yolo` or default when `best.pt` exists): full
  instance-segmentation model trained via `train_inspector.py`.
- **Hybrid classical** (`mode=classical` / `mode=hybrid`): vendored
  **YOLOv5** finds eggs, **OpenCV** (CLAHE → bilateral → black-hat
  morphology → adaptive threshold → connected components → shape
  classification) decides if each egg is *intact*, *cracked*, or *dirty*.
  Requires no cracked-egg training data; colour via Lab a*.

The classical mode exists because the crackedChickenEggs dataset is hard
to obtain. It runs out-of-the-box against the upstream demo PNGs in
`../crackedChickenEggs161/` and produces results like:

```
[crack-inspect] mode=classical -> YOLOv5 (.../best.pt) + OpenCV
...
[crack-inspect]   frame=1 cracked=1/1 rate=100.00% avg_mask=0.00
[crack-inspect] Done. eggs=6 cracked=5 rate=83.33% (conf=0.350, iou=0.450)
```

This repo includes both the **inference pipeline** and a **training
pipeline** that produces `best.pt` from scratch (no upstream weights
required). A small synthetic demo dataset is generated locally so the
whole loop can be exercised end-to-end without external data.

## Features

- YOLOv8-seg inference via the installed `ultralytics` package.
- 6-class model (`brown-egg`, `brown-egg-crack`, `brown-egg-dirty`,
  `white-egg`, `white-egg-crack`, `white-egg-dirty`).
- Hybrid YOLOv5 + classical-CV mode (no cracked-egg dataset required).
- Per-frame stats: cracked / intact / dirty counts, cracked ratio, mean
  and max crack mask-area ratio.
- Mask overlay + bounding boxes + per-class stable colours.
- CSV per-frame log + JSON run summary.
- `--conf-list` to sweep multiple confidence thresholds.
- Training script (`train_inspector.py`) with resume + auto best.pt copy.
- Synthetic dataset generator (`prepare_data.py --make-demo`) for
  pipeline self-testing.

## Quick start

### 1. Inference without any cracked-egg weights

```bash
# Use the vendored YOLOv5 to find eggs + OpenCV to classify cracks.
# No crackedChickenEggs dataset or best.pt needed.
python crack_inspect.py --mode classical --source ../crackedChickenEggs161 --no-display
python crack_inspect.py --mode classical --source 0
python crack_inspect.py --mode classical --source video.mp4 --no-display --save-video output/run.mp4
```

### 2. Inference with a trained YOLOv8-seg `best.pt`

```bash
# Auto-picks yolo mode when best.pt exists.
python crack_inspect.py --source 0
python crack_inspect.py --source video.mp4 --no-display
python crack_inspect.py --mode yolo --source frames/ --save-video output/run.mp4
```

### 3. Train your own model

```bash
# Generate a small synthetic demo dataset (60 train + 12 val images)
python prepare_data.py --make-demo --out data/cracked_eggs_demo

# Train (downloads yolov8n-seg.pt automatically, ~7 MB)
python train_inspector.py --data data/cracked_eggs_demo/data.yaml \
                          --weights yolov8n-seg.pt \
                          --epochs 50 --imgsz 640 --batch 8

# best.pt is auto-copied to ./best.pt
```

For real production accuracy, replace the demo dataset with the
crackedChickenEggs dataset (~4900 images) exported in YOLOv8-seg
format. See "Training on your own data" below.

### 3. Evaluate

```bash
python evaluate.py --weights best.pt --data data.yaml
```

### 4. Visual smoke test on upstream demo PNGs

```bash
python infer_demo.py                       # runs on ../crackedChickenEggs161/*.png
python infer_demo.py --source /path/to/images --out output/my_demo
```

### 5. One-shot local-video / webcam runner

`infer_local_video.py` is a thin convenience wrapper around
`crack_inspect.py` that defaults to **classical mode** (no
crackedChickenEggs weights required) and auto-wires the output paths
under `output/local_infer/`. It accepts a file path, a webcam index,
or a Tkinter file picker.

```bash
# Pick a file interactively
python infer_local_video.py --picker

# Explicit local video
python infer_local_video.py --video path/to/run.mp4

# Webcam shortcut (equivalent to ``python crack_inspect.py --source 0``)
python infer_local_video.py --webcam
python infer_local_video.py --webcam 1     # secondary camera

# Headless run with annotated mp4 + CSV + JSON
python infer_local_video.py --video run.mp4 --no-display \
                            --output-dir output/inspection \
                            --save-video annotated.mp4

# Inspect the command before executing
python infer_local_video.py --video run.mp4 --dry-run
```

Outputs land in `output/local_infer/`:

- `annotated.mp4` — preview with mask overlay (when not `--no-display`)
- `counts.csv` — per-frame cracked / intact / dirty counts
- `summary.json` — totals + class histogram

## Training on your own data

The pipeline expects a standard YOLOv8-seg layout:

```
<root>/
├── data.yaml            # path, train/val/test (relative), nc, names
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt   # YOLO-seg polygons
```

`prepare_data.py --validate <path>` confirms the layout and prints a
class histogram. The six default classes are:

```yaml
names:
  - brow-egg-dirty
  - brown-egg
  - brown-egg-crack
  - white-egg
  - white-egg-crack
  - white-egg-dirty
```

If you have access to the **crackedChickenEggs** dataset (4900 images):

1. Export it from Roboflow / CVAT in YOLOv8-seg format.
2. Place the resulting folder at e.g. `data/cracked_eggs/`.
3. Train:

   ```bash
   python train_inspector.py --data data/cracked_eggs/data.yaml \
                             --weights yolov8s-seg.pt \
                             --epochs 100 --imgsz 640 --batch 16
   ```

   `yolov8s-seg.pt` will be downloaded automatically by ultralytics
   (~22 MB). Use `yolov8n-seg.pt` (~7 MB) for faster training on CPU.

4. The script copies the resulting `best.pt` to `./best.pt`. Pass it to
   `crack_inspect.py --weights best.pt`.

### Resuming training

```bash
python train_inspector.py --data data.yaml --weights runs_seg/train/weights/last.pt --resume
```

## Project layout

```
crack_inspect/
├── best.pt                  # Trained weights (produced by train_inspector.py)
├── config.yaml              # Inference defaults
├── crack_inspect.py         # CLI inference entry
├── train_inspector.py       # CLI training entry (ultralytics YOLO.train)
├── prepare_data.py          # Demo data generator + dataset validator
├── evaluate.py              # Validation + per-class metrics
├── infer_demo.py            # Inference on upstream demo PNGs (no labels)
├── download_pretrained.py   # Best-effort public weights downloader
├── data/                    # Created by prepare_data.py
│   └── cracked_eggs_demo/   # Synthetic demo dataset (gitignored)
├── src/egg_inspect/
│   ├── __init__.py
│   ├── config.py            # AppConfig + YAML loader
│   ├── detector.py          # Ultralytics YOLO(seg) -> CrackFinding
│   ├── analyzer.py          # FrameStats + summarize
│   ├── renderer.py          # mask overlay + bbox + HUD
│   ├── report.py            # CSV + JSON
│   └── video.py             # webcam / video / image-folder source
├── tests/test_smoke.py      # analyzer / renderer without weights
├── logs/, output/, runs_seg/
└── README.md
```

## Smoke test (no weights required)

```bash
python tests/test_smoke.py
```

Validates `analyzer`, `summarize`, `Renderer` against synthetic
`CrackFinding` instances.

## End-to-end pipeline self-test

The whole loop can be exercised with **zero external data**:

```bash
python prepare_data.py --make-demo --out data/cracked_eggs_demo
python train_inspector.py --data data/cracked_eggs_demo/data.yaml \
                          --weights yolov8n-seg.pt \
                          --epochs 5 --imgsz 320 --batch 4 \
                          --device cpu --name self_test
python crack_inspect.py --source data/cracked_eggs_demo/images/val \
                        --weights best.pt --no-display
python evaluate.py --weights best.pt --data data/cracked_eggs_demo/data.yaml
```

5 epochs on synthetic data will not produce useful predictions — but it
proves every step works.

## Hardware notes

- CPU is sufficient for the demo dataset (≈ 5–15 s/epoch for yolov8n,
  640×640, 4-image batch).
- For real training (≥ 50 epochs on 4900 images): NVIDIA GPU strongly
  recommended.
- macOS / Linux / Windows all work via the ultralytics package.

## Pulling a pretrained model

If you don't want to train, you can try:

```bash
python download_pretrained.py --out best.pt
```

This iterates a small list of community / academic release URLs. If
none are reachable from your network, the script exits non-zero and
prints "Train your own weights via train_inspector.py."

The original upstream repo (`crackedChickenEggs161/`) deliberately does
not publish weights, so a real training run is the most reliable path
unless you have access to a published checkpoint elsewhere.

## Differences from the upstream repo

| | `crackedChickenEggs161/` (upstream) | `crack_inspect/` (this folder) |
|---|---|---|
| `best.pt` shipped | No | No (produced by `train_inspector.py`) |
| Web UI (`web.py`) | Yes, streamlit | Not reproduced |
| Local CLI inference | No | Yes (`crack_inspect.py`) |
| Local CLI training | No | Yes (`train_inspector.py`) |
| Demo data generator | No | Yes (`prepare_data.py --make-demo`) |
| Validation script | No | Yes (`evaluate.py`) |
| Demo image infer | No | Yes (`infer_demo.py`) |
| Threshold sweep | Manual rerun | Yes (`--conf-list`) |
| CSV + JSON report | Manual export | Automatic per run |