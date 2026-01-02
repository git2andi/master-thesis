## Setup

Install Ultralytics via pip:

```bash
pip install ultralytics
```

## Usage

All configuration is provided via the CLI (no separate config files).

### Training (YOLO / RT-DETR)

```bash
yolo detect train \
  model=yolo11m.pt \
  data=/path/to/data.yaml \
  project=path/to/output/folder \
  name=outputname \
  epochs=100 \
  patience=10 \
  seed=42 \
  imgsz=640 \
  device=0,1 \
  save_period=1 \
  workers=16
```

### Evaluation (YOLO / RT-DETR)

```bash
yolo detect val \
  split=test \
  rect=False \
  iou=0.5 \
  imgsz=640 \
  data=/path/to/data.yaml \
  model=path/to/best/model \
  project=/path/output \
  name=outputName \
  save_json=true \
  batch=1
```

## RT-DETR Notes (evaluation only)

For RT-DETR, validation may require a patched Ultralytics version to avoid issues when writing `predictions.json`. Training and validation logic are not affected; the fix only ensures correct padding/coordinate handling in the exported JSON.
Create a separate Conda environment and install the RT-DETR branch used for evaluation:

```bash
pip install git+https://github.com/ultralytics/ultralytics@rtdetr-transform
```

To run RT-DETR evaluation with this fix, use:

* `u_eval_rtdetr.py` (located in this folder)

After RT-DETR evaluation, reduce the exported predictions to the top-100 per image using:

* `filter_predictions.py`
