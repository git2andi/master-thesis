This folder contains the main prediction evaluation pipeline and plotting utilities.

## Main: `run_eval.py`

All evaluations are executed via `run_eval.py`:

```bash
python run_eval.py \
  --gt path/to/gt.json \
  --pred path/to/pred.json \
  --framework ultralytics \
  --dataset realcolon
```

Supported options:

* `--framework`: `ultralytics` or `detectron2`
* `--dataset`: `realcolon`, `sun`, or `piccolo`
* `--conf`: optional confidence threshold (default: `0.2`)

Each run writes its outputs into the `results/` folder.

## Aggregation: `summary_results.py`

Aggregates multiple evaluation runs (3 seeds per model)
Paths must be set manually in the script via `MODEL_RESULTS` (results.json)

* `Faster R-CNN`: 3 result files (seeded)
* `YOLOv8`: 3 result files (seeded)
* `YOLOv11`: 3 result files (seeded)
* `RT-DETR`: 3 result files (seeded)

The script generates a `.tsv` summary table.
`make_table.py` is a helper script to extract selected values from this `.tsv` (easy latex copy stuff for all the tables)

## Plotting

* **`plot_froc_afroc.py`**
  Plots aggregated FROC/AFROC curves across seeds per model.
  Paths to the per-seed outputs must be set inside the script. (curves.json)

* **`plot_lesions_frames.py`**
  Plots lesion-level detection statistics across seeds per model and stores figures in `lesion_barplots_seeded/`.
  Paths to the per-seed outputs must be set inside the script. (curves.json)
