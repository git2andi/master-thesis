## REAL-Colon preprocessing (run in order)

* **`1_export_coco_format_diff_neg.py`**
  Based on the original code. Exports REAL-Colon annotations in COCO format and performs negative-frame sampling.
  Configure the following variables inside the script before running:

  * `base_dataset_folder`   (path to original REAL-Colon)
  * `negative_ratio`        (ADAPTED, now samples negatives based on Positives)
  * `output_folder`         (output dataset path)

* **`2_yolo_from_coco.py`**
  Converts the COCO export into a YOLO-style dataset layout.
  Configure inside the script:

  * `DATASET_ROOT` (set to the `output_folder` from step 1)

* **`3_materialize_syslinks.py`**
  Finalizes the dataset by materializing symlinks into physical files.
  Configure inside the script:

  * `DATASET_ROOT` (same as step 1)
  
## PICCOLO preprocessing

* **`prepare_piccolo.py`**
  Converts PICCOLO into COCO/YOLO formats and writes a new folder `piccolo_split/` into the chosen output root.
Run:

```bash
python prepare_piccolo.py \
  --src /data/local/aschwab/data/piccolo \
  --dst_root /data/local/aschwab/data
```

## SUN preprocessing
* **`prepare_sun.py`**
  Converts SUN cases into COCO/YOLO formats and writes `sun_split/` and `sun_full/` into the chosen output root.

Run:

```bash
python prepare_sun.py \
  --src /data/local/aschwab/data/sun \
  --dst_root /data/local/aschwab/data
```
