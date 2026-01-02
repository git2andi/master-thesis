
# `code/`

This folder contains utilities for dataset inspection, visualization, and qualitative verification.

## Contents

* **`get_dataset_info_piccolo.py`**
  Computes dataset statistics and consistency checks for the PICCOLO dataset after preprocessing and split conversion.

* **`get_dataset_info_sun.py`**
  Reports split-wise image and label counts, per-case distributions, and positive/negative case statistics for the SUN dataset.

* **`get_dataset_info_realcolon.py`**
  Summarizes image resolution and aspect-ratio distributions across REAL-Colon dataset splits.

* **`verify_piccolo.py`**
  Visualizes example PICCOLO images with ground-truth annotations to verify that newly created splits align with the original dataset.

* **`verify_sun.py`**
  Visualizes example SUN images with ground-truth annotations to verify that newly created splits align with the original dataset.


