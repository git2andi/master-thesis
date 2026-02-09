# Master Thesis Code Repository



This thesis evaluates the applicability of modern real-time object detection algorithms for automated polyp detection in full-length clinical colonoscopy videos. While recent studies report high detection performance on curated image datasets or short video clips, these evaluation settings do not reflect real clinical conditions, where negative frames dominate and visual appearance varies strongly over time. As a result, reported performance often overestimates the effectiveness of computer-aided detection systems in practice.

Using the recently released REAL-Colon dataset, which consists of complete, clinical-grade colonoscopy procedures with frame- and lesion-level annotations, this work establishes a standardized and reproducible evaluation framework for real-time polyp detection under realistic conditions. Four representative detection architectures, namely Faster R-CNN, YOLOv8, YOLOv11, and RT-DETR, are systematically assessed using detection-level, frame-level, and lesion-level metrics to capture clinically relevant aspects such as false-positive behavior, temporal consistency, detection latency, and real-time suitability.

The results show that detection performance drops substantially when models are evaluated on full procedures compared to curated benchmarks. Transformer-based models demonstrate higher sensitivity and temporal stability, whereas CNN-based detectors offer higher throughput and specificity. Importantly, performance under realistic conditions remains insufficient for fully reliable clinical deployment, underscoring the need for more sophisticated modeling and evaluation approaches.



This repository contains the code developed for my Master’s thesis. 

## Structure

- [Done] **`preprocess/`**
  Dataset setup, filtering, and annotation preparation.

- [Done] **`code/`**
  Dataset Visualization and plotting.

- [Done] **`detectron2/`**  
  Faster R-CNN training and inference using Detectron2.

- [Done] **`ultralytics/`**
  YOLO and RT-DETR training and inference using the Ultralytics framework.

- [Done] **`eval/`**
  Central evaluation pipeline for all models.

## Notes

- Datasets are **not included** in this repository.
- Paths, hardware settings, and environment configurations are system-specific.
- The code reflects an experimental research workflow and prioritizes transparency and reproducibility.
