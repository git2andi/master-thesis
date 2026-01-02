from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetEvalConfig:
    name: str
    iou_thr: float
    conf: float

    do_detection_metrics: bool
    do_frame_metrics: bool
    do_frame_sweep: bool

    do_lesion_metrics: bool
    lesion_enable_timing: bool

    do_froc: bool
    do_afroc: bool


DATASET_CFG: Dict[str, DatasetEvalConfig] = {
    "realcolon": DatasetEvalConfig(
        name="realcolon",
        iou_thr=0.0,
        conf=0.2,
        do_detection_metrics=True,
        do_frame_metrics=True,
        do_frame_sweep=True,
        do_lesion_metrics=True,
        lesion_enable_timing=True,
        do_froc=True,
        do_afroc=True,
    ),
    "sun": DatasetEvalConfig(
        name="sun",
        iou_thr=0.3,
        conf=0.1,
        do_detection_metrics=True,
        do_frame_metrics=True,
        do_frame_sweep=True,
        do_lesion_metrics=True,
        lesion_enable_timing=False,
        do_froc=False,
        do_afroc=False,
    ),
    "piccolo": DatasetEvalConfig(
        name="piccolo",
        iou_thr=0.3,
        conf=0.1,
        do_detection_metrics=True,
        do_frame_metrics=True,
        do_frame_sweep=True,
        do_lesion_metrics=False,
        lesion_enable_timing=False,
        do_froc=False,
        do_afroc=False,
    ),
}


def get_dataset_cfg(dataset: str) -> DatasetEvalConfig:
    return DATASET_CFG[dataset]

