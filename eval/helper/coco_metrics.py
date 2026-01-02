from __future__ import annotations

from typing import Any, Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import Counter

def _coco_from_dict(coco_gt: Dict[str, Any]) -> COCO:
    # pycocotools COCO object from memory COCO dict.
    coco = COCO()
    coco.dataset = coco_gt
    coco.createIndex()
    return coco

def compute_coco_bbox_metrics(
    coco_gt: Dict[str, Any],
    preds: List[Dict[str, Any]],
) -> Dict[str, float]:
    # standard COCO bbox metrics
    # --- DIAGNOSTIC: how many GT annotations per image_id? ---
    anns = coco_gt.get("annotations", [])
    cnt = Counter(ann["image_id"] for ann in anns if "image_id" in ann)
    print("[INFO] max annotations per image_id:", max(cnt.values()) if cnt else 0)
    print("[INFO] #images with >1 annotation:", sum(v > 1 for v in cnt.values()))
    # --------------------------------------------------------


    coco = _coco_from_dict(coco_gt)
    coco_pd = coco.loadRes(preds)

    ev = COCOeval(coco, coco_pd, iouType="bbox")

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats

    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }

