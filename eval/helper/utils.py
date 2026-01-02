from __future__ import annotations

from typing import List, Tuple
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping,Tuple

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
    
def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0.0 else 0.0


def _xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def compute_iou_xywh(a: List[float], b: List[float]) -> float:
    """
    IoU for COCO boxes in [x, y, w, h].
    """
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _index_gt_boxes(
    coco_gt: Mapping[str, Any],
) -> Dict[Any, Dict[int, List[List[float]]]]:
    # gt[image_id][category_id] -> list[bbox_xywh]

    out: Dict[Any, Dict[int, List[List[float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in coco_gt.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = int(ann.get("category_id", 1))
        bbox = ann.get("bbox")
        if img_id is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        out[img_id][cat_id].append([_safe_float(v) for v in bbox])
    return out


def _index_pred_boxes(
    preds: Iterable[Mapping[str, Any]],
    *,
    conf: float,
) -> Dict[Any, Dict[int, List[Tuple[float, List[float]]]]]:
    # pred[image_id][category_id] -> list[(score, bbox_xywh)]

    out: Dict[Any, Dict[int, List[Tuple[float, List[float]]]]] = defaultdict(lambda: defaultdict(list))
    for det in preds:
        img_id = det.get("image_id")
        bbox = det.get("bbox")
        if img_id is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue

        score = _safe_float(det.get("score", 0.0))
        if score < conf:
            continue

        cat_id = int(det.get("category_id", 1))
        out[img_id][cat_id].append((score, [_safe_float(v) for v in bbox]))
    return out