
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from helper.utils import compute_iou_xywh, _index_gt_boxes, _index_pred_boxes


@dataclass(frozen=True)
class DetectionConfusion:
    tp: int
    fp: int
    fn: int

    @property
    def n_gt(self) -> int:
        return self.tp + self.fn

    @property
    def n_pred(self) -> int:
        return self.tp + self.fp



def _match_image_category_one_to_one(
    gt_boxes: List[List[float]],
    pred_boxes: List[Tuple[float, List[float]]],
    *,
    iou_thr: float,
) -> DetectionConfusion:
    """
    Greedy 1-to-1 matching within a single (image_id, category_id).
      - sort predictions by descending confidence
      - for each prediction: match to the best-IoU unmatched GT if IoU >= iou_thr
      - unmatched predictions are FP
      - unmatched GTs are FN
    """
    if not gt_boxes and not pred_boxes:
        return DetectionConfusion(tp=0, fp=0, fn=0)

    matched = [False] * len(gt_boxes)
    tp = fp = 0

    pred_sorted = sorted(pred_boxes, key=lambda x: x[0], reverse=True)
    for _, pbox in pred_sorted:
        best_iou = 0.0
        best_j: Optional[int] = None

        for j, gt_box in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = compute_iou_xywh(pbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou >= iou_thr:
            matched[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in matched if not m)
    return DetectionConfusion(tp=tp, fp=fp, fn=fn)


def compute_detection_confusion(
    coco_gt: Mapping[str, Any],
    preds: Iterable[Mapping[str, Any]],
    *,
    iou_thr: float,
    conf: float,
) -> DetectionConfusion:
    # Matching per img with 1-to-1 policy: each GT can be matched to at most one prediction.
    
    gt_by_img_cat = _index_gt_boxes(coco_gt)
    pred_by_img_cat = _index_pred_boxes(preds, conf=conf)
    
    # total GT boxes per image (summing over categories)
    gt_total_per_img = {
        img_id: sum(len(boxes) for boxes in cats.values())
        for img_id, cats in gt_by_img_cat.items()
    }

    n_pos_imgs = len(gt_total_per_img)
    n_multi_gt_imgs = sum(v > 1 for v in gt_total_per_img.values())
    max_gt_per_img = max(gt_total_per_img.values()) if gt_total_per_img else 0

    print(f"[INFO] GT images with >=1 GT box: {n_pos_imgs}")
    print(f"[INFO] GT images with >1 GT box: {n_multi_gt_imgs}")
    print(f"[INFO] max GT boxes in one image: {max_gt_per_img}")

    # also check how many categories per image (important diagnostic)
    cats_per_img = {img_id: len(cats) for img_id, cats in gt_by_img_cat.items()}
    print(f"[INFO] max categories in one image: {max(cats_per_img.values()) if cats_per_img else 0}")
    print(f"[INFO] images with >1 category: {sum(v > 1 for v in cats_per_img.values())}")




    n_gt_boxes = sum(len(boxes) for cats in gt_by_img_cat.values() for boxes in cats.values())
    n_pred_boxes = sum(len(items) for cats in pred_by_img_cat.values() for items in cats.values())
    print(f"[INFO] GT boxes total: {n_gt_boxes}")
    print(f"[INFO] Pred boxes after IoU + Conf: {n_pred_boxes}")


    all_imgs = set(gt_by_img_cat.keys()) | set(pred_by_img_cat.keys())

    tp = fp = fn = 0
    for img_id in all_imgs:
        gt_cats = set(gt_by_img_cat.get(img_id, {}).keys())
        pr_cats = set(pred_by_img_cat.get(img_id, {}).keys())
        for cat_id in (gt_cats | pr_cats):
            c = _match_image_category_one_to_one(
                gt_boxes=gt_by_img_cat.get(img_id, {}).get(cat_id, []),
                pred_boxes=pred_by_img_cat.get(img_id, {}).get(cat_id, []),
                iou_thr=iou_thr,
            )
            tp += c.tp
            fp += c.fp
            fn += c.fn

    return DetectionConfusion(tp=tp, fp=fp, fn=fn)


# Recall and Precision
def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0.0 else 0.0

def precision_from_counts(tp: int, fp: int) -> float:
    return _safe_div(tp, tp + fp)

def recall_from_counts(tp: int, fn: int) -> float:
    return _safe_div(tp, tp + fn)

# Sensitivity and Miss Rate
def sensitivity_from_counts(tp: int, fn: int) -> float:
    return recall_from_counts(tp, fn) # sens = recall

def miss_rate_from_sensitivity(sensitivity: float) -> float:
    return float(1.0 - sensitivity)


# F-Scores
def fbeta_from_pr(precision: float, recall: float, *, beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be > 0")

    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + b2) * (precision * recall) / denom)


def compute_detection_metrics(
    coco_gt: Mapping[str, Any],
    preds: Iterable[Mapping[str, Any]],
    *,
    iou_thr: float,
    conf: float,
    betas: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, Any]:

    c = compute_detection_confusion(coco_gt, preds, iou_thr=iou_thr, conf=conf)

    prec = precision_from_counts(c.tp, c.fp)
    rec = recall_from_counts(c.tp, c.fn)
    sens = rec
    miss = miss_rate_from_sensitivity(sens)

    def _f_key(beta: float) -> str:
        return f"F{int(beta)}" if float(beta).is_integer() else f"F{beta}"

    f_scores = {_f_key(b): fbeta_from_pr(prec, rec, beta=b) for b in betas}

    return {
        "iou_thr": float(iou_thr),
        "conf": float(conf),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "fn": int(c.fn),
        "n_gt": int(c.n_gt),
        "n_pred": int(c.n_pred),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(sens),
        "miss_rate": float(miss),
        "f_scores": f_scores,
    }