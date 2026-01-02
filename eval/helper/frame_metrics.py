from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from helper.utils import compute_iou_xywh, _safe_div, _index_pred_boxes, _index_gt_boxes


@dataclass(frozen=True)
class FrameMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    tpr: float
    fpr: float
    f1: float
    f2: float


# F-Scores
def _fbeta_from_pr(precision: float, recall: float, *, beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be > 0")
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + b2) * (precision * recall) / denom)


def compute_frame_metrics(
    coco_gt: Mapping[str, Any],
    preds: List[Dict[str, Any]],
    *,
    iou_thr: float,
    conf: float,
) -> FrameMetrics:
    """
    Frame-level reduction of detections to a binary decision per frame.

      - Predicted positive frame: contains >=1 detection with score >= conf.
      - GT-positive frame: contains >=1 GT box.
      - TP_frame: GT-positive frame with >=1 detection (score>=conf) that matches any GT with IoU>iou_thr.
      - FN_frame: GT-positive frame without such a valid detection.
      - FP_frame: GT-negative frame with >=1 detection (score>=conf).
      - TN_frame: GT-negative frame with no detections (score>=conf).
    """
    gt_by_img_cat = _index_gt_boxes(coco_gt)
    pred_by_img_cat = _index_pred_boxes(preds, conf=conf)

    tp = fp = tn = fn = 0

    for img in coco_gt.get("images", []):
        img_id = img.get("id")
        if img_id is None:
            continue

        # Flatten GT
        gt_cat_map = gt_by_img_cat.get(img_id, {})
        gt_boxes: List[List[float]] = []
        for boxes in gt_cat_map.values():
            gt_boxes.extend(boxes)

        # Flatten pred
        pred_cat_map = pred_by_img_cat.get(img_id, {})
        pred_boxes: List[List[float]] = []
        for scored_boxes in pred_cat_map.values():
            for _, bbox in scored_boxes:
                pred_boxes.append(bbox)

        gt_pos = len(gt_boxes) > 0
        pred_pos = len(pred_boxes) > 0

        # GT-negative frames: any detection makes it FP
        if not gt_pos:
            if pred_pos:
                fp += 1
            else:
                tn += 1
            continue

        # GT-positive frames: require at least one IoU-valid detection
        matched = False
        if pred_pos:
            for pb in pred_boxes:
                for gb in gt_boxes:
                    if compute_iou_xywh(pb, gb) > iou_thr:
                        matched = True
                        break
                if matched:
                    break

        if matched:
            tp += 1
        else:
            fn += 1


    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    sensitivity = recall
    specificity = _safe_div(tn, tn + fp)
    tpr = sensitivity
    fpr = _safe_div(fp, fp + tn)
    f1 = _fbeta_from_pr(precision, recall, beta=1.0)
    f2 = _fbeta_from_pr(precision, recall, beta=2.0)

    return FrameMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        sensitivity=sensitivity,
        specificity=specificity,
        tpr=tpr,
        fpr=fpr,
        f1=f1,
        f2=f2,
    )

def compute_frame_tpr_fpr_sweep(
    coco_gt: Mapping[str, Any],
    preds: List[Dict[str, Any]],
    *,
    iou_thr: float,
    selected_thresholds: List[float] | None = None,
    dense_lo: float = 0.01,
    dense_hi: float = 0.10,
    dense_step: float = 0.005,
    coarse_step: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes frame-level curves across a custom threshold grid and returns a JSON-ready dict.

    Threshold grid:
      - dense in [dense_lo, dense_hi] with dense_step
      - coarse from dense_hi to 1.0 with coarse_step
      - plus selected_thresholds (for TP/FP bars), all deduped

    If predictions are prefiltered (e.g., min score >= 0.1), then:
      - thresholds below that min score are removed
      - curves and plots should start from that min score
    """
    # --- Build GT boxes per frame (flatten across categories) ---
    gt_by_img_cat = _index_gt_boxes(coco_gt)
    gt_boxes_by_img: Dict[Any, List[List[float]]] = {}
    for img_id, cat_map in gt_by_img_cat.items():
        flat: List[List[float]] = []
        for boxes in cat_map.values():
            flat.extend(boxes)
        if flat:
            gt_boxes_by_img[img_id] = flat

    # --- Determine minimum confidence actually present in preds (prefilter detection) ---
    min_conf_available = 1.0
    found_any_score = False
    for p in preds:
        s = p.get("score", None)
        if isinstance(s, (int, float)):
            found_any_score = True
            if float(s) < min_conf_available:
                min_conf_available = float(s)
    if not found_any_score:
        # no predictions at all; still define a sane start
        min_conf_available = 0.001

    # clamp (and avoid exactly 0 for plotting)
    min_conf_available = max(0.001, min(1.0, float(min_conf_available)))

    # --- Predictions per frame (no thresholding here) ---
    pred_by_img_cat = _index_pred_boxes(preds, conf=0.0)

    # --- Precompute per-frame scores (efficient sweep) ---
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    for img in coco_gt.get("images", []):
        img_id = img.get("id")
        if img_id is None:
            continue

        gt_boxes = gt_boxes_by_img.get(img_id, [])
        gt_pos = len(gt_boxes) > 0

        pred_cat_map = pred_by_img_cat.get(img_id, {})
        scored_preds: List[Tuple[float, List[float]]] = []
        for scored_boxes in pred_cat_map.values():
            scored_preds.extend(scored_boxes)

        if not scored_preds:
            if gt_pos:
                pos_scores.append(0.0)
            else:
                neg_scores.append(0.0)
            continue

        max_any = max(s for s, _ in scored_preds)

        if not gt_pos:
            neg_scores.append(max_any)
            continue

        max_valid = 0.0
        for s, pb in scored_preds:
            for gb in gt_boxes:
                if compute_iou_xywh(pb, gb) > iou_thr:
                    if s > max_valid:
                        max_valid = s
                    break
        pos_scores.append(max_valid)

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # --- Build threshold set: dense region + coarse region + selected thresholds ---
    def _frange(lo: float, hi: float, step: float) -> List[float]:
        vals: List[float] = []
        if step <= 0:
            raise ValueError("step must be > 0")
        x = lo
        # include hi with tolerance
        while x <= hi + 1e-12:
            vals.append(round(float(x), 6))
            x += step
        return vals

    dense = _frange(dense_lo, dense_hi, dense_step)
    coarse = _frange(dense_hi, 1.0, coarse_step)

    base = dense + coarse
    sel = [float(t) for t in (selected_thresholds or [])]
    all_thr = sorted({t for t in (base + sel) if 0.0 <= t <= 1.0})

    # If preds are prefiltered, drop below min_conf_available
    all_thr = [t for t in all_thr if t >= min_conf_available - 1e-12]
    if not all_thr:
        all_thr = [min_conf_available]

    # --- Evaluate sweep ---
    out_thresholds: List[float] = []
    out_tpr: List[float] = []
    out_fpr: List[float] = []
    out_precision: List[float] = []
    out_recall: List[float] = []
    out_sensitivity: List[float] = []
    out_specificity: List[float] = []
    out_f1: List[float] = []
    out_f2: List[float] = []
    out_tp: List[int] = []
    out_fp: List[int] = []
    out_tn: List[int] = []
    out_fn: List[int] = []

    for t in all_thr:
        tp = sum(1 for s in pos_scores if s >= t)
        fn = n_pos - tp
        fp = sum(1 for s in neg_scores if s >= t)
        tn = n_neg - fp

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)

        sensitivity = recall
        specificity = _safe_div(tn, tn + fp)

        tpr = sensitivity
        fpr = _safe_div(fp, fp + tn)

        f1 = _fbeta_from_pr(precision, recall, beta=1.0)
        f2 = _fbeta_from_pr(precision, recall, beta=2.0)

        out_thresholds.append(float(t))
        out_tpr.append(float(tpr))
        out_fpr.append(float(fpr))
        out_precision.append(float(precision))
        out_recall.append(float(recall))
        out_sensitivity.append(float(sensitivity))
        out_specificity.append(float(specificity))
        out_f1.append(float(f1))
        out_f2.append(float(f2))
        out_tp.append(int(tp))
        out_fp.append(int(fp))
        out_tn.append(int(tn))
        out_fn.append(int(fn))

    result: Dict[str, Any] = {
        "iou_thr": float(iou_thr),
        "n_pos_frames": int(n_pos),
        "n_neg_frames": int(n_neg),
        "min_conf_available": float(min_conf_available),
        "thresholds": out_thresholds,
        "tpr": out_tpr,
        "fpr": out_fpr,
        "precision": out_precision,
        "recall": out_recall,
        "sensitivity": out_sensitivity,
        "specificity": out_specificity,
        "f1": out_f1,
        "f2": out_f2,
        "tp": out_tp,
        "fp": out_fp,
        "tn": out_tn,
        "fn": out_fn,
    }

    # --- Add "selected" subset (for TP/FP bars), robust to float mismatch ---
    if selected_thresholds is not None:
        # Filter selected thresholds to those >= min_conf_available
        filtered_selected = [float(t) for t in selected_thresholds if float(t) >= min_conf_available - 1e-12]

        def _nearest_index(thr_list: List[float], target: float) -> int:
            best_i = 0
            best_d = float("inf")
            for i, v in enumerate(thr_list):
                d = abs(v - target)
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        selected_rows: List[Dict[str, Any]] = []
        for t in filtered_selected:
            i = _nearest_index(out_thresholds, t)
            selected_rows.append(
                {
                    "threshold": float(t),
                    "threshold_used": float(out_thresholds[i]),
                    "tp": out_tp[i],
                    "fp": out_fp[i],
                    "tn": out_tn[i],
                    "fn": out_fn[i],
                    "tpr": out_tpr[i],
                    "fpr": out_fpr[i],
                    "sensitivity": out_sensitivity[i],
                    "specificity": out_specificity[i],
                    "f1": out_f1[i],
                    "f2": out_f2[i],
                }
            )

        result["selected_thresholds"] = [float(t) for t in filtered_selected]
        result["selected"] = selected_rows

    return result
