# froc_metrics.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from helper.utils import compute_iou_xywh


# ----------------------------
# Shared core for FROC + AFROC
# ----------------------------

@dataclass(frozen=True)
class _PreparedFrocInputs:
    images: List[Dict[str, Any]]
    n_images: int
    gt_id_set: Set[Any]
    gt_by_img: Dict[Any, List[List[float]]]        # img_id -> [bbox_xywh, ...]
    total_gt: int                                  # total GT boxes across all images
    neg_img_ids: Set[Any]                           # images with no GT boxes
    dets: List[Tuple[float, Any, List[float]]]      # [(score, img_id, bbox_xywh), ...], sorted desc
    dropped_unknown: int


def _prepare_froc_inputs(
    coco_gt: Dict[str, Any],
    preds: List[Dict[str, Any]],
) -> _PreparedFrocInputs:
    images = coco_gt.get("images", [])
    n_images = len(images)

    gt_id_set: Set[Any] = set()
    for img in images:
        gt_id_set.add(img.get("id"))

    gt_by_img: Dict[Any, List[List[float]]] = defaultdict(list)
    for ann in coco_gt.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in gt_id_set:
            bbox = ann.get("bbox")
            if bbox is not None:
                gt_by_img[img_id].append(bbox)

    total_gt = 0
    neg_img_ids: Set[Any] = set()
    for img in images:
        img_id = img.get("id")
        if img_id is None:
            continue
        n_gt = len(gt_by_img.get(img_id, []))
        total_gt += n_gt
        if n_gt == 0:
            neg_img_ids.add(img_id)

    # keep predictions only for known image ids
    dets: List[Tuple[float, Any, List[float]]] = []
    dropped_unknown = 0
    for d in preds:
        if not isinstance(d, dict):
            continue
        img_id = d.get("image_id")
        if img_id not in gt_id_set:
            dropped_unknown += 1
            continue
        bbox = d.get("bbox")
        if bbox is None:
            continue
        score = float(d.get("score", 0.0))
        dets.append((score, img_id, bbox))

    dets.sort(key=lambda x: x[0], reverse=True)

    return _PreparedFrocInputs(
        images=images,
        n_images=n_images,
        gt_id_set=gt_id_set,
        gt_by_img=gt_by_img,
        total_gt=total_gt,
        neg_img_ids=neg_img_ids,
        dets=dets,
        dropped_unknown=dropped_unknown,
    )


# ----------------------------
# Core matcher: one-to-one per image, strict IoU > iou_thr
# ----------------------------

def _evaluate_at_conf_threshold(
    prepared: _PreparedFrocInputs,
    *,
    iou_thr: float,
    conf_thr: float,
) -> Tuple[int, int, Set[Any]]:
    """
    Evaluate ONE operating point at fixed conf threshold:
      - consider detections with score >= conf_thr
      - strict IoU rule: IoU > iou_thr
      - one-to-one matching per image: each GT box can be matched at most once
    Returns:
      detected_gt (TP count for y-axis),
      fp_marks (FP count for x-axis),
      neg_images_flagged (for AFROC x-axis)
    """
    # filter by score
    dets = [(s, img_id, bb) for (s, img_id, bb) in prepared.dets if s >= conf_thr]

    # matched flags per image
    gt_matched: Dict[Any, List[bool]] = {
        img.get("id"): [False] * len(prepared.gt_by_img.get(img.get("id"), []))
        for img in prepared.images
        if img.get("id") is not None
    }

    detected_gt = 0
    fp_marks = 0
    neg_images_flagged: Set[Any] = set()

    for score, img_id, pb in dets:
        # negative image => cannot match any GT
        if img_id in prepared.neg_img_ids:
            fp_marks += 1
            neg_images_flagged.add(img_id)
            continue

        gts = prepared.gt_by_img.get(img_id, [])
        matched_flags = gt_matched.get(img_id, [])

        best_k = -1
        best_iou = 0.0
        for k, gb in enumerate(gts):
            if matched_flags[k]:
                continue
            iou = compute_iou_xywh(gb, pb)
            # STRICT: IoU > iou_thr
            if iou > iou_thr and iou > best_iou:
                best_iou = iou
                best_k = k

        if best_k >= 0:
            matched_flags[best_k] = True
            detected_gt += 1
        else:
            fp_marks += 1

    return detected_gt, fp_marks, neg_images_flagged


# ----------------------------
# Public: fixed confidence threshold points (recommended for overlays)
# ----------------------------

def compute_froc_points_for_conf_thresholds(
    coco_gt: Dict[str, Any],
    preds: List[Dict[str, Any]],
    *,
    iou_thr: float,
    conf_thresholds: List[float],
) -> Dict[str, Any]:
    """
    Compute FROC values at user-defined confidence thresholds.
    This is ideal for:
      - comparing multiple models on identical operating points
      - writing to JSON and overlaying curves later
    """
    prepared = _prepare_froc_inputs(coco_gt, preds)

    conf_sorted = sorted(float(c) for c in conf_thresholds)
    out_conf: List[float] = []
    out_fppi: List[float] = []
    out_y: List[float] = []
    out_tp: List[int] = []
    out_fp: List[int] = []

    for conf_thr in conf_sorted:
        detected_gt, fp_marks, _ = _evaluate_at_conf_threshold(
            prepared, iou_thr=float(iou_thr), conf_thr=float(conf_thr)
        )
        fppi = fp_marks / prepared.n_images if prepared.n_images > 0 else 0.0
        y = detected_gt / prepared.total_gt if prepared.total_gt > 0 else 0.0

        out_conf.append(float(conf_thr))
        out_fppi.append(float(fppi))
        out_y.append(float(y))
        out_tp.append(int(detected_gt))
        out_fp.append(int(fp_marks))

    return {
        "metric": "FROC",
        "mode": "fixed_conf_thresholds",
        "iou_thr": float(iou_thr),
        "conf_thresholds": out_conf,
        "fp_per_image": out_fppi,
        "tpr_loc": out_y,
        "tp": out_tp,
        "fp": out_fp,
        "total_gt": int(prepared.total_gt),
        "n_images": int(prepared.n_images),
        "n_detections_total": int(len(prepared.dets)),
        "filtered_unknown_image_ids": int(prepared.dropped_unknown),
        "matching": {"policy": "one_to_one_per_image", "tolerance": f"iou>{iou_thr}"},
        "x_axis": "FPPI (false positives per image)",
        "y_axis": "fraction of GT boxes detected (at least once)",
    }


def compute_afroc_points_for_conf_thresholds(
    coco_gt: Dict[str, Any],
    preds: List[Dict[str, Any]],
    *,
    iou_thr: float,
    conf_thresholds: List[float],
) -> Dict[str, Any]:
    """
    Compute AFROC values at user-defined confidence thresholds.
    """
    prepared = _prepare_froc_inputs(coco_gt, preds)

    conf_sorted = sorted(float(c) for c in conf_thresholds)
    out_conf: List[float] = []
    out_fpf: List[float] = []
    out_y: List[float] = []
    out_tp: List[int] = []
    out_fp: List[int] = []

    n_neg = len(prepared.neg_img_ids)

    for conf_thr in conf_sorted:
        detected_gt, fp_marks, neg_images_flagged = _evaluate_at_conf_threshold(
            prepared, iou_thr=float(iou_thr), conf_thr=float(conf_thr)
        )
        fpf = (len(neg_images_flagged) / n_neg) if n_neg > 0 else 0.0
        y = detected_gt / prepared.total_gt if prepared.total_gt > 0 else 0.0

        out_conf.append(float(conf_thr))
        out_fpf.append(float(fpf))
        out_y.append(float(y))
        out_tp.append(int(detected_gt))
        out_fp.append(int(fp_marks))

    return {
        "metric": "AFROC",
        "mode": "fixed_conf_thresholds",
        "iou_thr": float(iou_thr),
        "conf_thresholds": out_conf,
        "fpf": out_fpf,
        "tpr_loc": out_y,
        "tp": out_tp,
        "fp": out_fp,
        "total_gt": int(prepared.total_gt),
        "n_images": int(prepared.n_images),
        "n_neg_images": int(n_neg),
        "n_detections_total": int(len(prepared.dets)),
        "filtered_unknown_image_ids": int(prepared.dropped_unknown),
        "matching": {"policy": "one_to_one_per_image", "tolerance": f"iou>{iou_thr}"},
        "x_axis": "FPF (fraction of negative images with ≥1 FP)",
        "y_axis": "fraction of GT boxes detected (at least once)",
    }


# ----------------------------
# Optional: full curve by unique scores (your original approach)
# ----------------------------

@dataclass(frozen=True)
class _ThresholdStep:
    score: float
    new_tp: int
    new_fp_marks: int
    new_neg_images_flagged: Set[Any]


def _iter_threshold_steps_by_unique_score(
    prepared: _PreparedFrocInputs,
    *,
    iou_thr: float,
) -> Iterable[_ThresholdStep]:
    """
    Sweep detections grouped by identical score; strict IoU > iou_thr; one-to-one per image.
    """
    gt_matched: Dict[Any, List[bool]] = {
        img.get("id"): [False] * len(prepared.gt_by_img.get(img.get("id"), []))
        for img in prepared.images
        if img.get("id") is not None
    }

    dets = prepared.dets
    i = 0
    while i < len(dets):
        score = dets[i][0]
        new_tp = 0
        new_fp_marks = 0
        new_neg_images_flagged: Set[Any] = set()

        j = i
        while j < len(dets) and dets[j][0] == score:
            _, img_id, pb = dets[j]

            if img_id in prepared.neg_img_ids:
                new_fp_marks += 1
                new_neg_images_flagged.add(img_id)
                j += 1
                continue

            gts = prepared.gt_by_img.get(img_id, [])
            matched_flags = gt_matched.get(img_id, [])

            best_k = -1
            best_iou = 0.0
            for k, gb in enumerate(gts):
                if matched_flags[k]:
                    continue
                iou = compute_iou_xywh(gb, pb)
                if iou > iou_thr and iou > best_iou:
                    best_iou = iou
                    best_k = k

            if best_k >= 0:
                matched_flags[best_k] = True
                new_tp += 1
            else:
                new_fp_marks += 1

            j += 1

        yield _ThresholdStep(
            score=float(score),
            new_tp=int(new_tp),
            new_fp_marks=int(new_fp_marks),
            new_neg_images_flagged=set(new_neg_images_flagged),
        )
        i = j


def compute_froc_curve(
    coco_gt: Dict[str, Any],
    preds: List[Dict[str, Any]],
    *,
    iou_thr: float,
) -> Dict[str, Any]:
    """
    Full FROC curve sweeping all unique scores (descending).
    This is smooth but model-specific (thresholds differ per model).
    """
    prepared = _prepare_froc_inputs(coco_gt, preds)

    if prepared.n_images == 0 or prepared.total_gt == 0:
        return {
            "metric": "FROC",
            "mode": "unique_score_sweep",
            "iou_thr": float(iou_thr),
            "thresholds": [],
            "fp_per_image": [],
            "tpr_loc": [],
            "total_gt": int(prepared.total_gt),
            "n_images": int(prepared.n_images),
            "n_detections_total": int(len(prepared.dets)),
            "filtered_unknown_image_ids": int(prepared.dropped_unknown),
        }

    out_thr: List[float] = []
    out_x: List[float] = []
    out_y: List[float] = []

    fp_marks = 0
    detected_gt = 0

    for step in _iter_threshold_steps_by_unique_score(prepared, iou_thr=float(iou_thr)):
        fp_marks += step.new_fp_marks
        detected_gt += step.new_tp

        out_thr.append(float(step.score))
        out_x.append(float(fp_marks / prepared.n_images))
        out_y.append(float(detected_gt / prepared.total_gt))

    return {
        "metric": "FROC",
        "mode": "unique_score_sweep",
        "iou_thr": float(iou_thr),
        "thresholds": out_thr,
        "fp_per_image": out_x,
        "tpr_loc": out_y,
        "total_gt": int(prepared.total_gt),
        "n_images": int(prepared.n_images),
        "n_detections_total": int(len(prepared.dets)),
        "filtered_unknown_image_ids": int(prepared.dropped_unknown),
        "matching": {"policy": "one_to_one_per_image", "tolerance": f"iou>{iou_thr}"},
        "x_axis": "FPPI (false positives per image)",
        "y_axis": "fraction of GT boxes detected (at least once)",
    }
