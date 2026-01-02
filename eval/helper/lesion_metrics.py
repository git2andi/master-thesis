from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from helper.utils import compute_iou_xywh


REALCOLON_META_ROOT = Path("/data/local/aschwab/data/realColon")
DEFAULT_LESION_INFO = REALCOLON_META_ROOT / "lesion_info.csv"
DEFAULT_VIDEO_INFO = REALCOLON_META_ROOT / "video_info.csv"


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _load_csv_as_dict(path: Path, key_fields: List[str]) -> Dict[str, Dict[str, str]]:
    """
      key -> row
    """
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return out

    for delim in (",", ";"):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            if not r.fieldnames:
                continue

            headers_norm = {(h or "").strip(): h for h in r.fieldnames}

            chosen_key: Optional[str] = None
            for k in key_fields:
                if k in headers_norm:
                    chosen_key = headers_norm[k]
                    break
            if chosen_key is None:
                continue

            for row in r:
                k_raw = row.get(chosen_key)
                k = (k_raw or "").strip()
                if not k:
                    continue
                cleaned = {(kk or "").strip(): (vv or "").strip() for kk, vv in row.items()}
                out[k] = cleaned

        if out:
            return out

    return out


def _frame_index_from_image_id(image_id: str) -> Optional[int]:
    """
    Canonical image_id is '001-001_123' -> returns 123.
    """
    if "_" not in image_id:
        return None
    try:
        return int(image_id.rsplit("_", 1)[1])
    except ValueError:
        return None


def _pct(num: int, den: int) -> float:
    return (100.0 * num / den) if den > 0 else 0.0


def compute_lesion_frame_detection_stats(
    coco_gt_remapped: Dict[str, Any],
    preds_remapped: List[Dict[str, Any]],
    *,
    iou_thr: float,
    conf: float,
    enable_timing: bool=True,
) -> Dict[str, Any]:
    """
    lesion-level stats.
    """
    lesion_info_path = DEFAULT_LESION_INFO
    video_info_path = DEFAULT_VIDEO_INFO

    # CSV keys
    lesion_info = _load_csv_as_dict(lesion_info_path, key_fields=["unique_object_id"])
    video_info = _load_csv_as_dict(video_info_path, key_fields=["unique_video_name"])

    # GT: image > (uid, gt_box), lesion > frames(set) ---
    gt_uid_boxes_by_img: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    lesion_frames: Dict[str, Set[str]] = defaultdict(set)

    for ann in coco_gt_remapped.get("annotations", []):
        uid = ann.get("unique_id")
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if not isinstance(uid, str) or not isinstance(img_id, str) or not isinstance(bbox, list):
            continue
        gt_uid_boxes_by_img[img_id].append((uid, bbox))
        lesion_frames[uid].add(img_id)

    # preds: image > pred boxes (score-filtered)
    preds_by_img: Dict[str, List[List[float]]] = defaultdict(list)
    for det in preds_remapped:
        img_id = det.get("image_id")
        bbox = det.get("bbox")
        score = det.get("score", 0.0)
        if not isinstance(img_id, str) or not isinstance(bbox, list):
            continue
        if not isinstance(score, (int, float)) or float(score) < conf:
            continue
        preds_by_img[img_id].append(bbox)

    # detect lesion frames (single pass over GT images)
    lesion_detected_frames: Dict[str, Set[str]] = defaultdict(set)

    for img_id, gt_items in gt_uid_boxes_by_img.items():
        pred_boxes = preds_by_img.get(img_id)
        if not pred_boxes:
            continue

        remaining = {uid for uid, _ in gt_items}
        if not remaining:
            continue

        for pb in pred_boxes:
            if not remaining:
                break
            for uid, gb in gt_items:
                if uid not in remaining:
                    continue
                if compute_iou_xywh(gb, pb) >= iou_thr:
                    lesion_detected_frames[uid].add(img_id)
                    remaining.remove(uid)

    # aggregate per lesion
    rows: List[Dict[str, Any]] = []

    n_lesions = len(lesion_frames)
    n_any = n_25 = n_50 = 0
    n_within_1s = n_within_3s = n_within_5s = 0

    det_fraction_sum = 0.0

    total_pos_frames = 0
    total_det_pos_frames = 0

    FPS_FIXED = 30.0
    FPS_STR_FIXED = "30"
    MIN_DET_FRAMES_IN_WINDOW = 15 

    # Exclude the two gap-affected lesions from mean first-detection time
    EXCLUDE_FROM_MEAN_FIRST_DET = {"003-014_1", "004-014_1"}

    latency_frames_for_mean: List[float] = []
    latency_seconds_for_mean: List[float] = []


    for uid, frames in lesion_frames.items():
        video_name = uid.split("_", 1)[0]

        n_frames_lesion = len(frames)
        det_frames = lesion_detected_frames.get(uid, set())
        n_frames_detected = len(det_frames)

        total_pos_frames += n_frames_lesion
        total_det_pos_frames += n_frames_detected

        det_fraction = (n_frames_detected / n_frames_lesion) if n_frames_lesion > 0 else 0.0
        det_fraction_sum += float(det_fraction)

        detected_any = int(n_frames_detected > 0)
        detected_25 = int(det_fraction >= 0.25)
        detected_50 = int(det_fraction >= 0.50)
        n_any += detected_any
        n_25 += detected_25
        n_50 += detected_50

        # frame indices from canonical ids
        gt_idxs = [_frame_index_from_image_id(x) for x in frames]
        gt_idxs = [x for x in gt_idxs if x is not None]
        first_gt = min(gt_idxs) if gt_idxs else -1

        det_idxs = [_frame_index_from_image_id(x) for x in det_frames]
        det_idxs = [x for x in det_idxs if x is not None]
        first_det = min(det_idxs) if det_idxs else -1

        latency_frames = max(0, first_det - first_gt) if (first_gt >= 0 and first_det >= 0) else -1


        # TIMING CALC
        # - within_Xs requires >=15 detected frames inside [first_gt, first_gt + X*fps - 1]
        fps_val = FPS_FIXED
        fps_str = FPS_STR_FIXED

        latency_seconds: Optional[float] = None
        if latency_frames >= 0:
            latency_seconds = latency_frames / fps_val

        # Mean first-detection time (exclude selected outliers only)
        if uid not in EXCLUDE_FROM_MEAN_FIRST_DET and latency_frames >= 0 and latency_seconds is not None:
            latency_frames_for_mean.append(float(latency_frames))
            latency_seconds_for_mean.append(float(latency_seconds))



        def _count_dets_in_window(seconds: int) -> int:
            if first_gt < 0 or not det_idxs:
                return 0
            end_idx = first_gt + int(seconds * fps_val) - 1
            return sum(1 for i in det_idxs if first_gt <= i <= end_idx)

        dets_1s = _count_dets_in_window(1)
        dets_3s = _count_dets_in_window(3)
        dets_5s = _count_dets_in_window(5)

        within_1s = int(dets_1s >= MIN_DET_FRAMES_IN_WINDOW)
        within_3s = int(dets_3s >= MIN_DET_FRAMES_IN_WINDOW)
        within_5s = int(dets_5s >= MIN_DET_FRAMES_IN_WINDOW)

        n_within_1s += within_1s
        n_within_3s += within_3s
        n_within_5s += within_5s

        # lesion metadata from lesion_info.csv (keyed by unique_object_id == uid)
        les_meta = lesion_info.get(uid, {})
        size_mm = (les_meta.get("size [mm]") or "").strip()
        hist_class = (les_meta.get("histology_class") or "").strip()

        rows.append(
            {
                "unique_id": uid,
                "video_name": video_name,
                "size_mm": size_mm if enable_timing else None,
                "histology_class": hist_class if enable_timing else None,
                "fps": fps_str if enable_timing else None,
                "n_frames_lesion": n_frames_lesion,
                "n_frames_detected": n_frames_detected,
                "det_fraction": float(det_fraction),
                "detected_any": detected_any,
                "detected_25pct": detected_25,
                "detected_50pct": detected_50,
                "first_gt_frame_idx": int(first_gt) if enable_timing and first_gt is not None else None,
                "first_det_frame_idx": int(first_det) if enable_timing and first_det is not None else None,
                "latency_frames": int(latency_frames) if enable_timing and latency_frames is not None else None,
                "latency_seconds": float(latency_seconds) if enable_timing and latency_seconds is not None else None,
                "detected_within_1s": int(within_1s) if enable_timing and within_1s is not None else None,
                "detected_within_3s": int(within_3s) if enable_timing and within_3s is not None else None,
                "detected_within_5s": int(within_5s) if enable_timing and within_5s is not None else None,
            }
        )

    rows.sort(key=lambda r: (r["video_name"], r["unique_id"]))

    mean_det_fraction = (det_fraction_sum / n_lesions) if n_lesions > 0 else 0.0

    summary = {
        "n_lesions": n_lesions,
        "detected_any": n_any,
        "detected_25pct": n_25,
        "detected_50pct": n_50,
        "pct_detected_any": _pct(n_any, n_lesions),
        "pct_detected_25pct": _pct(n_25, n_lesions),
        "pct_detected_50pct": _pct(n_50, n_lesions),
        "mean_det_fraction": float(mean_det_fraction),
        "mean_det_fraction_pct": float(mean_det_fraction * 100.0),
        "detected_within_1s": int(n_within_1s) if enable_timing else None,
        "detected_within_3s": int(n_within_3s) if enable_timing else None,
        "detected_within_5s": int(n_within_5s) if enable_timing else None,
        "pct_detected_within_1s": _pct(n_within_1s, n_lesions) if enable_timing else None,
        "pct_detected_within_3s": _pct(n_within_3s, n_lesions) if enable_timing else None,
        "pct_detected_within_5s": _pct(n_within_5s, n_lesions) if enable_timing else None,
        "mean_latency_frames_excl_outliers": _mean(latency_frames_for_mean) if enable_timing else None,
        "mean_latency_seconds_excl_outliers": _mean(latency_seconds_for_mean) if enable_timing else None,
        "n_latency_lesions_excl_outliers": len(latency_seconds_for_mean) if enable_timing else None,

    }

    return {"summary": summary, "rows": rows}
