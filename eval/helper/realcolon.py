from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def canonical_realcolon_stem_id(name: str) -> Optional[str]:
    """
    Canonicalize a REAL-Colon frame identifier to 'VID_FRAME', e.g.:
      '001-001_000001.jpg' -> '001-001_1'
      '001-001_1.jpg'      -> '001-001_1'
      '.../001-001_000123' -> '001-001_123'

    Rule:
      - basename
      - drop extension
      - split at last '_' -> (prefix, suffix)
      - parse suffix as int (removes leading zeros)
      - return f"{prefix}_{int(suffix)}"
    """
    stem = Path(name).stem  # keeps prefix, removes extension
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    try:
        n = int(suffix)
    except ValueError:
        return None
    return f"{prefix}_{n}"


def remap_realcolon_coco_gt_ids(coco_gt_raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Remap COCO GT so that:
      - image['id'] becomes canonical string '001-001_1'
      - annotation['image_id'] updated accordingly

    Returns:
      - remapped COCO dict
      - set of all remapped image ids
    """
    images = coco_gt_raw.get("images", [])
    anns = coco_gt_raw.get("annotations", [])

    old_to_new: Dict[Any, str] = {}
    new_images: List[Dict[str, Any]] = []

    for img in images:
        fname = img.get("file_name")
        if not isinstance(fname, str):
            raise ValueError("GT image missing string 'file_name'.")
        new_id = canonical_realcolon_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize GT file_name: {fname}")

        old_to_new[img.get("id")] = new_id
        new_images.append({**img, "id": new_id})

    new_anns: List[Dict[str, Any]] = []
    dropped = 0
    for ann in anns:
        old_img_id = ann.get("image_id")
        if old_img_id not in old_to_new:
            dropped += 1
            continue
        new_anns.append({**ann, "image_id": old_to_new[old_img_id]})

    if dropped > 0:
        print(f"[WARN] GT annotations referencing unknown image_id: {dropped} (dropped)")

    remapped = {**coco_gt_raw, "images": new_images, "annotations": new_anns}
    return remapped, set(old_to_new.values())


def remap_realcolon_ultralytics_preds_ids(
    preds_raw: List[Dict[str, Any]],
    *,
    pred_image_id_key: str = "image_id",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remap Ultralytics preds to canonical '001-001_1' ids.

    Assumes det[pred_image_id_key] is a string like:
      '001-001_000001.jpg' or '001-001_1.jpg' or '001-001_000001'
    """
    out: List[Dict[str, Any]] = []
    missing = 0

    for det in preds_raw:
        if not isinstance(det, dict):
            continue
        if pred_image_id_key not in det:
            missing += 1
            continue

        v = det.get(pred_image_id_key)
        if not isinstance(v, str):
            continue

        new_id = canonical_realcolon_stem_id(v)
        if new_id is None:
            continue

        d2 = dict(det)
        d2[pred_image_id_key] = new_id
        out.append(d2)

    return out, missing

def build_realcolon_gt_imageid_to_canonical(coco_gt_raw: Dict[str, Any]) -> Dict[Any, str]:
    """
    Build mapping: original COCO GT image['id'] (often int) -> canonical REAL-Colon id 'VID_FRAME'.
    Uses GT images[*]['file_name'] as the source of truth for canonicalization.

    This is the mapping you want for Detectron2 predictions, because Detectron2 typically outputs
    numeric image_id matching the GT json.
    """
    images = coco_gt_raw.get("images", [])
    m: Dict[Any, str] = {}

    for img in images:
        old_id = img.get("id")
        fname = img.get("file_name")
        if not isinstance(fname, str):
            raise ValueError("GT image missing string 'file_name'.")
        new_id = canonical_realcolon_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize GT file_name: {fname}")
        m[old_id] = new_id

    return m


def remap_realcolon_detectron2_preds_ids(
    preds_raw: List[Dict[str, Any]],
    coco_gt_raw: Dict[str, Any],
    *,
    pred_image_id_key: str = "image_id",
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Remap Detectron2 COCO-style predictions:
      pred['image_id'] is typically the numeric COCO GT image id (int).
    We map it to canonical string ids ('001-001_1') based on GT images[*].

    Returns:
      - remapped predictions
      - n_missing_key: preds missing pred_image_id_key
      - n_unknown_id: preds whose image_id not found in GT mapping
    """
    id_map = build_realcolon_gt_imageid_to_canonical(coco_gt_raw)

    out: List[Dict[str, Any]] = []
    n_missing_key = 0
    n_unknown_id = 0

    for det in preds_raw:
        if not isinstance(det, dict):
            continue

        if pred_image_id_key not in det:
            n_missing_key += 1
            if strict:
                raise ValueError("Prediction missing image_id key.")
            continue

        old_id = det.get(pred_image_id_key)

        if old_id not in id_map:
            n_unknown_id += 1
            if strict:
                raise ValueError(f"Prediction image_id={old_id!r} not found in GT images.")
            continue

        d2 = dict(det)
        d2[pred_image_id_key] = id_map[old_id]
        out.append(d2)

    return out, n_missing_key, n_unknown_id