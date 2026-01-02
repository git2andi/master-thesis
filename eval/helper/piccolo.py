from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def canonical_piccolo_stem_id(name: str) -> Optional[str]:
    """
    Canonicalize a PICCOLO frame identifier to the filename stem, e.g.:
      '003_VP3_frame0030.jpg' -> '003_VP3_frame0030'
      '003_VP3_frame0030'     -> '003_VP3_frame0030'
      '/path/.../003_VP3_frame0030.jpg' -> '003_VP3_frame0030'
    """
    if not isinstance(name, str) or not name:
        return None
    stem = Path(name).stem
    return stem if stem else None


def remap_piccolo_coco_gt_ids(coco_gt_raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Remap COCO GT so that:
      - image['id'] becomes canonical stem string '003_VP3_frame0030'
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
        new_id = canonical_piccolo_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize PICCOLO GT file_name: {fname}")

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


def remap_piccolo_ultralytics_preds_ids(
    preds_raw: List[Dict[str, Any]],
    *,
    pred_image_id_key: str = "image_id",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remap Ultralytics preds to canonical stem ids.

    Assumes det[pred_image_id_key] is a string like:
      '003_VP3_frame0030' or '003_VP3_frame0030.jpg'
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

        new_id = canonical_piccolo_stem_id(v)
        if new_id is None:
            continue

        d2 = dict(det)
        d2[pred_image_id_key] = new_id
        out.append(d2)

    return out, missing


def build_piccolo_gt_imageid_to_canonical(coco_gt_raw: Dict[str, Any]) -> Dict[Any, str]:
    """
    Build mapping: original COCO GT image['id'] (often int) -> canonical PICCOLO id (stem string).
    """
    images = coco_gt_raw.get("images", [])
    m: Dict[Any, str] = {}

    for img in images:
        old_id = img.get("id")
        fname = img.get("file_name")
        if not isinstance(fname, str):
            raise ValueError("GT image missing string 'file_name'.")
        new_id = canonical_piccolo_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize PICCOLO GT file_name: {fname}")
        m[old_id] = new_id

    return m


def remap_piccolo_detectron2_preds_ids(
    preds_raw: List[Dict[str, Any]],
    coco_gt_raw: Dict[str, Any],
    *,
    pred_image_id_key: str = "image_id",
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Remap Detectron2 COCO-style predictions:
      pred['image_id'] is typically the numeric COCO GT image id (int).
    We map it to canonical stem string ids based on GT images[*].

    Returns:
      - remapped predictions
      - n_missing_key
      - n_unknown_id
    """
    id_map = build_piccolo_gt_imageid_to_canonical(coco_gt_raw)

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
