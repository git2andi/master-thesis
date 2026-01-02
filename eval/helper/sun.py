from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def canonical_sun_stem_id(name: str) -> Optional[str]:
    """
    Canonicalize a SUN frame identifier to the file stem (no extension).

    Examples:
      '.../case81_..._image0001.jpg' -> 'case81_..._image0001'
      'case81_..._image0001.jpg'     -> 'case81_..._image0001'
      'case81_..._image0001'         -> 'case81_..._image0001'
    """
    if not isinstance(name, str) or not name:
        return None
    return Path(name).stem


def remap_sun_coco_gt_ids(coco_gt_raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Remap SUN COCO GT so that:
      - image['id'] becomes canonical stem string (from image['file_name'])
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
            raise ValueError("SUN GT image missing string 'file_name'.")
        new_id = canonical_sun_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize SUN GT file_name: {fname}")

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
        print(f"[WARN] SUN GT annotations referencing unknown image_id: {dropped} (dropped)")

    remapped = {**coco_gt_raw, "images": new_images, "annotations": new_anns}
    return remapped, set(old_to_new.values())


def remap_sun_ultralytics_preds_ids(
    preds_raw: List[Dict[str, Any]],
    *,
    pred_image_id_key: str = "image_id",
    pred_file_name_key: str = "file_name",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remap Ultralytics preds to canonical SUN stem ids.

    Your Ultralytics preds use:
      image_id: '..._image0004' (stem, no extension)
      file_name: '..._image0004.jpg'

    We accept either field:
      - prefer image_id if present and string
      - else fall back to file_name
    """
    out: List[Dict[str, Any]] = []
    missing = 0

    for det in preds_raw:
        if not isinstance(det, dict):
            continue

        src: Optional[str] = None
        if pred_image_id_key in det and isinstance(det.get(pred_image_id_key), str):
            src = det.get(pred_image_id_key)
        elif pred_file_name_key in det and isinstance(det.get(pred_file_name_key), str):
            src = det.get(pred_file_name_key)

        if src is None:
            missing += 1
            continue

        new_id = canonical_sun_stem_id(src)
        if new_id is None:
            continue

        d2 = dict(det)
        d2[pred_image_id_key] = new_id
        out.append(d2)

    return out, missing


def build_sun_gt_imageid_to_canonical(coco_gt_raw: Dict[str, Any]) -> Dict[Any, str]:
    """
    Mapping: original COCO GT image['id'] (often int) -> canonical SUN stem id.
    Uses images[*]['file_name'] as source of truth.
    """
    images = coco_gt_raw.get("images", [])
    m: Dict[Any, str] = {}

    for img in images:
        old_id = img.get("id")
        fname = img.get("file_name")
        if not isinstance(fname, str):
            raise ValueError("SUN GT image missing string 'file_name'.")
        new_id = canonical_sun_stem_id(fname)
        if new_id is None:
            raise ValueError(f"Could not canonicalize SUN GT file_name: {fname}")
        m[old_id] = new_id

    return m


def remap_sun_detectron2_preds_ids(
    preds_raw: List[Dict[str, Any]],
    coco_gt_raw: Dict[str, Any],
    *,
    pred_image_id_key: str = "image_id",
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Remap Detectron2 COCO-style predictions:
      pred['image_id'] is typically the numeric COCO GT image id (int).
    We map it to canonical SUN stem ids based on GT images[*].

    Returns:
      - remapped predictions
      - n_missing_key: preds missing pred_image_id_key
      - n_unknown_id: preds whose image_id not found in GT mapping
    """
    id_map = build_sun_gt_imageid_to_canonical(coco_gt_raw)

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
