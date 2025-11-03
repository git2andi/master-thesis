#!/usr/bin/env python3
"""
recanon_coco_realcolon.py

Goal: Read existing COCO annotations for REAL-Colon (at original, 640, 224
roots), validate & standardize them, and emit **cleaned COCO JSONs** while
**preserving all valid information** and **adding missing important fields**.

What is standardized:
- Ensure single category id = 1, name = 'lesion', supercategory = 'lesion'.
- Ensure image/annotation IDs are unique, integer, and contiguous (1..N).
- Ensure each annotation has bbox=[x,y,w,h], iscrowd=0, area=w*h (recomputed).
- Clamp bboxes to image bounds using image width/height from the JSON.
- Drop degenerate boxes (w<=0 or h<=0) after clamping; counts are logged.
- Preserve any extra per-image fields if present (e.g., video_id, frame_id).
- Fill in reasonable `info` and `licenses` if missing; otherwise keep originals.

It does **not** move or touch images; it only writes cleaned JSONs next to the
originals (or into --out-dir if specified) with the suffix `.clean.json`.

Expected layout per root, e.g. /data/local/aschwab/data/realColon_640x640/
  images/{train,val,test}/...      # not modified
  annotations_coco_train.json
  annotations_coco_val.json
  annotations_coco_test.json

Usage examples
--------------
# Clean one root (writes *.clean.json next to originals)
python recanon_coco_realcolon.py \
  --root /data/local/aschwab/data/realColon_640x640

# Clean all three roots and send cleaned files to a sibling 'clean' dir
python recanon_coco_realcolon.py \
  --root /data/local/aschwab/data/realColon_full \
  --root /data/local/aschwab/data/realColon_640x640 \
  --root /data/local/aschwab/data/realColon_224x224 \
  --out-dir-name cleaned

Outputs
-------
<root>/annotations_coco_train.clean.json
<root>/annotations_coco_val.clean.json
<root>/annotations_coco_test.clean.json
(or under <root>/cleaned/ if --out-dir-name is used)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

SUBSETS = ("train", "val", "test")
INFILES = {s: f"annotations_coco_{s}.json" for s in SUBSETS}


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _ensure_categories(cats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a single-class categories list with id=1, name='lesion'.
    If existing categories already define a single class, remap to id=1.
    """
    if not cats:
        return [{"id": 1, "name": "lesion", "supercategory": "lesion"}]
    # Map any first category to id=1, keep its name if already 'lesion'
    # but standardize to lowercase 'lesion' for consistency.
    first = cats[0]
    name = str(first.get("name", "lesion")).strip().lower() or "lesion"
    return [{"id": 1, "name": name, "supercategory": name}]


def _reindex_images(images: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    new_images = []
    id_map: Dict[int, int] = {}
    next_id = 1
    for im in images:
        old_id = int(im.get("id")) if "id" in im else None
        id_map[old_id] = next_id
        new_im = dict(im)
        new_im["id"] = next_id
        new_images.append(new_im)
        next_id += 1
    return new_images, id_map


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _clean_annotations(anns: List[Dict[str, Any]], id_map_img: Dict[int, int], dim_lookup: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
    new_anns = []
    next_id = 1
    dropped = 0
    for a in anns:
        img_old = int(a["image_id"]) if "image_id" in a else None
        img_new = id_map_img.get(img_old)
        if img_new is None:
            continue
        bbox = a.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        # Clamp by image dims
        W, H = dim_lookup.get(img_old, (None, None))
        if W is None or H is None:
            # If dims missing (shouldn't happen), keep as-is but drop non-positive
            if w <= 0 or h <= 0:
                dropped += 1
                continue
        else:
            x1 = _clamp(x, 0.0, W - 1)
            y1 = _clamp(y, 0.0, H - 1)
            x2 = _clamp(x + w, 0.0, W - 1)
            y2 = _clamp(y + h, 0.0, H - 1)
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                dropped += 1
                continue
        new_a = dict(a)
        new_a["id"] = next_id
        new_a["image_id"] = img_new
        new_a["category_id"] = 1
        new_a["bbox"] = [x, y, w, h]
        new_a["area"] = w * h
        new_a["iscrowd"] = int(a.get("iscrowd", 0))
        next_id += 1
        new_anns.append(new_a)
    if dropped:
        print(f"  - Dropped {dropped} degenerate boxes after clamp")
    return new_anns


def _dimension_lookup(images: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int]]:
    lk = {}
    for im in images:
        iid = int(im.get("id"))
        W = int(im.get("width"))
        H = int(im.get("height"))
        lk[iid] = (W, H)
    return lk


def _merge_info(orig: Dict[str, Any] | None) -> Dict[str, Any]:
    base = {
        "description": "REAL-Colon — cleaned COCO annotations (IDs normalized, bboxes clamped)",
        "version": "1.0",
        "year": 2025,
        "contributor": "Andreas Schwab (Master's thesis)",
    }
    if orig:
        base.update({k: v for k, v in orig.items() if v is not None})
    return base


def _merge_licenses(orig: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if orig and len(orig) > 0:
        return orig
    return [{"id": 1, "name": "Dataset per REAL-Colon paper", "url": ""}]


def clean_one(in_json_path: Path, out_json_path: Path):
    data = _load_json(in_json_path)

    categories = _ensure_categories(data.get("categories", []))

    # Reindex images; preserve extra fields (e.g., video_id, frame_id) if present
    images_old = data.get("images", [])
    images_new, img_id_map = _reindex_images(images_old)

    # Dim lookup uses old IDs (because bbox clamp uses original mapping), but
    # widths/heights are the same for new entries; we can build with old ids.
    dim_lookup = _dimension_lookup(images_old)

    # Clean annotations
    anns_old = data.get("annotations", [])
    anns_new = _clean_annotations(anns_old, img_id_map, dim_lookup)

    out = {
        "info": _merge_info(data.get("info")),
        "licenses": _merge_licenses(data.get("licenses")),
        "images": images_new,
        "annotations": anns_new,
        "categories": categories,
    }
    _save_json(out_json_path, out)
    print(f"  Wrote: {out_json_path}")


def main():
    ap = argparse.ArgumentParser(description="Clean/standardize COCO JSONs for REAL-Colon roots.")
    ap.add_argument("--root", type=Path, action="append", required=True,
                    help="Root directory containing annotations_coco_{train,val,test}.json")
    ap.add_argument("--out-dir-name", type=str, default="", help="Optional subdir name to write cleaned JSONs under each root")
    args = ap.parse_args()

    for root in args.root:
        print(f"Processing root: {root}")
        out_dir = (root / args.out_dir_name) if args.out_dir_name else root
        for subset, fname in INFILES.items():
            in_p = root / fname
            if not in_p.exists():
                print(f"  [WARN] Missing {in_p}, skipping subset '{subset}'")
                continue
            out_p = out_dir / (fname.replace(".json", ".clean.json"))
            clean_one(in_p, out_p)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
