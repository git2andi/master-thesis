#!/usr/bin/env python3
import os
import re
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ======================================================================================
# REAL-Colon VOC → COCO for *resized* datasets (224x224 and 640x640), letterbox-aware.
# ======================================================================================
#
# Usage example:
#   python3 export_coco_format.py \
#       --base /data/local/aschwab/data/realColon \
#       --dst /data/local/aschwab/data/realColon_224x224 \
#       --dst /data/local/aschwab/data/realColon_640x640 \
#       --neg-ratio 1.0
#
# Notes:
# - Expects the usual REAL-Colon source layout under --base:
#       base/
#         frames/        (not used directly, we rely on filenames only)
#         annotations/   (contains SSS-VVV_annotations folders with XMLs)
# - Expects each --dst to contain:
#       data.yaml
#       images/train, images/val, images/test  (resized JPEGs/PNGs with original basenames)
# - Keeps ALL frames (positives+negatives) by default. To subsample negatives, use --neg-ratio <0..1].
# - Category id is kept at 0 ("lesion") for consistency with your pipeline.
#
# Output:
# - For each --dst, writes next to data.yaml:
#       annotations_coco_train.json
#       annotations_coco_val.json
#       annotations_coco_test.json
#
# ======================================================================================

PADDING_COLOR = (114, 114, 114)  # used for letterbox padding; for parity with your pipeline

def parse_voc_xml(xml_path: str) -> Tuple[int, int, List[Dict]]:
    """
    Parse a VOC XML and return (orig_w, orig_h, objects).
    objects: list of dict with keys: name, xmin, ymin, xmax, ymax
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))

    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        objs.append({
            "name": name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
        })
    return w, h, objs


def letterbox_params(orig_w: int, orig_h: int, new_shape: int) -> Tuple[float, int, int]:
    """
    Compute scale and symmetric padding to map original (W,H) → resized letterbox (new_shape,new_shape)
    Returns: (r, pad_left, pad_top)
    """
    r = min(new_shape / float(orig_w), new_shape / float(orig_h))
    new_unpad_w = max(int(round(orig_w * r)), 1)
    new_unpad_h = max(int(round(orig_h * r)), 1)

    pad_w = new_shape - new_unpad_w
    pad_h = new_shape - new_unpad_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return r, pad_left, pad_top


def transform_box_to_resized(
    xmin: float, ymin: float, xmax: float, ymax: float,
    scale: float, pad_left: int, pad_top: int, img_size_out: int
) -> Tuple[float, float, float, float]:
    """
    Transform a VOC (pixel) box into resized+letterboxed pixel coords.
    Returns (x1, y1, x2, y2), clipped to [0, img_size_out].
    """
    x1 = xmin * scale + pad_left
    y1 = ymin * scale + pad_top
    x2 = xmax * scale + pad_left
    y2 = ymax * scale + pad_top

    # clip
    x1 = max(0.0, min(float(img_size_out), x1))
    y1 = max(0.0, min(float(img_size_out), y1))
    x2 = max(0.0, min(float(img_size_out), x2))
    y2 = max(0.0, min(float(img_size_out), y2))
    return x1, y1, x2, y2


def bbox_xyxy_to_coco(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return float(x1), float(y1), float(w), float(h)


def infer_split_from_video(video_id: str) -> str:
    """
    Map 'SSS-VVV' → 'train'/'val'/'test' (1–10 train, 11–12 val, 13–15 test).
    """
    # video_id is like "001-011"
    m = re.match(r"^(\d{3})-(\d{3})$", video_id)
    if not m:
        return "train"  # fallback
    v = int(m.group(2))
    if 1 <= v <= 10:
        return "train"
    elif 11 <= v <= 12:
        return "val"
    else:
        return "test"


def build_index_for_dst(dst_root: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Enumerate resized images in dst_root/images/{train,val,test} and
    return index: split -> video_id -> [filenames (basenames)].
    Assumes filenames like "SSS-VVV_XXXXX.jpg".
    """
    index: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}
    imgs_root = os.path.join(dst_root, "images")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(imgs_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        for fn in os.listdir(split_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Expect prefix "SSS-VVV_"
            base = os.path.splitext(fn)[0]
            parts = base.split("_", 1)
            if len(parts) < 2:
                # Skip unexpected names
                continue
            video_id = parts[0]
            index[split].setdefault(video_id, []).append(fn)
        # sort filenames per video for reproducibility
        for k in index[split]:
            index[split][k].sort()
    return index


def locate_xml(base_root: str, video_id: str, image_basename: str) -> Optional[str]:
    """
    Given a video_id (SSS-VVV) and an image basename like 'SSS-VVV_XXXXX.jpg',
    return the path to the corresponding VOC XML in base_root/annotations/{video_id}_annotations/.
    """
    folder = f"{video_id}_annotations"
    xml_name = os.path.splitext(image_basename)[0] + ".xml"
    xml_path = os.path.join(base_root, "annotations", folder, xml_name)
    return xml_path if os.path.isfile(xml_path) else None


def build_coco_for_dst(
    base_root: str,
    dst_root: str,
    neg_ratio: float = 1.0,
    seed: int = 0
) -> None:
    """
    Build COCO JSONs (train/val/test) for one resized dataset root.
    Writes files next to data.yaml:
      annotations_coco_train.json / _val.json / _test.json
    """
    random.seed(seed)

    # Derive image size: look at one image file name length? Better: read from folder name or data.yaml path.
    # We infer from dst_root basename: e.g., ".../realColon_224x224" → 224; default to 640 if not parsable.
    m = re.search(r"_(\d{3,4})x\1$", os.path.basename(dst_root))
    img_size_out = int(m.group(1)) if m else 640

    # Enumerate images present in resized dataset
    index = build_index_for_dst(dst_root)

    # Prepare COCO skeletons per split
    categories = [{"id": 0, "name": "lesion"}]  # keep id=0 as in your pipeline
    def new_doc():
        return {
            "info": {"description": f"REAL-Colon resized {img_size_out} COCO", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

    docs = {"train": new_doc(), "val": new_doc(), "test": new_doc()}
    next_img_id = 1
    next_ann_id = 1

    # Collect negatives per-video per-split for optional sampling
    negatives_bucket: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}

    for split in ["train", "val", "test"]:
        for video_id, files in index[split].items():
            for fn in files:
                xml_path = locate_xml(base_root, video_id, fn)
                if not xml_path:
                    # If no XML exists, treat as negative with unknown orig size; skip to be safe.
                    # (You can change to raise if your dataset guarantees XML for all frames.)
                    continue
                try:
                    orig_w, orig_h, objs = parse_voc_xml(xml_path)
                except Exception as e:
                    # Skip malformed XMLs
                    continue

                # letterbox params
                r, pad_left, pad_top = letterbox_params(orig_w, orig_h, img_size_out)

                # Prepare COCO image entry
                img_entry = {
                    "id": next_img_id,
                    "width": img_size_out,
                    "height": img_size_out,
                    "file_name": fn  # basename; loaders use data.yaml paths
                }
                docs[split]["images"].append(img_entry)

                # Annotations (if any)
                if objs:
                    for obj in objs:
                        if obj.get("name", "").lower() != "lesion":
                            continue  # ignore any unexpected class
                        x1, y1, x2, y2 = transform_box_to_resized(
                            obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                            r, pad_left, pad_top, img_size_out
                        )
                        x, y, w, h = bbox_xyxy_to_coco(x1, y1, x2, y2)
                        # Skip degenerate
                        if w <= 0 or h <= 0:
                            continue
                        ann = {
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": 0,  # lesion
                            "iscrowd": 0,
                            "area": float(w * h),
                            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                            # rectangular segmentation polygon (optional but handy)
                            "segmentation": [[
                                round(x, 2), round(y, 2),
                                round(x + w, 2), round(y, 2),
                                round(x + w, 2), round(y + h, 2),
                                round(x, 2), round(y + h, 2),
                            ]],
                        }
                        docs[split]["annotations"].append(ann)
                        next_ann_id += 1
                else:
                    # Negative frame: store to bucket (may keep all later)
                    negatives_bucket.setdefault((split, video_id), []).append((next_img_id, fn))

                next_img_id += 1

    # Optional negative subsampling
    if neg_ratio < 1.0:
        keep_neg_ids = set()
        for (split, video_id), items in negatives_bucket.items():
            if not items:
                continue
            k = max(1, int(round(len(items) * neg_ratio))) if len(items) > 0 else 0
            # deterministic selection per video
            random.seed(hash((split, video_id, 1234567)) & 0xFFFFFFFF)
            selected = random.sample(items, k) if k < len(items) else items
            for img_id, _ in selected:
                keep_neg_ids.add(img_id)

        # Filter out negatives not selected (they have zero annotations already)
        for split in ["train", "val", "test"]:
            imgs = docs[split]["images"]
            anns = docs[split]["annotations"]
            # Find all image_ids that have at least one ann
            pos_img_ids = {a["image_id"] for a in anns}
            new_images = []
            for im in imgs:
                if (im["id"] in pos_img_ids) or (im["id"] in keep_neg_ids):
                    new_images.append(im)
            docs[split]["images"] = new_images
    # else: keep all negatives

    # Write files
    for split in ["train", "val", "test"]:
        out_path = os.path.join(dst_root, f"annotations_coco_{split}.json")
        with open(out_path, "w") as f:
            json.dump(docs[split], f)
        print(f"Wrote {out_path}  (images: {len(docs[split]['images'])}, ann: {len(docs[split]['annotations'])})")


def main():
    ap = argparse.ArgumentParser(description="REAL-Colon VOC → COCO for resized datasets (letterbox-aware).")
    ap.add_argument("--base", required=True, help="Path to REAL-Colon source root (has annotations/).")
    ap.add_argument("--dst", action="append", required=True,
                    help="Destination resized dataset root. Repeat for each (e.g., 224x224 and 640x640).")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="Fraction of negative frames to keep per video (default 1.0 = keep all).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for negative sampling.")
    args = ap.parse_args()

    base_root = args.base
    for dst_root in args.dst:
        print(f"Building COCO for: {dst_root}")
        build_coco_for_dst(base_root, dst_root, neg_ratio=args.neg_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
