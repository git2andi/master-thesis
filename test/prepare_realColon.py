#!/usr/bin/env python3
"""
REAL-Colon → YOLO/COCO builder (split: 10/2/3 per cohort)

Modes:
  - variant=orig   : keep original images (no resize). YOLO labels normalized by original W,H.
  - variant=square : pre-letterbox to --imgsz (e.g., 224). Labels normalized by imgsz.

Outputs (unchanged layout as requested):
  {dst}/images/{train,val,test}/*.jpg
  {dst}/labels/{train,val,test}/*.txt
  {dst}/annotations_coco_{train,val,test}.json
  {dst}/data.yaml

Assumptions about SRC:
  SRC/
    frames/        (e.g., 001-001_frames/*.jpg)
    annotations/   (e.g., 001-001_annotations/*.xml)
"""

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from shutil import rmtree, copy2
from typing import Tuple, List, Dict

from PIL import Image, ImageOps  # pip install pillow

# -----------------------------
# COCO metadata (match helper)
# -----------------------------
COCO_INFO = {
    "description": "Cosmo data",
    "url": "http://cosmoimd.com",
    "version": "1.0",
    "year": 2023,
    "contributor": "CosmoIMD",
    "date_created": "2023/02/28",
}

COCO_LICENSES = [{
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
}]

CAT_ID = 1  # COCO category id for 'lesion' (1-based to match helper/MMDet)
COCO_CATEGORIES = [{"supercategory": "lesion", "id": CAT_ID, "name": "lesion"}]

# YOLO class mapping (txt files): 'lesion' → 0
CLASS_MAP = {"lesion": 0}

# -----------------------------
# Helpers: geometry & transforms
# -----------------------------
def letterbox_pil(img: Image.Image, new_shape: int, color=(114, 114, 114)) -> Tuple[Image.Image, float, int, int]:
    """Resize with unchanged aspect ratio using padding (like YOLO). Returns (img, scale, pad_left, pad_top)."""
    w, h = img.size
    r = min(new_shape / w, new_shape / h)
    new_unpad = (max(int(round(w * r)), 1), max(int(round(h * r)), 1))
    img = img.resize(new_unpad, Image.BILINEAR)
    pad_w = new_shape - new_unpad[0]
    pad_h = new_shape - new_unpad[1]
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_w - pad_left, pad_h - pad_top), fill=color)
    return img, r, pad_left, pad_top

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(w - 1.0, x1))
    y1 = max(0.0, min(h - 1.0, y1))
    x2 = max(0.0, min(w - 1.0, x2))
    y2 = max(0.0, min(h - 1.0, y2))
    return x1, y1, x2, y2

def yolo_norm_from_abs(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    """Absolute pixel bbox → YOLO normalized (cx,cy,w,h) given image size w,h."""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

# -----------------------------
# Split logic (10/2/3 per cohort)
# -----------------------------
VID_RE = re.compile(r"^(\d{3})-(\d{3})$")

def parse_vid_id(path_stem: str) -> tuple:
    m = VID_RE.match(path_stem)
    if not m:
        raise ValueError(f"Unexpected video id format: {path_stem}")
    return int(m.group(1)), int(m.group(2))

def build_split(video_ids: List[str]) -> Dict[str, str]:
    """Return vid → split (train/val/test) per cohort by vvv ranges."""
    by_cohort: Dict[int, List[int]] = {}
    for vid in video_ids:
        sss, vvv = parse_vid_id(vid)
        by_cohort.setdefault(sss, []).append(vvv)

    split_map: Dict[str, str] = {}
    for sss, vvvs in by_cohort.items():
        for v in sorted(vvvs):
            if 1 <= v <= 10:
                tag = "train"
            elif 11 <= v <= 12:
                tag = "val"
            else:
                tag = "test"
            split_map[f"{sss:03d}-{v:03d}"] = tag
    return split_map

# -----------------------------
# I/O helpers
# -----------------------------
def safe_symlink_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        rel_src = os.path.relpath(src, start=os.path.dirname(dst))
        os.symlink(rel_src, dst)
    except Exception:
        copy2(src, dst)

def write_yaml_files(dst_root: str):
    yaml_path = os.path.join(dst_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# REAL-Colon YOLO data
train: {os.path.join(dst_root, 'images/train')}
val:   {os.path.join(dst_root, 'images/val')}
test:  {os.path.join(dst_root, 'images/test')}

nc: 1
names: ["lesion"]
""")
    print(f"✔ Wrote {yaml_path}")

# -----------------------------
# COCO helpers
# -----------------------------
def coco_init_per_split():
    return {
        "train": {"info": COCO_INFO, "licenses": COCO_LICENSES, "images": [], "annotations": [], "categories": COCO_CATEGORIES},
        "val":   {"info": COCO_INFO, "licenses": COCO_LICENSES, "images": [], "annotations": [], "categories": COCO_CATEGORIES},
        "test":  {"info": COCO_INFO, "licenses": COCO_LICENSES, "images": [], "annotations": [], "categories": COCO_CATEGORIES},
    }

def add_coco_image(coco_split_dict: dict, split: str, img_id: int, file_name: str, w: int, h: int):
    coco_split_dict[split]["images"].append({
        "id": img_id, "file_name": file_name, "height": h, "width": w
    })

def add_coco_ann(coco_split_dict: dict, split: str, ann_id: int, img_id: int,
                 x1: float, y1: float, bw: float, bh: float):
    # rectangle segmentation
    seg = [[x1, y1, x1 + bw, y1, x1 + bw, y1 + bh, x1, y1 + bh]]
    coco_split_dict[split]["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": CAT_ID,      # 1-based
        "bbox": [x1, y1, bw, bh],
        "area": bw * bh,
        "iscrowd": 0,
        "segmentation": seg
    })

# -----------------------------
# Core builders
# -----------------------------
def process_variant_square(
    src_root: str, dst_root: str, imgsz: int, padding_color=(114, 114, 114), expect_60: bool = False
) -> None:
    """Build pre-letterboxed square dataset at imgsz."""
    # Clean output
    if os.path.exists(dst_root):
        print(f"Removing existing folder: {dst_root}")
        rmtree(dst_root)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(dst_root, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "labels", sp), exist_ok=True)

    coco_by_split = coco_init_per_split()
    ann_id_counter = {"train": 0, "val": 0, "test": 0}

    # Discover videos
    frame_folders = sorted(glob(os.path.join(src_root, "frames", "*_frames")))
    video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])
    if expect_60:
        assert len(video_ids) == 60, f"Expected 60 videos, found {len(video_ids)}"
    print(f"Found {len(video_ids)} videos.")
    split_map = build_split(video_ids)

    processed = skipped_xml = skipped_parse = 0

    for vid in video_ids:
        split = split_map.get(vid, "test")
        frame_dir = os.path.join(src_root, "frames", f"{vid}_frames")
        anno_dir  = os.path.join(src_root, "annotations", f"{vid}_annotations")

        img_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if not img_paths:
            print(f"[WARN] No frames for {vid}")
            continue

        for img_path in img_paths:
            base = Path(img_path).stem
            xml_path = os.path.join(anno_dir, f"{base}.xml")

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # Square letterbox
            img_sq, scale, pad_l, pad_t = letterbox_pil(img, new_shape=imgsz, color=padding_color)
            out_img = os.path.join(dst_root, "images", split, f"{base}.jpg")
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            img_sq.save(out_img, format="JPEG", quality=95, subsampling=0)

            # COCO image (square geometry)
            img_id = len(coco_by_split[split]["images"]) + 1
            add_coco_image(coco_by_split, split, img_id, f"{base}.jpg", w=imgsz, h=imgsz)

            # Labels
            lines: List[str] = []
            if os.path.exists(xml_path):
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall("object"):
                        cls_name = (obj.findtext("name") or "").strip().lower()
                        if cls_name not in CLASS_MAP:
                            continue
                        b = obj.find("bndbox")
                        if b is None:
                            continue
                        try:
                            xmin = float(b.findtext("xmin")); ymin = float(b.findtext("ymin"))
                            xmax = float(b.findtext("xmax")); ymax = float(b.findtext("ymax"))
                        except (TypeError, ValueError):
                            continue

                        # map to square coords
                        x1 = xmin * scale + pad_l
                        y1 = ymin * scale + pad_t
                        x2 = xmax * scale + pad_l
                        y2 = ymax * scale + pad_t
                        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, imgsz, imgsz)
                        bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
                        if bw <= 0.0 or bh <= 0.0:
                            continue

                        # YOLO txt (class 0)
                        cx, cy, ww, hh = yolo_norm_from_abs(x1, y1, x2, y2, w=imgsz, h=imgsz)
                        lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                        # COCO ann (square pixels) + segmentation
                        ann_id = ann_id_counter[split]
                        add_coco_ann(coco_by_split, split, ann_id, img_id, x1, y1, bw, bh)
                        ann_id_counter[split] += 1
                except ET.ParseError:
                    skipped_parse += 1
                except Exception:
                    skipped_parse += 1
            else:
                skipped_xml += 1

            out_lbl = os.path.join(dst_root, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("\n".join(lines))

            processed += 1

        print(f"✅ {vid} → {split} ({len(img_paths)} frames)")

    write_yaml_files(dst_root)
    for sp in ("train", "val", "test"):
        out_json = os.path.join(dst_root, f"annotations_coco_{sp}.json")
        with open(out_json, "w") as f:
            json.dump(coco_by_split[sp], f, indent=2)
        print(f"✔ Wrote {out_json}")

    print(f"\nDone. Converted {processed} frames.")
    print(f"Missing-XML frames: {skipped_xml}, XML-parse skips: {skipped_parse}")
    print(f"Dataset ready at: {dst_root}")

def process_variant_orig(
    src_root: str, dst_root: str, expect_60: bool = False
) -> None:
    """Build dataset that preserves each image's original dimensions (no resize)."""
    # Clean output
    if os.path.exists(dst_root):
        print(f"Removing existing folder: {dst_root}")
        rmtree(dst_root)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(dst_root, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "labels", sp), exist_ok=True)

    coco_by_split = coco_init_per_split()
    ann_id_counter = {"train": 0, "val": 0, "test": 0}

    # Discover videos
    frame_folders = sorted(glob(os.path.join(src_root, "frames", "*_frames")))
    video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])
    if expect_60:
        assert len(video_ids) == 60, f"Expected 60 videos, found {len(video_ids)}"
    print(f"Found {len(video_ids)} videos.")
    split_map = build_split(video_ids)

    processed = skipped_xml = skipped_parse = 0

    for vid in video_ids:
        split = split_map.get(vid, "test")
        frame_dir = os.path.join(src_root, "frames", f"{vid}_frames")
        anno_dir  = os.path.join(src_root, "annotations", f"{vid}_annotations")

        img_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if not img_paths:
            print(f"[WARN] No frames for {vid}")
            continue

        for img_path in img_paths:
            base = Path(img_path).stem
            xml_path = os.path.join(anno_dir, f"{base}.xml")

            # Link/copy image without re-encoding
            out_img = os.path.join(dst_root, "images", split, f"{base}.jpg")
            safe_symlink_or_copy(img_path, out_img)

            # Get original dims
            try:
                with Image.open(img_path) as im:
                    w0, h0 = im.size
            except Exception:
                continue

            # COCO image (original geometry)
            img_id = len(coco_by_split[split]["images"]) + 1
            add_coco_image(coco_by_split, split, img_id, f"{base}.jpg", w=w0, h=h0)

            # Labels
            lines: List[str] = []
            if os.path.exists(xml_path):
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall("object"):
                        cls_name = (obj.findtext("name") or "").strip().lower()
                        if cls_name not in CLASS_MAP:
                            continue
                        b = obj.find("bndbox")
                        if b is None:
                            continue
                        try:
                            xmin = float(b.findtext("xmin")); ymin = float(b.findtext("ymin"))
                            xmax = float(b.findtext("xmax")); ymax = float(b.findtext("ymax"))
                        except (TypeError, ValueError):
                            continue

                        x1, y1, x2, y2 = clip_box(xmin, ymin, xmax, ymax, w0, h0)
                        bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
                        if bw <= 0.0 or bh <= 0.0:
                            continue

                        # YOLO normalized to ORIGINAL dims (class 0)
                        cx, cy, ww, hh = yolo_norm_from_abs(x1, y1, x2, y2, w=w0, h=h0)
                        lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                        # COCO ann in ORIGINAL pixels + segmentation
                        ann_id = ann_id_counter[split]
                        add_coco_ann(coco_by_split, split, ann_id, img_id, x1, y1, bw, bh)
                        ann_id_counter[split] += 1
                except ET.ParseError:
                    skipped_parse += 1
                except Exception:
                    skipped_parse += 1
            else:
                skipped_xml += 1

            out_lbl = os.path.join(dst_root, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("\n".join(lines))

            processed += 1

        print(f"✅ {vid} → {split} ({len(img_paths)} frames)")

    write_yaml_files(dst_root)
    for sp in ("train", "val", "test"):
        out_json = os.path.join(dst_root, f"annotations_coco_{sp}.json")
        with open(out_json, "w") as f:
            json.dump(coco_by_split[sp], f, indent=2)
        print(f"✔ Wrote {out_json}")

    print(f"\nDone. Converted {processed} frames.")
    print(f"Missing-XML frames: {skipped_xml}, XML-parse skips: {skipped_parse}")
    print(f"Dataset ready at: {dst_root}")

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Build REAL-Colon YOLO/COCO datasets.")
    p.add_argument("--src", required=True, help="Root with frames/ and annotations/")
    p.add_argument("--dst", required=True, help="Output dataset root")
    p.add_argument("--variant", choices=["orig", "square"], default="square",
                   help="orig=preserve original dims; square=pre-letterbox to --imgsz")
    p.add_argument("--imgsz", type=int, default=224, help="Square size for variant=square")
    p.add_argument("--expect-60", action="store_true",
                   help="Assert there are exactly 60 videos (4 cohorts × 15).")
    args = p.parse_args()

    if args.variant == "square":
        process_variant_square(
            src_root=args.src,
            dst_root=args.dst,
            imgsz=args.imgsz,
            padding_color=(114, 114, 114),
            expect_60=args.expect_60
        )
    else:
        process_variant_orig(
            src_root=args.src,
            dst_root=args.dst,
            expect_60=args.expect_60
        )

if __name__ == "__main__":
    main()
