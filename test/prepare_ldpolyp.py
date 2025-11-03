#!/usr/bin/env python3
"""
process_ldpolyp.py

Prepare LDPolypVideo into a clean, reproducible layout with COCO + YOLO outputs.

Input (as downloaded):
  /data/local/aschwab/data/ldpolyp/
    trainVal/
      images/
        1/ ... 100/
          0001.jpg ...
      annotations/
        1/ ... 100/
          0001.txt ...
    test/
      images/
        101/ ... 160/
          *.jpg
      annotations/
        101/ ... 160/
          *.txt

Output (created under --out-root, default: <source>/ldpolyp_clean/):
  ldpolyp_clean/
    images/{train,val,test}/caseXXX_0001.jpg ...
    labels/{train,val,test}/caseXXX_0001.txt  # YOLO format (empty file if negative)
    annotations_coco_train.json
    annotations_coco_val.json
    annotations_coco_test.json
    data.yaml

Splitting policy:
  - Keep official test set (cases 101–160) untouched.
  - From development cases 1–100, carve out ~10% for validation.
    Default (deterministic): cases 91–100 => validation; 1–90 => train.
    Overridable via --val-cases or --val-ratio/--seed.

Notes:
  - Images are NOT resized; labels are transformed as needed for YOLO and COCO only.
  - To preserve space, the script will hardlink images by default (fallback to copy).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

from PIL import Image

# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            return
        os.link(src, dst)  # hardlink
    except OSError:
        shutil.copy2(src, dst)


def parse_label_file(txt_path: Path) -> List[Tuple[float, float, float, float]]:
    """Return list of boxes as (x1, y1, x2, y2) in pixels. Empty list for negatives.
    Accepts either the specified 2-line positive format or a single '0'.
    Robust to extra whitespace and multiple boxes (if present line-wise)."""
    with txt_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    if len(lines) == 1 and lines[0] == "0":
        return []
    # Positive: may be '1' then one or more bbox lines; tolerate files that omit the leading '1'.
    if lines[0] == "1":
        lines = lines[1:]
    boxes = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        if len(parts) != 4:
            # Skip malformed lines gracefully
            continue
        try:
            x1, y1, x2, y2 = map(float, parts)
        except ValueError:
            continue
        # Normalize ordering just in case
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        boxes.append((x1, y1, x2, y2))
    return boxes


def clamp_box(x1, y1, x2, y2, w, h):
    x1c = max(0.0, min(x1, w - 1))
    y1c = max(0.0, min(y1, h - 1))
    x2c = max(0.0, min(x2, w - 1))
    y2c = max(0.0, min(y2, h - 1))
    # Ensure valid ordering after clamp
    x1c, x2c = min(x1c, x2c), max(x1c, x2c)
    y1c, y2c = min(y1c, y2c), max(y1c, y2c)
    return x1c, y1c, x2c, y2c


# -------------------------------
# COCO building
# -------------------------------
class COCOWriter:
    def __init__(self, out_json: Path):
        self.out_json = out_json
        self.images = []
        self.annotations = []
        self.categories = [
            {"id": 1, "name": "Lesion"}
        ]
        self._img_id = 1
        self._ann_id = 1

    def add_image(self, file_name: str, w: int, h: int, video_id: int, frame_id: int) -> int:
        img_id = self._img_id
        self._img_id += 1
        self.images.append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h,
            "video_id": video_id,
            "frame_id": frame_id,
        })
        return img_id

    def add_box(self, img_id: int, x1: float, y1: float, x2: float, y2: float):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            return
        ann_id = self._ann_id
        self._ann_id += 1
        self.annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    def save(self):
        out = {
            "info": {
                "description": "LDPolypVideo (cleaned) — original resolution; thesis split",
                "version": "1.0",
                "year": 2025,
                "contributor": "Andreas Schwab (Master's thesis)",
            },
            "licenses": [
                {"id": 1, "name": "Dataset license per LDPolypVideo", "url": "https://github.com/xiahaifeng1995/LDPolypVideo"}
            ],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        with self.out_json.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)


# -------------------------------
# Main processing
# -------------------------------

def pick_val_cases(cases_1_100: List[int], val_ratio: float, seed: int, deterministic_tail: bool, explicit: List[int] | None) -> List[int]:
    if explicit:
        ex = sorted(set(explicit))
        for c in ex:
            if c < 1 or c > 100:
                raise ValueError(f"Explicit val case {c} not in 1..100")
        return ex
    n_total = len(cases_1_100)
    n_val = max(1, int(round(n_total * val_ratio)))
    if deterministic_tail:
        return sorted(cases_1_100)[-n_val:]
    random.seed(seed)
    return sorted(random.sample(cases_1_100, n_val))


def iter_cases(src_root: Path, subset_dir: str) -> List[int]:
    """Discover case IDs by scanning the images/ subdirectory for the subset.
    Expected layout:
      <root>/<subset_dir>/images/<case_id>/...jpg
    """
    base_images = src_root / subset_dir / "images"
    cases = []
    if not base_images.is_dir():
        return cases
    for child in sorted(base_images.iterdir()):
        if child.is_dir():
            try:
                cid = int(child.name)
            except ValueError:
                continue
            cases.append(cid)
    return cases


def process_subset(src_root: Path, subset_name: str, case_ids: List[int], out_root: Path, coco_writer: COCOWriter):
    img_out_dir = out_root / "images" / subset_name
    lbl_out_dir = out_root / "labels" / subset_name
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    for case_id in sorted(case_ids):
        subset_root = src_root / ("trainVal" if case_id <= 100 else "test")
        img_dir = subset_root / "images" / f"{case_id}"
        ann_dir = subset_root / "annotations" / f"{case_id}"
        if not img_dir.is_dir() or not ann_dir.is_dir():
            print(f"[WARN] Missing images/annotations for case {case_id}; skipping")
            continue
        # Collect frames sorted by stem
        frames = sorted(img_dir.glob("*.jpg"))
        for img_path in frames:
            stem = img_path.stem  # e.g., 0001
            ann_path = ann_dir / f"{stem}.txt"
            if not ann_path.exists():
                print(f"[WARN] Missing annotation for {img_path}")
                boxes = []
            else:
                boxes = parse_label_file(ann_path)

            # Read image size lazily
            with Image.open(img_path) as im:
                w, h = im.size

            # Destination filename to avoid collisions across cases
            dst_name = f"case{case_id:03d}_{stem}.jpg"
            dst_img = img_out_dir / dst_name
            hardlink_or_copy(img_path, dst_img)

            # COCO: register image and boxes (clamped)
            img_id = coco_writer.add_image(
                file_name=f"images/{subset_name}/{dst_name}", w=w, h=h, video_id=case_id, frame_id=int(stem)
            )
            for (x1, y1, x2, y2) in boxes:
                x1c, y1c, x2c, y2c = clamp_box(x1, y1, x2, y2, w, h)
                coco_writer.add_box(img_id, x1c, y1c, x2c, y2c)

            # YOLO label file (class index 0, normalized cx,cy,w,h). Empty file for negatives
            ylbl = lbl_out_dir / f"case{case_id:03d}_{stem}.txt"
            if not boxes:
                ylbl.write_text("", encoding="utf-8")
            else:
                lines = []
                for (x1, y1, x2, y2) in boxes:
                    x1c, y1c, x2c, y2c = clamp_box(x1, y1, x2, y2, w, h)
                    bw, bh = max(0.0, x2c - x1c), max(0.0, y2c - y1c)
                    if bw <= 0 or bh <= 0:
                        continue
                    cx = (x1c + x2c) / 2.0 / w
                    cy = (y1c + y2c) / 2.0 / h
                    nw = bw / w
                    nh = bh / h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                ylbl.write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml(out_root: Path):
    # Write a real YAML file (not JSON) compatible with Ultralytics
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names: ['Lesion']",
        "",
    ]
    (out_root / "data.yaml").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Process LDPolypVideo into COCO + YOLO with a deterministic split.")
    ap.add_argument("--src-root", type=Path, required=True, help="Path to original ldpolyp root")
    ap.add_argument("--out-root", type=Path, default=None, help="Output root (default: <src-root>/ldpolyp_clean)")
    args = ap.parse_args()

    src_root: Path = args.src_root
    out_root: Path = args.out_root or (src_root / "ldpolyp_clean")
    ensure_dir(out_root / "images")
    ensure_dir(out_root / "labels")

    cases_trainval = sorted(iter_cases(src_root, "trainVal"))
    cases_test = sorted(iter_cases(src_root, "test"))

    # Deterministic split: last 10 of 1..100 => val; rest => train
    val_cases = [c for c in cases_trainval if 91 <= c <= 100]
    train_cases = [c for c in cases_trainval if 1 <= c <= 90]

    print(f"Train cases ({len(train_cases)}): {train_cases[:5]} ... {train_cases[-5:]}")
    print(f"Val cases   ({len(val_cases)}): {val_cases}")
    print(f"Test cases  ({len(cases_test)}): {cases_test[:5]} ... {cases_test[-5:]}")

    coco_train = COCOWriter(out_root / "annotations_coco_train.json")
    coco_val   = COCOWriter(out_root / "annotations_coco_val.json")
    coco_test  = COCOWriter(out_root / "annotations_coco_test.json")

    process_subset(src_root, "train", train_cases, out_root, coco_train)
    process_subset(src_root, "val",   val_cases,   out_root, coco_val)
    process_subset(src_root, "test",  cases_test,  out_root, coco_test)

    coco_train.save(); coco_val.save(); coco_test.save()
    write_data_yaml(out_root)

    print("Done. Output written to:")
    print(f"  {out_root}")
    print("  - images/{train,val,test}")
    print("  - labels/{train,val,test}")
    print("  - annotations_coco_{train,val,test}.json")
    print("  - data.yaml")


if __name__ == "__main__":
    main()

