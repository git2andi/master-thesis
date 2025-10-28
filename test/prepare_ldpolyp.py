#!/usr/bin/env python3
"""
LDPolyp → ldpolyp_clean (YOLO + COCO)

Input (current layout you showed):
  <SRC>/
    trainVal/
      images/<video>/*.jpg
      annotations/<video>/*.txt   # first line = N, then N lines of x1 y1 x2 y2 (pixels)
    test/
      images/<video>/*.jpg
      annotations/<video>/*.txt

Output:
  <DST>/                        # default: <SRC parent>/ldpolyp_clean
    images/{train,val,test}/<video>/<frame>.jpg
    labels/{train,val,test}/<video>/<frame>.txt        # YOLO: "0 cx cy w h"
    annotations_coco_{train,val,test}.json             # COCO: 1 category: id=1, name="lesion"
    data.yaml                                          # names: [lesion]
    val_videos.txt                                     # chosen video IDs for val (video-level split)

Notes:
- YOLO class index is 0 (required), COCO category id is 1 ("lesion"), as requested.
- Uses symlinks for images when possible; falls back to copy.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
COCO_CAT = {"id": 1, "name": "lesion"}  # single category, id=1
YOLO_CLASS_INDEX = 0                    # single class for YOLO

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, force_copy: bool):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def read_xyxy_txt(txt_path: Path) -> List[Tuple[float, float, float, float]]:
    """
    Parses LDPolyp per-frame TXT:
      line 1: N
      next N lines: x1 y1 x2 y2  (pixels)
    Returns list of (x1,y1,x2,y2) as floats, with x1<=x2, y1<=y2.
    """
    s = txt_path.read_text().strip().splitlines()
    if not s:
        return []
    try:
        n = int(s[0].strip())
    except Exception:
        n = 0
    boxes = []
    for line in s[1:1+n]:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        if len(nums) < 4:
            continue
        x1,y1,x2,y2 = nums[:4]
        # normalize ordering
        if x2 < x1: x1,x2 = x2,x1
        if y2 < y1: y1,y2 = y2,y1
        boxes.append((x1,y1,x2,y2))
    return boxes

def img_size(img_path: Path) -> Tuple[int,int]:
    im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    h, w = im.shape[:2]
    return w, h

def xyxy_to_yolo(x1,y1,x2,y2,w,h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0
    # normalize
    return (
        YOLO_CLASS_INDEX,
        max(0.0, min(1.0, cx / w)),
        max(0.0, min(1.0, cy / h)),
        max(0.0, min(1.0, bw / w)),
        max(0.0, min(1.0, bh / h)),
    )

def write_yolo(lbl_path: Path, yolo_boxes: List[Tuple[int,float,float,float,float]]):
    ensure_dir(lbl_path.parent)
    if not yolo_boxes:
        lbl_path.write_text("")  # negative frame => empty label
        return
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for (c,cx,cy,bw,bh) in yolo_boxes if bw>0 and bh>0]
    lbl_path.write_text("\n".join(lines))

def add_coco(imgs, anns, next_ids, split, rel_img_path: Path, w:int, h:int, xyxy_boxes: List[Tuple[float,float,float,float]]):
    img_id = next_ids["img"][split]
    imgs[split].append({
        "id": img_id,
        "file_name": str(rel_img_path).replace("\\","/"),
        "width": w, "height": h
    })
    for (x1,y1,x2,y2) in xyxy_boxes:
        bw = max(0.0, x2-x1)
        bh = max(0.0, y2-y1)
        if bw<=0 or bh<=0: 
            continue
        anns[split].append({
            "id": next_ids["ann"][split],
            "image_id": img_id,
            "category_id": COCO_CAT["id"],
            "bbox": [float(x1), float(y1), float(bw), float(bh)],
            "area": float(bw*bh),
            "iscrowd": 0
        })
        next_ids["ann"][split] += 1
    next_ids["img"][split] += 1

def gather_images(images_dir: Path) -> List[Path]:
    out = []
    if not images_dir.exists():
        return out
    for vid_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()], key=lambda p: (p.name.zfill(6))):
        for img in sorted(vid_dir.iterdir()):
            if is_image(img):
                out.append(img)
    return out

def main():
    ap = argparse.ArgumentParser(description="Prepare ldpolyp_clean for YOLO and COCO-based DETR")
    ap.add_argument("--src", required=True, type=Path,
                    help="Source LDPolyp root (…/ldpolyp)")
    ap.add_argument("--dst", type=Path, default=None,
                    help="Output root (default: <SRC parent>/ldpolyp_clean)")
    ap.add_argument("--val_ratio", type=float, default=0.10,
                    help="Fraction of TRAIN VIDEOS to put in val (video-level split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="Copy images instead of symlinking")
    args = ap.parse_args()

    src = args.src.resolve()
    if args.dst is None:
        dst = (src.parent / "ldpolyp_clean").resolve()
    else:
        dst = args.dst.resolve()

    # source dirs
    tr_imgs = src / "trainVal" / "images"
    tr_anns = src / "trainVal" / "annotations"
    te_imgs = src / "test"     / "images"
    te_anns = src / "test"     / "annotations"

    if not tr_imgs.exists() or not tr_anns.exists() or not te_imgs.exists() or not te_anns.exists():
        raise SystemExit("Expected folders missing. Need trainVal/{images,annotations} and test/{images,annotations}.")

    # list training videos (folders under images/)
    train_videos = sorted([p.name for p in tr_imgs.iterdir() if p.is_dir()], key=lambda s: s.zfill(6))
    if not train_videos:
        raise SystemExit(f"No video folders found under {tr_imgs}")

    # choose validation videos (video-level)
    rnd = random.Random(args.seed)
    vids = train_videos[:]
    rnd.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_ratio))
    val_videos = set(sorted(vids[:n_val]))
    train_videos_set = set(train_videos) - val_videos

    # prepare dst dirs
    out_images = {s: dst / "images" / s for s in ("train","val","test")}
    out_labels = {s: dst / "labels" / s for s in ("train","val","test")}
    for d in list(out_images.values()) + list(out_labels.values()):
        ensure_dir(d)

    # init COCO containers
    coco_images = {s: [] for s in ("train","val","test")}
    coco_anns   = {s: [] for s in ("train","val","test")}
    next_ids = {"img": {s: 1 for s in ("train","val","test")},
                "ann": {s: 1 for s in ("train","val","test")}}

    def process_split(images_root: Path, ann_root: Path, split: str, only_videos: set[str]|None=None):
        count = 0
        for vid_dir in sorted([p for p in images_root.iterdir() if p.is_dir()], key=lambda p: p.name.zfill(6)):
            vid = vid_dir.name
            if only_videos is not None and vid not in only_videos:
                continue
            for img_path in sorted(vid_dir.iterdir()):
                if not is_image(img_path):
                    continue
                stem = img_path.stem
                ann_txt = ann_root / vid / f"{stem}.txt"
                xyxy_boxes = read_xyxy_txt(ann_txt) if ann_txt.exists() else []

                w, h = img_size(img_path)
                # YOLO
                yolo_boxes = [xyxy_to_yolo(x1,y1,x2,y2,w,h) for (x1,y1,x2,y2) in xyxy_boxes if x2>x1 and y2>y1]

                rel = Path(vid) / img_path.name
                out_img = out_images[split] / rel
                out_lbl = out_labels[split] / rel.with_suffix(".txt")
                link_or_copy(img_path, out_img, force_copy=args.copy)
                write_yolo(out_lbl, yolo_boxes)

                # COCO
                rel_for_coco = Path("images") / split / rel
                add_coco(coco_images, coco_anns, next_ids, split, rel_for_coco, w, h, xyxy_boxes)
                count += 1
        return count

    # process train/val (video-level split)
    n_train = process_split(tr_imgs, tr_anns, "train", only_videos=train_videos_set)
    n_val   = process_split(tr_imgs, tr_anns, "val",   only_videos=val_videos)
    n_test  = process_split(te_imgs, te_anns, "test",  only_videos=None)

    # write data.yaml
    (dst / "data.yaml").write_text(
        f"""# LDPolyp → YOLO format
path: {dst}
train: images/train
val: images/val
test: images/test
names: [ lesion ]
"""
    )

    # write COCO jsons
    for split in ("train","val","test"):
        out_json = dst / f"annotations_coco_{split}.json"
        out_json.write_text(json.dumps({
            "images": coco_images[split],
            "annotations": coco_anns[split],
            "categories": [COCO_CAT],
        }))

    # record chosen val videos
    (dst / "val_videos.txt").write_text("\n".join(sorted(val_videos)))

    # summary
    print("\n== Summary ==")
    print(f"dst: {dst}")
    print(f"train images: {n_train}")
    print(f"val   images: {n_val}")
    print(f"test  images: {n_test}")
    for s in ("train","val","test"):
        nl = len(list((dst / "labels" / s).rglob("*.txt")))
        print(f"{s:5s} labels: {nl}")
    print("\nCOCO files:", *[f.name for f in dst.glob("annotations_coco_*.json")], sep="\n  ")
    print("Wrote:", dst / "data.yaml")
    print("Val videos recorded to:", dst / "val_videos.txt")
    print("Done.")
    
if __name__ == "__main__":
    main()
