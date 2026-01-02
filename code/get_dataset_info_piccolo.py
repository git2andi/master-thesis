#!/usr/bin/env python3
"""
PICCOLO quick sanity checks (original + converted split).
Usage:
  python piccolo_sanity.py
  python piccolo_sanity.py --orig /path/to/dataset --split /path/to/split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


# util
def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (p for p in root.rglob("*") if p.is_file())

def count_suffix_ci(root: Path, suffix: str) -> int:
    """Count root files - case-insensitive."""
    suf = suffix.lower()
    return sum(1 for p in _iter_files(root) if p.name.lower().endswith(suf))


def count_glob(root: Path, pattern: str) -> int:
    """Count root files with same glob pattern"""
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern) if _.is_file())


def stem_set(root: Path, ext: str) -> set[str]:
    """Make common stem for root files"""
    if not root.exists():
        return set()
    ext = ext.lower()
    out = set()
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() == ext:
            out.add(p.stem)
    return out


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_yolo_lines(path: Path) -> int:
    """Count non-empty lines in YOLO label (means its negative)"""
    try:
        txt = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = path.read_text(encoding="latin-1")
    return sum(1 for ln in txt.splitlines() if ln.strip())


# Checks
def check_original(orig: Path) -> None:
    print("## Original PICCOLO")
    totals = {"masks": 0, "void": 0, "polyps": 0}
    for split in ("train", "validation", "test"):
        masks = count_suffix_ci(orig / split / "masks", "corrected.tif")
        voids = count_suffix_ci(orig / split / "void", "void.tif")
        polyps = count_suffix_ci(orig / split / "polyps", ".png")
        totals["masks"] += masks
        totals["void"] += voids
        totals["polyps"] += polyps
        print(f"{split:10s}  masks={masks:4d}  void={voids:4d}  polyps={polyps:4d}")
    print(f"{'TOTAL':10s}  masks={totals['masks']:4d}  void={totals['void']:4d}  polyps={totals['polyps']:4d}")
    print()


def check_converted_basic(split_base: Path) -> None:
    print("## Converted split (images/labels)")
    for split in ("train", "val", "test"):
        imgs_dir = split_base / "images" / split
        lbls_dir = split_base / "labels" / split
        n_imgs = count_glob(imgs_dir, "*.jpg")
        n_lbls = count_glob(lbls_dir, "*.txt")
        flag = "" if n_imgs == n_lbls else "  [WARN: images!=labels]"
        print(f"{split:5s}  images={n_imgs:4d}  labels={n_lbls:4d}{flag}")
    print()


def check_coco_consistency(split_base: Path) -> None:
    print("## COCO / YOLO consistency")

    split_specs: Tuple[Tuple[str, str], ...] = (
        ("train", "coco_annotations_train.json"), # Ensure naming is correct
        ("val",   "coco_annotations_val.json"),
        ("test",  "coco_annotations_test.json"),
    )

    overall = {"frames": 0, "pos": 0, "neg": 0, "boxes": 0, "yolo_mismatch": 0, "missing_lbl": 0}

    for split, coco_name in split_specs:
        coco_path = split_base / coco_name
        # TODO make error check

        coco = read_json(coco_path)
        images = coco.get("images", [])
        anns = coco.get("annotations", [])

        # ann count per iid
        ann_by_img: Dict[int, int] = {}
        for a in anns:
            img_id = a.get("image_id")
            if img_id is not None:
                ann_by_img[img_id] = ann_by_img.get(img_id, 0) + 1

        labels_dir = split_base / "labels" / split
        stem_to_lbl = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

        frames = len(images)
        boxes = len(anns)
        pos = 0
        neg = 0
        yolo_mismatch = 0
        missing_lbl = 0

        for img in images:
            img_id = img.get("id")
            fname = img.get("file_name", "")
            stem = Path(fname).stem

            gt = ann_by_img.get(img_id, 0)
            if gt > 0:
                pos += 1
            else:
                neg += 1

            lbl = stem_to_lbl.get(stem)
            if lbl is None:
                missing_lbl += 1
                continue

            if count_yolo_lines(lbl) != gt:
                yolo_mismatch += 1

        overall["frames"] += frames
        overall["pos"] += pos
        overall["neg"] += neg
        overall["boxes"] += boxes
        overall["yolo_mismatch"] += yolo_mismatch
        overall["missing_lbl"] += missing_lbl

        print(
            f"{split:5s}  frames={frames:4d}  boxes={boxes:4d}  pos={pos:4d}  neg={neg:4d}  "
            f"missing_lbl={missing_lbl:3d}  yolo_mismatch={yolo_mismatch:3d}"
        )

    print(
        f"{'TOTAL':5s}  frames={overall['frames']:4d}  boxes={overall['boxes']:4d}  "
        f"pos={overall['pos']:4d}  neg={overall['neg']:4d}  "
        f"missing_lbl={overall['missing_lbl']:3d}  yolo_mismatch={overall['yolo_mismatch']:3d}"
    )
    print()


# Main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=Path, default=Path("/data/local/aschwab/data/piccolo"))
    ap.add_argument("--split", type=Path, default=Path("/data/local/aschwab/data/piccolo_split"))
    args = ap.parse_args()

    check_original(args.orig)
    check_converted_basic(args.split)
    check_coco_consistency(args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
