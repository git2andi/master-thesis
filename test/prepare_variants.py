#!/usr/bin/env python3
"""
Create REAL-Colon dataset variants by subsampling negatives in the TRAIN split only,
while keeping VAL/TEST intact. Works on an already resized YOLO-style dataset
(e.g., /data/local/aschwab/data/realColon_640x640) that has this structure:

  <SRC>/images/{train,val,test}/*.jpg
  <SRC>/labels/{train,val,test}/*.txt

For each requested ratio R, we create a new dataset <SRC>_neg{R}x with:
  - Train: keep ALL positives (frames whose label txt is non-empty) + sample
           floor(R * #positives_per_video) negatives (empty label files) per video.
           If a train video has 0 positives, keep up to --min-neg-if-no-pos negatives.
  - Val/Test: copy/link ALL frames (unchanged).

We use hardlinks by default for efficiency; fallback to copy if hardlink fails.
"""

import argparse
import os
import re
import random
from pathlib import Path
from typing import List, Tuple

def parse_args():
    p = argparse.ArgumentParser(description="Make negative-ratio variants from an existing resized YOLO dataset.")
    p.add_argument("--src", required=True, type=Path,
                   help="Path to existing resized dataset root (e.g., /data/local/.../realColon_640x640)")
    p.add_argument("--ratios", required=True, type=float, nargs="+",
                   help="Negative:positive ratios for TRAIN (e.g., 1 3 9)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling")
    p.add_argument("--min-neg-if-no-pos", type=int, default=200,
                   help="If a train video has zero positives, keep up to this many negatives")
    p.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink",
                   help="How to materialize files in the new dataset")
    p.add_argument("--dst-root-base", type=Path, default=None,
                   help="Optional base directory to place new variants. If None, siblings next to --src are created.")
    p.add_argument("--verbose", action="store_true", help="Print per-video details")
    return p.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_yaml(dst_root: Path):
    yaml = f"""# REAL-Colon YOLO data (negative-subset variants)
train: {dst_root.as_posix()}/images/train
val:   {dst_root.as_posix()}/images/val
test:  {dst_root.as_posix()}/images/test

nc: 1
names: ["polyp"]
"""
    (dst_root / "data.yaml").write_text(yaml)

def is_positive_label(lbl_path: Path) -> bool:
    try:
        if not lbl_path.exists():
            return False
        txt = lbl_path.read_text().strip()
        return len(txt) > 0
    except Exception:
        return False

VID_RE = re.compile(r'^(\d{3}-\d{3})')

def video_id_from_name(stem: str) -> str:
    """
    Try to extract leading 'SSS-VVV' from filename (e.g., '001-001_22997_resized').
    Fallback to 'unknown' group to avoid crashes.
    """
    m = VID_RE.match(stem)
    return m.group(1) if m else "unknown"

def link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    try:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            # copy
            from shutil import copy2
            copy2(src, dst)
    except Exception:
        # Fallback to copy if link fails (e.g., cross-device)
        from shutil import copy2
        copy2(src, dst)

def gather_split_files(src_root: Path, split: str) -> List[Tuple[Path, Path, str]]:
    """
    Return list of (img_path, lbl_path, video_id)
    """
    imgs_dir = src_root / "images" / split
    lbls_dir = src_root / "labels" / split
    items = []
    for img in sorted(imgs_dir.glob("*.jpg")):
        stem = img.stem
        lbl = lbls_dir / f"{stem}.txt"
        vid = video_id_from_name(stem)
        items.append((img, lbl, vid))
    return items

def make_variant(src_root: Path, dst_root: Path, ratio: float, seed: int,
                 min_neg_if_no_pos: int, link_mode: str, verbose: bool):
    assert (src_root / "images/train").exists(), f"Missing {src_root}/images/train"
    rng = random.Random(seed)

    # Prepare dirs
    for sp in ("train","val","test"):
        ensure_dir(dst_root / "images" / sp)
        ensure_dir(dst_root / "labels" / sp)

    # VAL/TEST: copy/link everything as-is (images & labels)
    for sp in ("val","test"):
        items = gather_split_files(src_root, sp)
        for img, lbl, _ in items:
            dst_img = dst_root / "images" / sp / img.name
            dst_lbl = dst_root / "labels" / sp / lbl.name
            link_or_copy(img, dst_img, link_mode)
            if lbl.exists():
                link_or_copy(lbl, dst_lbl, link_mode)
            else:
                dst_lbl.touch()

    # TRAIN: keep all positives; sample negatives per video
    items = gather_split_files(src_root, "train")

    # group by video
    from collections import defaultdict
    by_vid_pos = defaultdict(list)
    by_vid_neg = defaultdict(list)

    for img, lbl, vid in items:
        if is_positive_label(lbl):
            by_vid_pos[vid].append((img, lbl))
        else:
            by_vid_neg[vid].append((img, lbl))

    kept_total = 0
    pos_total = 0
    neg_total = 0

    videos = sorted(set(list(by_vid_pos.keys()) + list(by_vid_neg.keys())))

    for vid in videos:
        pos_list = by_vid_pos.get(vid, [])
        neg_list = by_vid_neg.get(vid, [])

        n_pos = len(pos_list)
        n_neg = len(neg_list)

        if n_pos > 0:
            k_neg = int(ratio * n_pos)
            k_neg = min(k_neg, n_neg)
            selected_neg = rng.sample(neg_list, k_neg) if k_neg < n_neg else neg_list
        else:
            # keep a small fixed set if the video has no positives at all
            k_neg = min(min_neg_if_no_pos, n_neg)
            selected_neg = rng.sample(neg_list, k_neg) if k_neg < n_neg else neg_list

        # materialize
        for (img, lbl) in pos_list + selected_neg:
            dst_img = dst_root / "images" / "train" / img.name
            dst_lbl = dst_root / "labels" / "train" / lbl.name
            link_or_copy(img, dst_img, link_mode)
            if lbl.exists():
                link_or_copy(lbl, dst_lbl, link_mode)
            else:
                dst_lbl.touch()

        if verbose:
            print(f"[{vid}] pos={n_pos:6d}  neg_total={n_neg:6d}  kept_neg={len(selected_neg):6d}")

        pos_total += n_pos
        neg_total += len(selected_neg)
        kept_total += n_pos + len(selected_neg)

    # Write data.yaml
    write_yaml(dst_root)

    # Manifest
    for sp in ("train","val","test"):
        imgs = sorted((dst_root / "images" / sp).glob("*.jpg"))
        with open(dst_root / f"manifest_{sp}.txt", "w") as f:
            for p in imgs:
                f.write(p.name + "\n")

    print(f"✔ Created {dst_root}")
    print(f"  TRAIN kept: positives={pos_total:,}  negatives={neg_total:,}  total={kept_total:,}")
    # Rough reality check: ensure there are labels for all imgs
    for sp in ("train","val","test"):
        n_img = len(list((dst_root / "images" / sp).glob("*.jpg")))
        n_lbl = len(list((dst_root / "labels" / sp).glob("*.txt")))
        if n_img != n_lbl:
            print(f"[WARN] {sp}: images({n_img}) != labels({n_lbl})")

def main():
    args = parse_args()
    src_root = args.src.resolve()
    assert (src_root / "images").exists() and (src_root / "labels").exists(), "SRC must contain images/ and labels/"
    parent = args.dst_root_base.resolve() if args.dst_root_base else src_root.parent

    # Make a variant for each ratio
    for r in args.ratios:
        suffix = f"neg{int(r) if abs(r - int(r)) < 1e-6 else str(r).replace('.','p')}x"
        dst_root = parent / f"{src_root.name}_{suffix}"
        make_variant(src_root, dst_root, ratio=r, seed=args.seed,
                     min_neg_if_no_pos=args.min_neg_if_no_pos,
                     link_mode=args.link_mode, verbose=args.verbose)

if __name__ == "__main__":
    main()
