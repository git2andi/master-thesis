#!/usr/bin/env python3
"""
prepare_positives_adapted.py

Creates a *positives-augmented* dataset variant from a base YOLO layout while avoiding the pitfalls
we discussed:

✔ Robust event grouping with configurable frame-id GAP tolerance (no fid==last+1 assumption)
✔ Event-centered densification around FIRST and PEAK-AREA frames with an optional STRIDE
✔ Clear per-video CSV summary (originals and duplicates broken down by source)
✔ Size-bin quotas that count *effective* samples (originals + duplicates)
✔ Safer video_id/fid parsing + reentrancy/overwrite guards
✔ Hardlink-first materialization (falls back to copy)

Expected base layout (SRC):
  SRC/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt

Output (DST):
  DST/
    images/{train,val,test}/*.jpg      (hardlinked full copy from SRC + added duplicates in train)
    labels/{train,val,test}/*.txt
  DST/posdup_summary.csv               (per-video counts)
  DST/posdup_manifest.json             (global summary + args)

Notes:
- This script duplicates ONLY positives in the *train* split. Val/Test are copied 1:1.
- Duplicates always come with matching label files.
- By default, we assume YOLO txt labels with normalized (cx, cy, w, h), class id as the first token.
- Frame-id (fid) is parsed as the LAST integer run found in the image STEM; if missing, an index fallback is used.

Usage example:
  python prepare_positives_adapted.py \
    --src /data/local/aschwab/data/realColon_640x640 \
    --dst /data/local/aschwab/data/realColon_640x640_pos3x_evt \
    --imgsz 640 \
    --pos-multiplier 3 \
    --event-gap 5 \
    --event-window 15 \
    --event-stride 2 \
    --min-per-bin 400,350,250 \
    --px-bins 0,16,32,1e9 \
    --seed 42

"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Helpers
# -----------------------------

def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.link(src, dst)
    except OSError:
        # cross-device or FS does not support hardlinks
        shutil.copy2(src, dst)

_int_run_re = re.compile(r"(\d+)")
_video_id_re = re.compile(r"(\d{3}-\d{3})")


def extract_video_id(p: Path) -> str:
    """Try to extract a stable video id like 001-003 from name or parent tree.
    Fallback to the deepest folder that looks like a video folder, else stem prefix.
    """
    m = _video_id_re.search(p.as_posix())
    if m:
        return m.group(1)
    # fallback to parent directory
    for parent in p.parents:
        m2 = _video_id_re.search(parent.name)
        if m2:
            return m2.group(1)
    # last resort: use immediate parent name (prevents collapsing into "unknown")
    return p.parent.name or "unknown"


def extract_fid(stem: str) -> Optional[int]:
    """Return the last integer run in a filename stem, or None if not found."""
    matches = list(_int_run_re.finditer(stem))
    if not matches:
        return None
    return int(matches[-1].group(1))


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    out = []
    if not label_path.exists():
        return out
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # tolerate extra whitespace: keep only first 5 tokens
                parts = parts[:5]
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                out.append((cid, cx, cy, w, h))
            except Exception:
                # skip malformed line
                continue
    return out


def frame_box_stats(labels: List[Tuple[int, float, float, float, float]], imgsz: int) -> Tuple[float, float]:
    """Return (max_area_px2, max_min_side_px) across boxes in the frame.
    If no boxes: (0,0).
    """
    if not labels:
        return 0.0, 0.0
    max_area = 0.0
    max_min_side = 0.0
    for _, _, _, w, h in labels:
        area = (w * imgsz) * (h * imgsz)
        min_side = min(w, h) * imgsz
        if area > max_area:
            max_area = area
        if min_side > max_min_side:
            max_min_side = min_side
    return max_area, max_min_side


@dataclass
class FrameRec:
    img: Path
    lbl: Path
    video_id: str
    fid: Optional[int]
    order_idx: int  # stable order fallback when fid is None
    max_area_px2: float
    max_min_side_px: float


# -----------------------------
# Event grouping
# -----------------------------

def group_events(frames: List[FrameRec], event_gap: int) -> List[List[FrameRec]]:
    """Group frames (positives from a single video) into events using fid gaps.
    If fid is None, use order_idx to judge continuity. event_gap is applied to fid when available,
    else applied to order index.
    """
    if not frames:
        return []
    # Sort by a unified scalar key: fid if present else order_idx
    frames_sorted = sorted(frames, key=lambda r: (r.fid if r.fid is not None else r.order_idx))

    events: List[List[FrameRec]] = []
    current: List[FrameRec] = []

    def too_far(prev: FrameRec, nxt: FrameRec) -> bool:
        if prev.fid is not None and nxt.fid is not None:
            return (nxt.fid - prev.fid) > event_gap
        # fallback to order index continuity
        return (nxt.order_idx - prev.order_idx) > event_gap

    for rec in frames_sorted:
        if not current:
            current = [rec]
            continue
        if too_far(current[-1], rec):
            events.append(current)
            current = [rec]
        else:
            current.append(rec)
    if current:
        events.append(current)
    return events


# -----------------------------
# Materialization helpers
# -----------------------------

def make_dup_name(img: Path, suffix: str) -> Path:
    return img.with_name(f"{img.stem}_{suffix}{img.suffix}")


def materialize_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    hardlink_or_copy(src_img, dst_img)
    hardlink_or_copy(src_lbl, dst_lbl)


# -----------------------------
# Data.yaml writer
# -----------------------------

def write_data_yaml(dst_root: Path, class_names: Optional[List[str]] = None) -> None:
    class_names = class_names or ["lesion"]
    train_dir = dst_root / "images" / "train"
    val_dir = dst_root / "images" / "val"
    test_dir = dst_root / "images" / "test"
    if not train_dir.exists():
        raise SystemExit(f"Cannot write data.yaml: missing {train_dir}")
    if not val_dir.exists():
        raise SystemExit(f"Cannot write data.yaml: missing {val_dir}")
    # Compose minimal Ultralytics YAML
    names_yaml = ", ".join(f"'{n}'" for n in class_names)
    yaml_txt = [
        f"path: {dst_root.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if test_dir.exists():
        yaml_txt.append("test: images/test")
    yaml_txt.append(f"nc: {len(class_names)}")
    yaml_txt.append(f"names: [{names_yaml}]")
    (dst_root / "data.yaml").write_text("\n".join(yaml_txt) + "\n", encoding="utf-8")

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Base YOLO dataset root (with images/ and labels/)")
    ap.add_argument("--dst", type=Path, required=True, help="Output dataset root (will be created)")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (for size-bin calculations)")

    # Positive frame duplication
    ap.add_argument("--pos-multiplier", type=int, default=1, help="Duplicate each positive frame this many times (1 = none)")

    # Event densification
    ap.add_argument("--event-gap", type=int, default=5, help="Max allowed gap between consecutive positive frames to stay in the same event (frames)")
    ap.add_argument("--event-window", type=int, default=15, help="Window size around anchors (FIRST & PEAK) in frame-id units")
    ap.add_argument("--event-stride", type=int, default=1, help="Duplicate only every kth frame in the anchor window (1 = every frame)")
    ap.add_argument("--event-max-dups-per-event", type=int, default=999999, help="Cap total event-based duplicates per event (after stride)")

    # Size-bin quotas (effective samples = originals + duplicates)
    ap.add_argument("--px-bins", type=str, default="0,16,32,1e9", help="Comma-separated bin edges in *pixels* for min-side; e.g., '0,16,32,1e9' -> 3 bins: [0,16), [16,32), [32,inf)")
    ap.add_argument("--min-per-bin", type=str, default="", help="Comma list of minimum effective samples per bin (same length as computed bins). Leave empty to disable.")

    # Misc
    ap.add_argument("--class-names", type=str, default="lesion", help="Comma-separated class names for data.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fail-if-exists", action="store_true", help="Abort if DST already exists")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    class_names = [s.strip() for s in args.class_names.split(',') if s.strip()] or ["lesion"]

    src = args.src
    dst = args.dst

    if dst.exists():
        if args.fail_if_exists:
            raise SystemExit(f"DST already exists: {dst}")
        # If exists, ensure it's empty-ish: allow creating fresh content
        if any(dst.iterdir()):
            raise SystemExit(f"DST exists and is not empty: {dst}")

    # 1) Replicate SRC -> DST via hardlinks for *all* splits first
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            src_dir = src / sub / split
            dst_dir = dst / sub / split
            if not src_dir.exists():
                # Allow missing val/test, but require train exists
                if split == "train":
                    raise SystemExit(f"Missing required directory: {src_dir}")
                else:
                    continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in src_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src_dir)
                    hardlink_or_copy(p, dst_dir / rel)

    # 2) Index train positives
    train_img_dir = src / "images" / "train"
    train_lbl_dir = src / "labels" / "train"
    if not train_img_dir.exists() or not train_lbl_dir.exists():
        raise SystemExit("Train split missing under images/ or labels/")

    # Map image -> label
    img_paths = sorted([p for p in train_img_dir.rglob("*.jpg")])
    order_idx = 0
    positives: List[FrameRec] = []
    by_video: Dict[str, List[FrameRec]] = defaultdict(list)

    for img in img_paths:
        lbl = train_lbl_dir / img.relative_to(train_img_dir)
        lbl = lbl.with_suffix(".txt")
        labels = read_yolo_labels(lbl)
        if not labels:
            order_idx += 1
            continue  # negative; ignore for this positives-focused script
        vid = extract_video_id(img)
        fid = extract_fid(img.stem)
        max_area, max_min_side = frame_box_stats(labels, args.imgsz)
        rec = FrameRec(img=img, lbl=lbl, video_id=vid, fid=fid, order_idx=order_idx, max_area_px2=max_area, max_min_side_px=max_min_side)
        positives.append(rec)
        by_video[vid].append(rec)
        order_idx += 1

    if args.verbose:
        print(f"Found {len(positives)} positive frames across {len(by_video)} videos in train.")

    # 3) Build events per video
    events_by_video: Dict[str, List[List[FrameRec]]] = {}
    total_events = 0
    for vid, frames in by_video.items():
        evts = group_events(frames, event_gap=args.event_gap)
        events_by_video[vid] = evts
        total_events += len(evts)
    if args.verbose:
        print(f"Built {total_events} events across {len(by_video)} videos (gap={args.event_gap}).")

    # Precompute anchor windows per event
    anchor_windows: Dict[str, List[List[FrameRec]]] = defaultdict(list)
    def within_window(a: FrameRec, b: FrameRec) -> bool:
        if a.fid is not None and b.fid is not None:
            return abs(a.fid - b.fid) <= args.event_window
        return abs(a.order_idx - b.order_idx) <= args.event_window

    for vid, evts in events_by_video.items():
        for evt in evts:
            evt_sorted = sorted(evt, key=lambda r: (1 if r.fid is None else 0, r.fid if r.fid is not None else r.order_idx))
            first_anchor = evt_sorted[0]
            peak_anchor = max(evt_sorted, key=lambda r: r.max_area_px2)
            win_set = []
            for r in evt_sorted:
                if within_window(first_anchor, r) or within_window(peak_anchor, r):
                    win_set.append(r)
            # stride
            if args.event_stride > 1:
                win_set = [r for i, r in enumerate(win_set) if (i % args.event_stride) == 0]
            # cap per-event dups
            if len(win_set) > args.event_max_dups_per_event:
                win_set = win_set[:args.event_max_dups_per_event]
            anchor_windows[vid].append(win_set)

    # 4) Materialize duplicates in DST/train
    dup_counts_per_img: Counter[Path] = Counter()  # counts of duplicates per *original* image
    src_train_img = src / "images" / "train"
    src_train_lbl = src / "labels" / "train"
    dst_train_img = dst / "images" / "train"
    dst_train_lbl = dst / "labels" / "train"

    # a) pos-multiplier
    if args.pos_multiplier > 1:
        for rec in positives:
            for k in range(1, args.pos_multiplier):
                suffix = f"posdup{k:02d}"
                new_img = make_dup_name(dst_train_img / rec.img.relative_to(src_train_img), suffix)
                new_lbl = make_dup_name(dst_train_lbl / rec.lbl.relative_to(src_train_lbl), suffix)
                materialize_pair(rec.img, rec.lbl, new_img, new_lbl)
                dup_counts_per_img[rec.img] += 1

    # b) event-based densification (one extra copy per selected frame)
    evt_dup_counter = 0
    for vid, win_lists in anchor_windows.items():
        for win_set in win_lists:
            for rec in win_set:
                suffix = "evtdup"
                # count up for uniqueness per *image*
                c = dup_counts_per_img[rec.img]
                new_img = make_dup_name(dst_train_img / rec.img.relative_to(src_train_img), f"{suffix}{c:02d}")
                new_lbl = make_dup_name(dst_train_lbl / rec.lbl.relative_to(src_train_lbl), f"{suffix}{c:02d}")
                materialize_pair(rec.img, rec.lbl, new_img, new_lbl)
                dup_counts_per_img[rec.img] += 1
                evt_dup_counter += 1

    # 5) Size-bin quotas (effective samples = originals + duplicates)
    # Parse bins and quotas
    edges = [float(x) for x in args.px_bins.split(',') if x.strip()]
    if sorted(edges) != edges:
        raise SystemExit("px-bins must be non-decreasing, e.g., 0,16,32,1e9")
    # bins: [e0,e1), [e1,e2), ..., [e_{n-2}, e_{n-1}), [e_{n-1}, inf) if last is big number
    # For simplicity we treat provided edges as exact; last edge can be 1e9.
    nbins = max(0, len(edges) - 1)
    min_per_bin: Optional[List[int]] = None
    if args.min_per_bin.strip():
        min_per_bin = [int(x) for x in args.min_per_bin.split(',') if x.strip()]
        if len(min_per_bin) != nbins:
            raise SystemExit(f"min-per-bin must have length {nbins} for px-bins edges {edges}")

    def bin_index(min_side_px: float) -> int:
        # find j s.t. edges[j] <= val < edges[j+1]
        for j in range(nbins):
            if edges[j] <= min_side_px < edges[j+1]:
                return j
        return nbins - 1 if nbins > 0 else 0

    if min_per_bin is not None and nbins > 0:
        # Count effective samples per bin
        # Originals
        bin_orig_counts = [0] * nbins
        frames_by_bin: List[List[FrameRec]] = [[] for _ in range(nbins)]
        for rec in positives:
            b = bin_index(rec.max_min_side_px)
            bin_orig_counts[b] += 1
            frames_by_bin[b].append(rec)
        # Add duplicates already created
        bin_eff_counts = bin_orig_counts[:]
        for rec in positives:
            b = bin_index(rec.max_min_side_px)
            bin_eff_counts[b] += dup_counts_per_img[rec.img]
        # Determine deficits
        deficits = [max(0, m - c) for m, c in zip(min_per_bin, bin_eff_counts)]
        # Fill deficits by round-robin duplicating frames in that bin
        bindup_counter = 0
        for b, need in enumerate(deficits):
            if need <= 0 or not frames_by_bin[b]:
                continue
            idx = 0
            while need > 0:
                rec = frames_by_bin[b][idx % len(frames_by_bin[b])]
                c = dup_counts_per_img[rec.img]
                new_img = make_dup_name(dst_train_img / rec.img.relative_to(src_train_img), f"bindup{c:02d}")
                new_lbl = make_dup_name(dst_train_lbl / rec.lbl.relative_to(src_train_lbl), f"bindup{c:02d}")
                materialize_pair(rec.img, rec.lbl, new_img, new_lbl)
                dup_counts_per_img[rec.img] += 1
                bindup_counter += 1
                need -= 1
                idx += 1
        if args.verbose:
            print("Applied size-bin quotas. (Original, Effective)->Target per bin:")
            for j in range(nbins):
                eff = sum(1 for rec in positives if bin_index(rec.max_min_side_px) == j) + \
                      sum(dup_counts_per_img[rec.img] for rec in positives if bin_index(rec.max_min_side_px) == j)
                print(f"  Bin {j}: orig={bin_orig_counts[j]} eff={eff} target={min_per_bin[j]}")

    # 6) Per-video CSV summary
    summary_csv = dst / "posdup_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "orig_pos", "dup_total", "dup_posmult", "dup_event", "dup_bindup"])
        # Count dup types by parsing suffix tags from filenames in DST/train
        # We'll compute per-video sums based on the *original* image mapping
        per_vid_orig = Counter()
        per_vid_dups = Counter()
        per_vid_posmult = Counter()
        per_vid_evt = Counter()
        per_vid_bindup = Counter()

        # Map from original path -> video id for quick lookup
        orig_vid_map: Dict[Path, str] = {rec.img: rec.video_id for rec in positives}

        # Iterate duplicates we just created, based on dup_counts_per_img
        for rec in positives:
            vid = rec.video_id
            per_vid_orig[vid] += 1
            dups = dup_counts_per_img[rec.img]
            if dups:
                per_vid_dups[vid] += dups
        # Now more granular by scanning filenames in DST/train
        for p in dst_train_img.rglob("*.jpg"):
            # Only consider duplicates (those that have suffix markers)
            stem = p.stem
            if re.search(r"_(posdup|evtdup|bindup)\d+$", stem):
                # find matching SRC original by removing the last _tagNN
                base_stem = re.sub(r"_(posdup|evtdup|bindup)\d+$", "", stem)
                src_img = src_train_img / p.relative_to(dst_train_img)
                src_img = src_img.with_name(base_stem + src_img.suffix)
                # if base file exists in SRC, accumulate by its vid
                if src_img.exists() and src_img in orig_vid_map:
                    vid = orig_vid_map[src_img]
                    if "_posdup" in stem:
                        per_vid_posmult[vid] += 1
                    elif "_evtdup" in stem:
                        per_vid_evt[vid] += 1
                    elif "_bindup" in stem:
                        per_vid_bindup[vid] += 1

        # Emit rows
        vids = sorted(by_video.keys())
        for vid in vids:
            w.writerow([
                vid,
                per_vid_orig.get(vid, 0),
                per_vid_dups.get(vid, 0),
                per_vid_posmult.get(vid, 0),
                per_vid_evt.get(vid, 0),
                per_vid_bindup.get(vid, 0),
            ])

    # 7) Manifest JSON
    manifest = {
        "args": {**vars(args), "class_names": class_names},
        "counts": {
            "train_pos_orig": len(positives),
            "train_dup_total": int(sum(dup_counts_per_img.values())),
        },
        "notes": {
            "event_gap": "Max tolerated jump in fid/order_idx to keep frames within same event",
            "event_window": "Frames within +/- window of first & peak anchors are eligible (after stride)",
            "effective_samples": "Size-bin quotas count originals + duplicates",
        },
    }
    with (dst / "posdup_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # 8) Write data.yaml
    write_data_yaml(dst, class_names=class_names)

    print("Done.")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()