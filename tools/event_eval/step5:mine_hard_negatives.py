#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
from typing import Dict, List
from event_eval.utils_event import load_gt_frames, load_pred_frames, link_tracks, match_tracks
from event_eval.utils_event import parse_stem

ap = argparse.ArgumentParser()
ap.add_argument('--base_ds', required=True, type=Path, help='Existing YOLO dataset root with images/{train,val,test}, labels/{...}')
ap.add_argument('--train_dump', required=True, type=Path, help='Directory for e.g. step1_sweeps/iou0.50_c0.05_train')
ap.add_argument('--out_ds', required=True, type=Path, help='New dataset root with added hard negatives')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--conf', type=float, required=True)
ap.add_argument('--link_iou', type=float, default=0.30)
ap.add_argument('--max_gap', type=int, default=3)
ap.add_argument('--min_len', type=int, default=2)
ap.add_argument('--match_iou', type=float, default=0.30)
ap.add_argument('--tiny_area_px', type=float, default=64)
ap.add_argument('--per_video_cap', type=int, default=50)
ap.add_argument('--total_cap', type=int, default=20000)
args = ap.parse_args()

src = args.base_ds
train_img = src / 'images' / 'train'
train_lbl = src / 'labels' / 'train'

# Load frames
GT = load_gt_frames(train_lbl, args.imgsz)
PR = load_pred_frames(args.train_dump, args.imgsz, min(0.05, args.conf))

# Build tracks
GT_tracks = link_tracks(GT, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)
PR_tracks = link_tracks(PR, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)

# Match to identify unmatched PR tracks (candidate hard negatives)
_, _, gt2pr, unmatched_pr = match_tracks(GT_tracks, PR_tracks, args.match_iou)

# Gather center frames of unmatched PR tracks, capped per video
by_vid: Dict[str, List[int]] = {}
# Build reverse map from (vid,fid)-> image path in train
idx: Dict[tuple, Path] = {}
for p in train_img.rglob('*.jpg'):
    try:
        vid, fid = parse_stem(p.stem)
        idx[(vid,fid)] = p
    except Exception:
        pass

candidates: List[Path] = []
for pi in unmatched_pr:
    t = PR_tracks[pi]
    vid = t.vid
    mid = t.frames[len(t.frames)//2]
    if len(by_vid.get(vid, [])) >= args.per_video_cap: continue
    ipath = idx.get((vid, mid))
    if ipath is None: continue
    by_vid.setdefault(vid, []).append(mid)
    candidates.append(ipath)
    if len(candidates) >= args.total_cap: break

# Materialize overlay dataset with added negatives
out = args.out_ds
if out.exists():
    raise SystemExit(f"out_ds exists: {out}")

for split in ('train','val','test'):
    for sub in ('images','labels'):
        src_dir = src/sub/split
        dst_dir = out/sub/split
        if src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            # hardlink all
            for p in src_dir.rglob('*'):
                if p.is_file():
                    d = dst_dir / p.relative_to(src_dir)
                    d.parent.mkdir(parents=True, exist_ok=True)
                    try: d.hardlink_to(p)
                    except Exception: shutil.copy2(p, d)

# add hard-neg frames to train (images + empty labels)
for ipath in candidates:
    rel = ipath.relative_to(train_img)
    dst_img = out / 'images' / 'train' / rel
    dst_lbl = out / 'labels' / 'train' / rel.with_suffix('.txt')
    # ensure unique name if collision
    k=0
    base = dst_img
    while dst_img.exists():
        k+=1
        dst_img = base.with_name(base.stem + f"_hardneg{k:02d}" + base.suffix)
        dst_lbl = dst_img.with_suffix('.txt').with_name(dst_img.stem + '.txt')
    # materialize
    try: dst_img.hardlink_to(ipath)
    except Exception: shutil.copy2(ipath, dst_img)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.write_text('', encoding='utf-8')  # empty label = negative

# write manifest
manifest = {
  'source_ds': str(src),
  'train_dump': str(args.train_dump),
  'imgsz': args.imgsz,
  'conf': args.conf,
  'link_iou': args.link_iou,
  'max_gap': args.max_gap,
  'min_len': args.min_len,
  'match_iou': args.match_iou,
  'tiny_area_px': args.tiny_area_px,
  'per_video_cap': args.per_video_cap,
  'total_cap': args.total_cap,
  'added_negatives': len(candidates)
}
(out / 'hardneg_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
