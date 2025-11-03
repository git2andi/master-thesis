#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from typing import List
from utils_event import (
    load_gt_frames, load_pred_frames,
    link_tracks, match_tracks, reaction_time_frames, auc_trapz
)

parser = argparse.ArgumentParser()
parser.add_argument('--labels', required=True, type=Path)
parser.add_argument('--pred_json', required=True, type=Path, help='path to predictions.json OR directory containing labels/')
parser.add_argument('--out_csv', required=True, type=Path)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--conf_sweep', type=str, default='0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90')
parser.add_argument('--link_iou', type=float, default=0.30)
parser.add_argument('--max_gap', type=int, default=3)
parser.add_argument('--min_len', type=int, default=2)
parser.add_argument('--match_iou', type=float, default=0.30)
parser.add_argument('--tiny_area_px', type=float, default=64)
parser.add_argument('--px_bins', type=str, default='0,16,32,1e9', help='min-side bins in px for size-binned sensitivity')
parser.add_argument('--progress', action='store_true')
args = parser.parse_args()

conf_vals = [float(x) for x in args.conf_sweep.split(',') if x.strip()]
conf_vals = sorted(set(conf_vals))
if not conf_vals:
    raise SystemExit("conf_sweep is empty")

conf_keep = min(conf_vals)

labels_dir = args.labels
pred_root = args.pred_json if args.pred_json.is_dir() else args.pred_json.parent

# Load frames (unchanged)
GT = load_gt_frames(labels_dir, args.imgsz)
PR_all = load_pred_frames(pred_root, args.imgsz, conf_keep)

# Original filter → re-link semantics
def filter_pr(PR_all, thr):
    from utils_event import Box
    out = {}
    for k, boxes in PR_all.items():
        bb = [Box(b.x1,b.y1,b.x2,b.y2,b.conf) for b in boxes if b.conf >= thr]
        if bb:
            out[k] = bb
    return out

# Pre-build GT tracks (fixed once, unchanged)
GT_tracks = link_tracks(GT, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)

# size bins
edges = [float(x) for x in args.px_bins.split(',') if x.strip()]
nbins = max(0, len(edges)-1)

def track_min_side_bin(t):
    if nbins <= 0:
        return 0
    mins = [min(b.x2-b.x1, b.y2-b.y1) for b in t.boxes]
    val = max(mins) if mins else 0.0
    for j in range(nbins):
        if edges[j] <= val < edges[j+1]:
            return j
    return nbins-1

# videos count (for FP/video)
vids = {vid for (vid,_) in GT.keys()}
n_videos = max(1, len(vids))

# Open CSV and stream rows
args.out_csv.parent.mkdir(parents=True, exist_ok=True)
with args.out_csv.open('w', newline='', encoding='utf-8') as f:
    # meta line (auc appended later)
    f.write(f"# meta imgsz={args.imgsz} link_iou={args.link_iou} max_gap={args.max_gap} "
            f"min_len={args.min_len} match_iou={args.match_iou} tiny_area_px={args.tiny_area_px}\n")

    fieldnames = [
        'conf','link_iou','max_gap','min_len','match_iou','tiny_area_px','imgsz',
        'n_videos','n_gt_tracks','n_pr_tracks',
        'sensitivity','fp_per_video','rt_median_frames','rt_p90_frames',
    ] + ([f'sens_bin{j}' for j in range(nbins)] if nbins>0 else [])

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader(); f.flush(); os.fsync(f.fileno())

    # For AUC–FROC (0..4)
    xs, ys = [], []

    for idx, conf in enumerate(conf_vals, 1):
        if args.progress:
            print(f"[{idx}/{len(conf_vals)}] conf={conf:.2f} → linking PR...", flush=True)

        PR = filter_pr(PR_all, conf)
        PR_tracks = link_tracks(PR, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)

        n_gt, n_pr, gt2pr, unmatched_pr = match_tracks(GT_tracks, PR_tracks, args.match_iou)

        tp = len(gt2pr)
        sens = tp / n_gt if n_gt>0 else 0.0
        fp_per_video = (len(unmatched_pr) / n_videos)

        # reaction time stats
        rts = []
        for gi, pi in gt2pr.items():
            rt = reaction_time_frames(GT_tracks[gi], PR_tracks[pi])
            if rt is not None: rts.append(rt)
        rt_med = int(sorted(rts)[len(rts)//2]) if rts else -1
        rt_p90 = int(sorted(rts)[int(0.9*(len(rts)-1))]) if rts else -1

        # size-binned sensitivity
        row = {
            'conf': conf,
            'link_iou': args.link_iou,
            'max_gap': args.max_gap,
            'min_len': args.min_len,
            'match_iou': args.match_iou,
            'tiny_area_px': args.tiny_area_px,
            'imgsz': args.imgsz,
            'n_videos': n_videos,
            'n_gt_tracks': n_gt,
            'n_pr_tracks': n_pr,
            'sensitivity': sens,
            'fp_per_video': fp_per_video,
            'rt_median_frames': rt_med,
            'rt_p90_frames': rt_p90,
        }
        if nbins>0:
            bins_tot = [0]*nbins
            bins_tp  = [0]*nbins
            for gi, g in enumerate(GT_tracks):
                b = track_min_side_bin(g)
                bins_tot[b]+=1
                if gi in gt2pr: bins_tp[b]+=1
            for j in range(nbins):
                row[f'sens_bin{j}'] = (bins_tp[j]/bins_tot[j] if bins_tot[j]>0 else 0.0)

        writer.writerow(row)
        f.flush(); os.fsync(f.fileno())

        xs.append(min(4.0, fp_per_video))
        ys.append(sens)

        if args.progress:
            print(f"   conf={conf:.2f} sens={sens:.4f} fp/video={fp_per_video:.4f}", flush=True)

    auc_0_4 = auc_trapz(xs, ys) if xs else 0.0
    f.write(f"# summary auc0_4={auc_0_4:.6f}\n")
    f.flush(); os.fsync(f.fileno())

print(f"Wrote {args.out_csv}")
