#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import List
from utils_event import load_gt_frames, load_pred_frames, link_tracks, match_tracks, reaction_time_frames, auc_trapz

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
args = parser.parse_args()

conf_vals = [float(x) for x in args.conf_sweep.split(',')]
conf_vals = sorted(set(conf_vals))
conf_keep = min(conf_vals)

labels_dir = args.labels
pred_root = args.pred_json if args.pred_json.is_dir() else args.pred_json.parent

# Load frames
GT = load_gt_frames(labels_dir, args.imgsz)
PR_all = load_pred_frames(pred_root, args.imgsz, conf_keep)

# Build tracks once per conf threshold (we re-filter by conf each time from PR_all)
def filter_pr(PR_all, thr):
    from utils_event import Box
    out = {}
    for k, boxes in PR_all.items():
        bb = [Box(b.x1,b.y1,b.x2,b.y2,b.conf) for b in boxes if b.conf >= thr]
        if bb:
            out[k] = bb
    return out

# Pre-build GT tracks (fixed params)
GT_tracks = link_tracks(GT, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)

# size bins precompute
edges = [float(x) for x in args.px_bins.split(',') if x.strip()]
nbins = max(0, len(edges)-1)

def track_min_side_bin(t):
    # use max min-side across boxes (more conservative)
    mins = []
    for b in t.boxes:
        mins.append(min(b.x2-b.x1, b.y2-b.y1))
    val = max(mins) if mins else 0.0
    for j in range(nbins):
        if edges[j] <= val < edges[j+1]:
            return j
    return nbins-1 if nbins>0 else 0

# Run sweep
rows = []
for conf in conf_vals:
    PR = filter_pr(PR_all, conf)
    PR_tracks = link_tracks(PR, args.link_iou, args.max_gap, args.min_len, args.tiny_area_px)
    n_gt, n_pr, gt2pr, unmatched_pr = match_tracks(GT_tracks, PR_tracks, args.match_iou)
    tp = len(gt2pr)
    sens = tp / n_gt if n_gt>0 else 0.0
    # FP/video: count unmatched predicted tracks / #videos in VAL set
    vids = {vid for (vid,_) in GT.keys()}
    fp_per_video = (len(unmatched_pr) / max(1,len(vids)))
    # reaction time stats
    rts = []
    for gi, pi in gt2pr.items():
        rt = reaction_time_frames(GT_tracks[gi], PR_tracks[pi])
        if rt is not None: rts.append(rt)
    rt_med = int(sorted(rts)[len(rts)//2]) if rts else -1
    rt_p90 = int(sorted(rts)[int(0.9*(len(rts)-1))]) if rts else -1
    # size-binned sensitivity
    bins_tot = [0]*nbins
    bins_tp  = [0]*nbins
    for gi, g in enumerate(GT_tracks):
        b = track_min_side_bin(g)
        bins_tot[b]+=1
        if gi in gt2pr: bins_tp[b]+=1
    bins_sens = [ (bins_tp[j]/bins_tot[j] if bins_tot[j]>0 else 0.0) for j in range(nbins) ]
    rows.append({
        'conf': conf,
        'link_iou': args.link_iou,
        'max_gap': args.max_gap,
        'min_len': args.min_len,
        'match_iou': args.match_iou,
        'tiny_area_px': args.tiny_area_px,
        'imgsz': args.imgsz,
        'n_videos': len(vids), 'n_gt_tracks': n_gt, 'n_pr_tracks': n_pr,
        'sensitivity': sens,
        'fp_per_video': fp_per_video,
        'rt_median_frames': rt_med, 'rt_p90_frames': rt_p90,
        **{f'sens_bin{j}': bins_sens[j] for j in range(nbins)}
    })

# AUC–FROC on 0..4 FP/video
xs = [min(4.0, r['fp_per_video']) for r in rows]
ys = [r['sensitivity'] for r in rows]
pairs = sorted(zip(xs,ys))
xs_s, ys_s = zip(*pairs) if pairs else ([],[])
auc_0_4 = auc_trapz(list(xs_s), list(ys_s))

# write CSV with header + meta comment in line 1
args.out_csv.parent.mkdir(parents=True, exist_ok=True)
with args.out_csv.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    f.write(f"# meta imgsz={args.imgsz} link_iou={args.link_iou} max_gap={args.max_gap} min_len={args.min_len} match_iou={args.match_iou} tiny_area_px={args.tiny_area_px} auc0_4={auc_0_4:.6f}\n")
    if rows:
        w.writeheader(); w.writerows(rows)
print(f"Wrote {args.out_csv}  (rows={len(rows)}  auc0_4={auc_0_4:.6f})")
