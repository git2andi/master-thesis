#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from event_eval.utils_event import load_gt_frames, load_pred_frames, link_tracks, match_tracks, reaction_time_frames

ap = argparse.ArgumentParser()
ap.add_argument('--labels_test', required=True, type=Path)
ap.add_argument('--pred_root', required=True, type=Path, help='root with step1_sweeps/iouX_cY_test/...')
ap.add_argument('--locked_ops', required=True, type=Path)
ap.add_argument('--out_csv', required=True, type=Path)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--px_bins', type=str, default='0,16,32,1e9')
args = ap.parse_args()

ops = json.loads(args.locked_ops.read_text())['locked_ops']

GT = load_gt_frames(args.labels_test, args.imgsz)
vids = {vid for (vid,_) in GT.keys()}
nb_videos = len(vids)

edges = [float(x) for x in args.px_bins.split(',') if x.strip()]
nbins = max(0, len(edges)-1)

def track_min_side_bin(t):
    mins = [min(b.x2-b.x1, b.y2-b.y1) for b in t.boxes]
    val = max(mins) if mins else 0.0
    for j in range(nbins):
        if edges[j] <= val < edges[j+1]:
            return j
    return nbins-1 if nbins>0 else 0

rows=[]
for op in ops:
    iou = float(op['iou_dump'])
    conf = float(op['conf'])
    keep_floor = min(0.05, conf)
    tag = f"iou{iou:.2f}_c{keep_floor:.2f}_test"
    pred_dir = args.pred_root / tag
    PR = load_pred_frames(pred_dir, args.imgsz, keep_floor)
    PR_tracks = link_tracks(PR, op['link_iou'], op['max_gap'], op['min_len'], op['tiny_area_px'])
    GT_tracks = link_tracks(GT, op['link_iou'], op['max_gap'], op['min_len'], op['tiny_area_px'])

    n_gt, n_pr, gt2pr, unmatched_pr = match_tracks(GT_tracks, PR_tracks, op['match_iou'])
    tp = len(gt2pr)
    sens = tp / n_gt if n_gt>0 else 0.0
    fp_per_video = (len(unmatched_pr) / max(1, nb_videos))

    rts = []
    for gi, pi in gt2pr.items():
        rt = reaction_time_frames(GT_tracks[gi], PR_tracks[pi])
        if rt is not None: rts.append(rt)
    rt_med = int(sorted(rts)[len(rts)//2]) if rts else -1
    rt_p90 = int(sorted(rts)[int(0.9*(len(rts)-1))]) if rts else -1

    bins_tot = [0]*nbins
    bins_tp  = [0]*nbins
    for gi, g in enumerate(GT_tracks):
        b = track_min_side_bin(g)
        bins_tot[b]+=1
        if gi in gt2pr: bins_tp[b]+=1
    bins_sens = [ (bins_tp[j]/bins_tot[j] if bins_tot[j]>0 else 0.0) for j in range(nbins) ]

    rows.append({
        'tag': tag, 'iou_dump': iou, 'conf': conf,
        'imgsz': args.imgsz,
        'link_iou': op['link_iou'], 'max_gap': op['max_gap'], 'min_len': op['min_len'], 'match_iou': op['match_iou'], 'tiny_area_px': op['tiny_area_px'],
        'n_videos': nb_videos, 'n_gt_tracks': n_gt, 'n_pr_tracks': n_pr,
        'sensitivity': sens, 'fp_per_video': fp_per_video,
        'rt_median_frames': rt_med, 'rt_p90_frames': rt_p90,
        **{f'sens_bin{j}': bins_sens[j] for j in range(nbins)}
    })

args.out_csv.parent.mkdir(parents=True, exist_ok=True)
with args.out_csv.open('w', newline='', encoding='utf-8') as f:
    if rows:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print(f"Wrote {args.out_csv}  (ops={len(rows)})")
