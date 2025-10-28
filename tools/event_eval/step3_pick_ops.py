#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path

def row_distance(fpv, target):
    return abs(fpv - target)

ap = argparse.ArgumentParser()
ap.add_argument('--csv_dir', required=True, type=Path)
ap.add_argument('--targets', required=True, type=str, help='comma list of target FP/video, e.g., "2,4"')
ap.add_argument('--out_json', required=True, type=Path)
# optional filters to ensure consistent tracker config
ap.add_argument('--filter_liou', type=float)
ap.add_argument('--filter_gap', type=int)
ap.add_argument('--filter_minlen', type=int)
ap.add_argument('--filter_miou', type=float)
ap.add_argument('--filter_tapx', type=float)
args = ap.parse_args()

targets = [float(x) for x in args.targets.split(',')]
rows = []
for csvf in sorted(args.csv_dir.glob('*.csv')):
    with csvf.open('r', encoding='utf-8') as f:
        header_line = f.readline()
        reader = csv.DictReader(f)
        for r in reader:
            r['__src'] = csvf.name
            # cast
            for k in ('conf','fp_per_video','sensitivity','link_iou','match_iou','tiny_area_px'):
                if k in r and r[k] != '': r[k] = float(r[k])
            for k in ('max_gap','min_len','imgsz','n_videos','n_gt_tracks','n_pr_tracks'):
                if k in r and r[k] != '': r[k] = int(float(r[k]))
            # optional filters
            if args.filter_liou is not None and r.get('link_iou') != args.filter_liou: continue
            if args.filter_gap  is not None and r.get('max_gap')  != args.filter_gap: continue
            if args.filter_minlen is not None and r.get('min_len') != args.filter_minlen: continue
            if args.filter_miou is not None and r.get('match_iou') != args.filter_miou: continue
            if args.filter_tapx is not None and r.get('tiny_area_px') != args.filter_tapx: continue
            rows.append(r)

locked = []
for t in targets:
    # closest fp/video; tie-break by higher sens, then higher conf
    candidates = sorted(rows, key=lambda r: (row_distance(r['fp_per_video'], t), -r['sensitivity'], -r['conf']))
    if not candidates:
        raise SystemExit('No rows found to pick from; check filters/csv_dir')
    pick = candidates[0]
    locked.append({
        'target_fp_per_video': t,
        'conf': pick['conf'],
        'iou_dump': float(pick['__src'].split('_')[0].replace('iou','')),  # from filename iou0.50_...
        'imgsz': pick['imgsz'],
        'tiny_area_px': pick['tiny_area_px'],
        'link_iou': pick['link_iou'],
        'max_gap': pick['max_gap'],
        'min_len': pick['min_len'],
        'match_iou': pick['match_iou'],
        'n_videos_val': pick['n_videos'],
        'n_gt_tracks_val': pick['n_gt_tracks'],
        'source_csv': pick['__src']
    })

args.out_json.parent.mkdir(parents=True, exist_ok=True)
args.out_json.write_text(json.dumps({'locked_ops': locked}, indent=2), encoding='utf-8')
print(f"Wrote {args.out_json} with {len(locked)} ops")
