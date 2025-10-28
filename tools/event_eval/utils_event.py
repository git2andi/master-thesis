#!/usr/bin/env python3
from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

STEM = re.compile(r"^(\d{3}-\d{3})_(\d+(?:\.\d+)?)$")

def parse_stem(stem: str) -> Tuple[str, int]:
    m = STEM.match(stem)
    if not m:
        raise ValueError(stem)
    return m.group(1), int(float(m.group(2)))

@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float; conf: float = 1.0
    def area(self) -> float: return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

@dataclass
class Track:
    vid: str
    frames: List[int]
    boxes: List[Box]
    def length(self) -> int: return len(self.frames)
    def max_area(self) -> float: return max((b.area() for b in self.boxes), default=0.0)
    def max_conf(self) -> float: return max((b.conf for b in self.boxes), default=0.0)

# ---- geometry

def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    w, h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = w * h
    den = a.area() + b.area() - inter
    return inter / den if den > 0 else 0.0

# ---- conversions

def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, imgsz: int) -> Tuple[float,float,float,float]:
    x1 = (cx - w/2) * imgsz
    y1 = (cy - h/2) * imgsz
    x2 = (cx + w/2) * imgsz
    y2 = (cy + h/2) * imgsz
    # clip
    x1 = max(0.0, min(imgsz, x1)); y1 = max(0.0, min(imgsz, y1))
    x2 = max(0.0, min(imgsz, x2)); y2 = max(0.0, min(imgsz, y2))
    return x1, y1, x2, y2

# ---- loaders

def load_gt_frames(labels_dir: Path, imgsz: int) -> Dict[Tuple[str,int], List[Box]]:
    out: Dict[Tuple[str,int], List[Box]] = {}
    for p in labels_dir.rglob('*.txt'):
        stem = p.stem
        try:
            vid, fid = parse_stem(stem)
        except Exception:
            continue
        boxes = []
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 5: continue
                cls = int(float(parts[0]))
                if cls != 0:  # single-class dataset expected
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                x1,y1,x2,y2 = yolo_to_xyxy(cx,cy,w,h,imgsz)
                boxes.append(Box(x1,y1,x2,y2, conf=1.0))
        out.setdefault((vid,fid), []).extend(boxes)
    return out

# Ultralytics save_txt format: one line per det: cls conf cx cy w h (normalized)
# We prefer reading predictions from labels/ if present; else from predictions.json (xywh pixel)

def load_pred_frames(pred_root: Path, imgsz: int, conf_keep: float) -> Dict[Tuple[str,int], List[Box]]:
    labels_dir = pred_root / 'labels'
    out: Dict[Tuple[str,int], List[Box]] = {}
    if labels_dir.exists():
        for p in labels_dir.rglob('*.txt'):
            stem = p.stem
            try: vid, fid = parse_stem(stem)
            except Exception: continue
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    parts=line.strip().split()
                    if len(parts) < 6: continue
                    cls = int(float(parts[0])); conf=float(parts[1])
                    if conf < conf_keep: continue
                    cx,cy,w,h = map(float, parts[2:6])
                    x1,y1,x2,y2 = yolo_to_xyxy(cx,cy,w,h,imgsz)
                    out.setdefault((vid,fid), []).append(Box(x1,y1,x2,y2, conf=conf))
        return out
    # fallback to predictions.json
    pj = pred_root / 'predictions.json'
    if pj.exists():
        data = json.loads(pj.read_text())
        # try ultralytics-like structure
        if isinstance(data, list):
            for item in data:
                name = item.get('name') or item.get('image') or ''
                stem = Path(name).stem
                try: vid,fid = parse_stem(stem)
                except Exception: continue
                boxes = item.get('boxes') or {}
                xywh = boxes.get('xywh') or []
                confs = boxes.get('conf') or []
                for (x,y,w,h),conf in zip(xywh, confs):
                    if conf < conf_keep: continue
                    x1,y1,x2,y2 = x - w/2, y - h/2, x + w/2, y + h/2
                    out.setdefault((vid,fid), []).append(Box(x1,y1,x2,y2, conf=float(conf)))
    return out

# ---- simple online linker (per video)

def link_tracks(frames: Dict[Tuple[str,int], List[Box]], link_iou: float, max_gap: int, min_len: int, tiny_area_px: float) -> List[Track]:
    by_vid: Dict[str, Dict[int, List[Box]]] = {}
    for (vid,fid), boxes in frames.items():
        by_vid.setdefault(vid, {})[fid] = [b for b in boxes if b.area() >= tiny_area_px]
    tracks: List[Track] = []
    for vid, fmap in by_vid.items():
        open_tracks: List[Track] = []
        for fid in sorted(fmap.keys()):
            dets = fmap[fid]
            used = [False]*len(dets)
            # try to extend existing tracks (best IoU)
            for t in open_tracks:
                # look back at last frame for t
                last_box = t.boxes[-1]
                best_i, best_iou = -1, 0.0
                for i, d in enumerate(dets):
                    if used[i]: continue
                    iouv = iou(last_box, d)
                    if iouv >= link_iou and iouv > best_iou:
                        best_i, best_iou = i, iouv
                if best_i >= 0:
                    t.frames.append(fid); t.boxes.append(dets[best_i]); used[best_i] = True
                else:
                    # check gap; if too large, finalize
                    if fid - t.frames[-1] > max_gap:
                        if t.length() >= min_len: tracks.append(t)
                    # keep it open otherwise
            # start new tracks for remaining dets
            for i, d in enumerate(dets):
                if not used[i]:
                    open_tracks.append(Track(vid=vid, frames=[fid], boxes=[d]))
        # finalize leftovers
        for t in open_tracks:
            if t.length() >= min_len:
                tracks.append(t)
    return tracks

# ---- matching & metrics

def match_tracks(gt: List[Track], pr: List[Track], match_iou: float) -> Tuple[int,int,Dict[int,int],List[int]]:
    # returns: (n_gt, n_pr, gt_to_pr, unmatched_pr_indices)
    gt_to_pr: Dict[int,int] = {}
    used_pr = set()
    for gi, g in enumerate(gt):
        best_p, best_s = -1, 0.0
        for pi, p in enumerate(pr):
            if pi in used_pr: continue
            # compute max per-frame IoU over temporal overlap
            s = 0.0
            i=j=0
            while i < len(g.frames) and j < len(p.frames):
                fg, fp = g.frames[i], p.frames[j]
                if fg == fp:
                    s = max(s, iou(g.boxes[i], p.boxes[j])); i+=1; j+=1
                elif fg < fp:
                    i+=1
                else:
                    j+=1
            if s >= match_iou and s > best_s:
                best_s, best_p = s, pi
        if best_p >= 0:
            gt_to_pr[gi] = best_p
            used_pr.add(best_p)
    unmatched_pr = [pi for pi in range(len(pr)) if pi not in used_pr]
    return len(gt), len(pr), gt_to_pr, unmatched_pr

def reaction_time_frames(gt: Track, pr: Track) -> Optional[int]:
    # frames from first GT frame to first matched PR frame in overlap; None if no overlap
    i=j=0
    first=None
    while i < len(gt.frames) and j < len(pr.frames):
        fg, fp = gt.frames[i], pr.frames[j]
        if fg == fp:
            first = fp; break
        elif fg < fp:
            i+=1
        else:
            j+=1
    return None if first is None else max(0, first - gt.frames[0])

def auc_trapz(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2: return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += dx * (ys[i] + ys[i-1]) * 0.5
    return area
