#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_realcolon.py — Zero-arg auditor for REAL-Colon 640×640 and its variants.

What it does:
  • Auto-discovers datasets under /data/local/aschwab/data starting with "realColon_640x640"
  • Skips non-datasets (folders without images/{train,val,test})
  • Checks:
      - Folder schema
      - Image readability & size (expects 640×640)
      - Label integrity (class id, ranges, bounds)
      - COCO↔YOLO per-image agreement (IoU≥0.90) where COCO JSONs present
      - Split policy (vvv ranges)
      - Subset gating (_i1.._i4, _1_3)
      - Variant rules:
          * negR×: val/test IDENTICAL to base (filenames & label content);
                   train keeps ALL base positives; per-video negatives ≈ R× positives (±2 tol)
          * pos*:  val/test IDENTICAL to base (filenames & label content);
                   train contains all base train frames; duplicate labels equal originals
      - Event-level integrity (contiguous positive spans, ignoring *_posdup/_evt/_bin/_hnm)
          * For subsets: compare using VIDEO INTERSECTION (no false flags for excluded institutions)
          * For others:  compare using VIDEO UNION (strict equality)

  • Generates per-dataset visual galleries (BASE | VARIANT) for quick manual review
  • Writes ./audit_realcolon_YYYYmmdd_HHMMSS/ with report.json, report.md, event_diffs.txt, visual PNGs + HTML

Run:
  python audit_realcolon.py
"""

from __future__ import annotations
import json, re, sys, hashlib, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from PIL import Image, ImageDraw

# ---------- Defaults ----------
BASE_ROOT = Path("/data/local/aschwab/data")
BASE_NAME = "realColon_640x640"
IMG_EXT = ".jpg"; LBL_EXT = ".txt"
AUG_SUFFIXES = ("_posdup", "_evt", "_bin", "_hnm")
COCO_FILES = {
    "train":"annotations_coco_train.json",
    "val":"annotations_coco_val.json",
    "test":"annotations_coco_test.json"
}

# ---------- Regex helpers ----------
VID_RE   = re.compile(r'^(\d{3}-\d{3})')           # e.g., 001-001_...
FID_RE   = re.compile(r'_(\d+)(?:\D.*)?$')         # trailing digits at end
DUP_TAG  = re.compile(r'_(posdup|evtdup|bindup)\d+')  # detect duplicate stems

def vid_of(stem: str) -> Optional[str]:
    m = VID_RE.match(stem); return m.group(1) if m else None

def fid_of(stem: str) -> Optional[int]:
    m = FID_RE.search(stem); return int(m.group(1)) if m else None

def is_augmented(stem: str) -> bool:
    return any(tag in stem for tag in AUG_SUFFIXES)

# ---------- IO helpers ----------
def list_images(root: Path, split: str) -> List[Path]:
    return sorted((root/"images"/split).glob("*.jpg"))

def yolo_label(lbl: Path) -> str:
    try: return lbl.read_text(encoding="utf-8")
    except Exception: return ""

def load_yolo_boxes(lbl: Path) -> List[Tuple[float,float,float,float]]:
    out = []
    txt = yolo_label(lbl).strip()
    if not txt: return out
    for line in txt.splitlines():
        ps = line.split()
        if len(ps) < 5: continue
        try:
            cx, cy, w, h = map(float, ps[1:5])
        except Exception:
            continue
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        out.append((x1,y1,x2,y2))
    return out

def draw_overlay(img_path: Path, lbl_path: Path, title: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    boxes = load_yolo_boxes(lbl_path)
    for (x1n,y1n,x2n,y2n) in boxes:
        x1 = int(x1n*W); y1 = int(y1n*H); x2 = int(x2n*W); y2 = int(y2n*H)
        draw.rectangle([x1,y1,x2,y2], width=3, outline=(255,0,0))
    # title band
    band_h = 28
    canvas = Image.new("RGB", (W, H+band_h), (0,0,0))
    canvas.paste(img, (0, band_h))
    ImageDraw.Draw(canvas).text((8,6), f"{title} | boxes:{len(boxes)}", fill=(255,255,255))
    return canvas

# ---------- Greedy IoU ----------
def iou(a,b) -> float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.0, ix2-ix1); ih=max(0.0, iy2-iy1); inter=iw*ih
    area_a=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    area_b=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    den=area_a+area_b-inter
    return (inter/den) if den>0 else 0.0

def greedy_match(y: List[Tuple[float,float,float,float]],
                 c: List[Tuple[float,float,float,float]],
                 thr: float=0.90) -> bool:
    used_y=set(); used_c=set(); matched=0
    pairs=[]
    for i,a in enumerate(y):
        for j,b in enumerate(c):
            val=iou(a,b)
            if val>=thr:
                pairs.append((val,i,j))
    pairs.sort(reverse=True)
    for val,i,j in pairs:
        if i in used_y or j in used_c: continue
        used_y.add(i); used_c.add(j); matched+=1
    return (matched==len(y)==len(c))

# ---------- COCO loader ----------
def load_coco_boxes(coco_json: Path, imgsz: int=640) -> Dict[str, List[Tuple[float,float,float,float]]]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in data.get("images",[])}
    out: Dict[str, List[Tuple[float,float,float,float]]] = {}
    for a in data.get("annotations", []):
        st = id2stem.get(a["image_id"]); x,y,w,h = a["bbox"]
        if st is None: continue
        out.setdefault(st, []).append((x/imgsz, y/imgsz, (x+w)/imgsz, (y+h)/imgsz))
    return out

# ---------- Basic checks ----------
def schema_info(root: Path, issues: Dict[str,List[str]]) -> Dict[str, Any]:
    info = {"splits":{}}
    for sp in ("train","val","test"):
        idr = root/"images"/sp; ldr = root/"labels"/sp
        if not idr.exists(): issues["errors"].append(f"{root.name}: missing images/{sp}")
        if not ldr.exists(): issues["errors"].append(f"{root.name}: missing labels/{sp}")
        nimg = len(list(idr.glob("*.jpg"))) if idr.exists() else 0
        nlbl = len(list(ldr.glob("*.txt"))) if ldr.exists() else 0
        if nimg != nlbl:
            issues["warnings"].append(f"{root.name}:{sp}: images({nimg}) != labels({nlbl})")
        info["splits"][sp] = {"n_img": nimg, "n_lbl": nlbl}
    yaml = root/"data.yaml"
    if not yaml.exists():
        issues["warnings"].append(f"{root.name}: missing data.yaml")
    else:
        txt = yaml.read_text(encoding="utf-8")
        if "lesion" not in txt: issues["warnings"].append(f"{root.name}: data.yaml lacks 'lesion' class")
    return info

def check_images_640(root: Path, issues: Dict[str,List[str]]) -> Dict[str,int]:
    """Open ALL images once; assert 640x640 and readable."""
    bad_size=0; unreadable=0; total=0
    for sp in ("train","val","test"):
        for img in list_images(root, sp):
            total += 1
            try:
                with Image.open(img) as im:
                    w,h = im.size
                if (w,h)!=(640,640):
                    bad_size += 1
            except Exception:
                unreadable += 1
    if bad_size:
        issues["errors"].append(f"{root.name}: {bad_size} images not 640x640")
    if unreadable:
        issues["errors"].append(f"{root.name}: {unreadable} images unreadable by PIL")
    return {"total": total, "bad_size": bad_size, "unreadable": unreadable}

def check_labels_basic(root: Path, issues: Dict[str,List[str]]) -> Dict[str,Any]:
    bad = 0; empties = 0; tot = 0
    for sp in ("train","val","test"):
        for img in list_images(root, sp):
            tot += 1
            lbl = root/"labels"/sp/(img.stem + ".txt")
            if not lbl.exists():
                issues["errors"].append(f"{root.name}:{sp}:{img.name}: missing label txt")
                continue
            txt = yolo_label(lbl).strip()
            if not txt:
                empties += 1; continue
            for i, line in enumerate(txt.splitlines(), 1):
                ps = line.split()
                if len(ps) < 5:
                    bad += 1; continue
                try:
                    cid = int(float(ps[0])); cx,cy,w,h = map(float, ps[1:5])
                except Exception:
                    bad += 1; continue
                if cid != 0:
                    issues["errors"].append(f"{root.name}:{sp}:{lbl.name}:{i}: unexpected class {cid}")
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <=1.0 and 0.0 < h <=1.0):
                    issues["errors"].append(f"{root.name}:{sp}:{lbl.name}:{i}: coords out of [0,1] or non-positive")
                x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
                if not (x1 < x2 and y1 < y2 and x1 >= -1e-6 and y1 >= -1e-6 and x2 <= 1.0+1e-6 and y2 <= 1.0+1e-6):
                    issues["errors"].append(f"{root.name}:{sp}:{lbl.name}:{i}: box exceeds image bounds/degenerate")
    if bad:
        issues["errors"].append(f"{root.name}: {bad} malformed label lines")
    return {"total": tot, "empty": empties, "bad_lines": bad}

def coco_vs_yolo(root: Path, issues: Dict[str,List[str]], thr: float=0.90) -> Dict[str,Any]:
    stats={"checked":False,"splits":{}}
    coco_present = all((root/f).exists() for f in COCO_FILES.values())
    if not coco_present: return stats
    stats["checked"]=True
    for sp,fname in COCO_FILES.items():
        cjson = root/fname
        C = load_coco_boxes(cjson, imgsz=640); total=0; mism=0
        for img in list_images(root, sp):
            y = load_yolo_boxes(root/"labels"/sp/(img.stem+LBL_EXT))
            c = C.get(img.stem, [])
            total += 1
            if (len(y)!=len(c)) or (not greedy_match(y,c,thr)):
                mism += 1
        if mism:
            issues["errors"].append(f"{root.name}:{sp}: COCO↔YOLO mismatch {mism}/{total} (IoU≥{thr})")
        stats["splits"][sp]={"n_total":total,"n_mismatch":mism}
    return stats

def split_policy_ok(root: Path, issues: Dict[str,List[str]]) -> Dict[str,Any]:
    def policy(vvv:int)->str:
        if 1<=vvv<=10: return "train"
        if 11<=vvv<=12: return "val"
        return "test"
    by_split = {"train":set(),"val":set(),"test":set()}
    for sp in ("train","val","test"):
        for img in list_images(root, sp):
            vid=vid_of(img.stem); fid=fid_of(img.stem)
            if not vid or fid is None: 
                issues["warnings"].append(f"{root.name}:{sp}:{img.name}: cannot parse vid/fid")
                continue
            vvv = int(vid.split("-")[1])
            exp = policy(vvv)
            if exp != sp:
                issues["errors"].append(f"{root.name}: {vid} frame {img.name} in {sp} but policy says {exp}")
            by_split[sp].add(vid)
    return {"by_split": {k: len(v) for k,v in by_split.items()}}

def subset_gate(root: Path, issues: Dict[str,List[str]]) -> Dict[str,Any]:
    name=root.name; allow=None
    if name.endswith("_i1"): allow={1}
    elif name.endswith("_i2"): allow={2}
    elif name.endswith("_i3"): allow={3}
    elif name.endswith("_i4"): allow={4}
    elif name.endswith("_1_3"): allow={1,2,3}
    if allow is None: return {"enforced": False}
    bad=0
    for sp in ("train","val","test"):
        for img in list_images(root, sp):
            vid=vid_of(img.stem)
            if not vid: continue
            sss=int(vid.split("-")[0])
            if sss not in allow: bad+=1
    if bad: issues["errors"].append(f"{name}: {bad} frames outside allowed cohorts {sorted(allow)}")
    return {"enforced": True, "allowed": sorted(allow)}

def set_equal(a: Path, b: Path) -> bool:
    A={p.name for p in a.glob('*') if p.is_file()}
    B={p.name for p in b.glob('*') if p.is_file()}
    return A==B

def md5_text(p: Path) -> str:
    m=hashlib.md5()
    try:
        m.update(p.read_bytes())
    except Exception:
        return ""
    return m.hexdigest()

def label_content_equal(base: Path, var: Path, split: str, only_base_positives: bool=False) -> Tuple[int, List[str]]:
    """Return (mismatch_count, samples[]). If only_base_positives=True, compare only frames positive in BASE."""
    mism=0; samples=[]
    bdir = base/"labels"/split
    vdir = var/"labels"/split
    for b in sorted(bdir.glob("*.txt")):
        stem=b.stem
        if only_base_positives and (yolo_label(b).strip()== ""):
            continue
        v=vdir/(stem+LBL_EXT)
        if not v.exists():
            mism += 1; 
            if len(samples)<20: samples.append(f"{split}:{stem}.txt missing in variant")
            continue
        if md5_text(b) != md5_text(v):
            mism += 1; 
            if len(samples)<20: samples.append(f"{split}:{stem}.txt content differs")
    return mism, samples

# ---------- Variant rules ----------
def variant_neg_rules(root: Path, base: Path, issues: Dict[str,List[str]]) -> Dict[str,Any]:
    name=root.name
    m=re.search(r"_neg(\d+)x", name); ratio=int(m.group(1)) if m else None

    # val/test equality (filenames + label content)
    for sp in ("val","test"):
        if not (set_equal(root/"images"/sp, base/"images"/sp) and set_equal(root/"labels"/sp, base/"labels"/sp)):
            issues["errors"].append(f"{name}: {sp} file set differs from base {base.name}")
        mm, samples = label_content_equal(base, root, sp, only_base_positives=False)
        if mm:
            issues["errors"].append(f"{name}: {sp} label CONTENT differs from base in {mm} files (e.g., {', '.join(samples[:5])})")

    # train: all base positives kept; negatives ≈ ratio× positives/video (clipped)
    base_pos_by_vid={}; base_neg_by_vid={}
    for img in list_images(base, "train"):
        lbl=base/"labels"/"train"/(img.stem+LBL_EXT); vid=vid_of(img.stem) or "unknown"
        if yolo_label(lbl).strip(): base_pos_by_vid[vid]=base_pos_by_vid.get(vid,0)+1
        else: base_neg_by_vid[vid]=base_neg_by_vid.get(vid,0)+1

    var_pos_by_vid={}; var_neg_by_vid={}
    missing_pos=0
    base_train_imgs={img.name for img in list_images(base,"train")}
    var_train_imgs={img.name for img in list_images(root,"train")}
    for img in list_images(root,"train"):
        lbl=root/"labels"/"train"/(img.stem+LBL_EXT); vid=vid_of(img.stem) or "unknown"
        if yolo_label(lbl).strip(): var_pos_by_vid[vid]=var_pos_by_vid.get(vid,0)+1
        else: var_neg_by_vid[vid]=var_neg_by_vid.get(vid,0)+1
    for imgname in base_train_imgs:
        lbl=base/"labels"/"train"/(Path(imgname).stem+LBL_EXT)
        if yolo_label(lbl).strip() and imgname not in var_train_imgs:
            missing_pos+=1
    if missing_pos:
        issues["errors"].append(f"{name}: {missing_pos} positive frames from base/train are missing")

    ratio_deviation=0
    if ratio is not None:
        for vid,npos in base_pos_by_vid.items():
            nneg_total = base_neg_by_vid.get(vid,0)
            expected = min(int(ratio*npos), nneg_total)
            got = var_neg_by_vid.get(vid,0)
            if abs(got-expected) > 2:
                ratio_deviation += 1
        if ratio_deviation:
            issues["warnings"].append(f"{name}: {ratio_deviation} videos deviate from neg≈{ratio}×pos (±2)")

    # train label content equality for base-positive frames
    mm_train, samples_train = label_content_equal(base, root, "train", only_base_positives=True)
    if mm_train:
        issues["errors"].append(f"{name}: train label CONTENT differs from base for {mm_train} base-positive frames (e.g., {', '.join(samples_train[:5])})")

    return {"ratio": ratio, "ratio_deviation_videos": ratio_deviation, "missing_pos": missing_pos}

def variant_pos_rules(root: Path, base: Path, issues: Dict[str,List[str]]) -> Dict[str,Any]:
    name=root.name
    # val/test equality (filenames + label content)
    for sp in ("val","test"):
        if not (set_equal(root/"images"/sp, base/"images"/sp) and set_equal(root/"labels"/sp, base/"labels"/sp)):
            issues["errors"].append(f"{name}: {sp} file set differs from base {base.name}")
        mm, samples = label_content_equal(base, root, sp, only_base_positives=False)
        if mm:
            issues["errors"].append(f"{name}: {sp} label CONTENT differs from base in {mm} files (e.g., {', '.join(samples[:5])})")

    # all base train images present
    base_train_imgs={img.name for img in list_images(base,"train")}
    missing = [n for n in base_train_imgs if not (root/"images"/"train"/n).exists()]
    if missing:
        issues["errors"].append(f"{name}: {len(missing)} base train frames missing (e.g., {', '.join(missing[:5])})")

    # duplicate labels identical to originals
    dup_checked=0; dup_bad=0
    for img in list_images(root,"train"):
        st=img.stem
        if not DUP_TAG.search(st): continue
        dup_lbl=root/"labels"/"train"/(st+LBL_EXT)
        base_lbl=base/"labels"/"train"/(DUP_TAG.sub("",st)+LBL_EXT)
        if dup_lbl.exists() and base_lbl.exists():
            dup_checked += 1
            if yolo_label(dup_lbl) != yolo_label(base_lbl): dup_bad += 1
    if dup_bad:
        issues["errors"].append(f"{name}: {dup_bad}/{dup_checked} duplicate labels differ from base")

    # train label content equality for base-positive frames (originals)
    mm_train, samples_train = label_content_equal(base, root, "train", only_base_positives=True)
    if mm_train:
        issues["errors"].append(f"{name}: train label CONTENT differs from base for {mm_train} base-positive frames (e.g., {', '.join(samples_train[:5])})")

    return {"dup_checked": dup_checked, "dup_bad": dup_bad}

# ---------- EVENT checks ----------
def build_events(root: Path, split: str, ignore_augmented: bool=True) -> Dict[str, List[Tuple[int,int]]]:
    """
    Return per-video list of (start_fid, end_fid) for contiguous positive runs (gap=1).
    A frame is positive if its label file is non-empty.
    """
    by_vid_pos: Dict[str, List[int]] = {}
    for img in list_images(root, split):
        st=img.stem
        if ignore_augmented and is_augmented(st): continue
        vid=vid_of(st); fid=fid_of(st)
        if (vid is None) or (fid is None): continue
        lbl = root/"labels"/split/(st+LBL_EXT)
        if yolo_label(lbl).strip():
            by_vid_pos.setdefault(vid, []).append(fid)
    events: Dict[str,List[Tuple[int,int]]] = {}
    for vid, fids in by_vid_pos.items():
        if not fids: continue
        fids=sorted(set(fids))
        s=fids[0]; last=s
        spans=[]
        for f in fids[1:]:
            if f == last+1:
                last=f; continue
            spans.append((s,last)); s=f; last=f
        spans.append((s,last))
        events[vid]=spans
    return events

def compare_events(base: Dict[str,List[Tuple[int,int]]], 
                   var: Dict[str,List[Tuple[int,int]]], 
                   mode: str = "union") -> Tuple[int,List[str]]:
    """
    Compare base vs var event spans.
    mode="union": strict equality on union of videos (flags base-only/variant-only).
    mode="intersection": compare only videos present in BOTH (for subset datasets).
    Returns (mismatch_count, details_list).
    """
    if mode == "intersection":
        vids = set(base.keys()) & set(var.keys())
    else:
        vids = set(base.keys()) | set(var.keys())
    mism=0; details=[]
    for v in sorted(vids):
        b = base.get(v, [])
        r = var.get(v, [])
        if b != r:
            mism += 1
            details.append(f"{v}: base={b[:5]} ... vs var={r[:5]} ...")
    return mism, details

# ---------- Visual galleries ----------
def make_gallery(base: Path, var: Path, split: str, out_dir: Path, mode: str, n: int=24, seed:int=42):
    rng=random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    def is_pos(root, stem):
        return yolo_label(root/"labels"/split/(stem+LBL_EXT)).strip()!=""
    # find common stems
    bset={p.stem for p in list_images(base, split)}
    vset={p.stem for p in list_images(var, split)}
    items=[]
    if mode=="common_pos":
        common=[s for s in (bset & vset) if is_pos(base, s)]
        rng.shuffle(common); items=common[:n]
        pairs=[(s,s) for s in items]
    elif mode=="common_any":
        common=list(bset & vset); rng.shuffle(common); items=common[:n]
        pairs=[(s,s) for s in items]
    elif mode=="variant_only_neg":
        cands=[s for s in vset if (not is_pos(var, s)) and (s not in bset or is_pos(base, s))]
        rng.shuffle(cands); items=cands[:n]; pairs=[(s,s) for s in items]
    elif mode=="variant_duplicates":
        pairs=[]
        for s in sorted(vset):
            if DUP_TAG.search(s):
                base_s=DUP_TAG.sub("", s); pairs.append((base_s, s))
        rng.shuffle(pairs); pairs=pairs[:n]
    else:
        return
    # render
    created=0
    for i,(sb,sv) in enumerate(pairs, start=1):
        b_img=base/"images"/split/(sb+IMG_EXT)
        v_img=var/"images"/split/(sv+IMG_EXT)
        b_lbl=base/"labels"/split/(sb+LBL_EXT)
        v_lbl=var/"labels"/split/(sv+LBL_EXT)
        if not (b_img.exists() and v_img.exists()): continue
        left=draw_overlay(b_img, b_lbl, f"BASE:{sb}")
        right=draw_overlay(v_img, v_lbl, f"VAR:{sv}")
        W=left.width+right.width; H=max(left.height,right.height)
        canvas=Image.new("RGB",(W,H),(20,20,20)); canvas.paste(left,(0,0)); canvas.paste(right,(left.width,0))
        canvas.save(out_dir/f"{i:03d}__{sv}.png"); created+=1
    # html
    html=["<html><head><meta charset='utf-8'><title>Gallery</title>",
          "<style>body{background:#111;color:#ddd;font-family:Arial} .g{display:flex;flex-wrap:wrap;gap:10px} img{max-width:48%;border:1px solid #333}</style>",
          "</head><body><h1>Gallery</h1><div class='g'>"]
    for p in sorted(out_dir.glob("*.png")):
        html.append(f"<div><img src='{p.name}'/><div>{p.name}</div></div>")
    html.append("</div></body></html>")
    (out_dir/"index.html").write_text("\n".join(html), encoding="utf-8")
    return created

# ---------- Discovery ----------
def discover_datasets(base_root: Path, base_name: str) -> List[Path]:
    # take all dirs starting with base_name that have images/{train,val,test}
    candidates = [p for p in sorted(base_root.glob(base_name+"*")) if p.is_dir()]
    ds = []
    for p in candidates:
        if not (p/"images/train").exists(): 
            continue
        if not (p/"images/val").exists():
            continue
        if not (p/"images/test").exists():
            continue
        ds.append(p.resolve())
    # ensure base first
    base = (base_root/base_name).resolve()
    ds = [base] + [p for p in ds if p != base]
    return ds

# ---------- Main ----------
def main():
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root=Path(f"./audit_realcolon_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    base = (BASE_ROOT/BASE_NAME).resolve()
    if not base.exists():
        print(f"Base dataset not found: {base}", file=sys.stderr); sys.exit(2)

    # discover datasets
    datasets = discover_datasets(BASE_ROOT, BASE_NAME)

    summary={}; hard_fail=False

    # precompute base events for val/test/train
    base_ev = {sp: build_events(base, sp, ignore_augmented=True) for sp in ("val","test","train")}

    for root in datasets:
        name=root.name
        issues={"errors":[], "warnings":[]}
        res={"root": str(root)}

        res["schema"]=schema_info(root, issues)
        res["images"]=check_images_640(root, issues)
        res["labels"]=check_labels_basic(root, issues)
        res["coco_agreement"]=coco_vs_yolo(root, issues, thr=0.90)
        res["split_policy"]=split_policy_ok(root, issues)
        res["subset_filter"]=subset_gate(root, issues)

        is_subset = name.endswith(("_i1","_i2","_i3","_i4","_1_3")) or (res.get("subset_filter",{}).get("enforced", False) is True)

        # variant-specific
        if name != BASE_NAME:
            if "_neg" in name:
                res["variant_neg"]=variant_neg_rules(root, base, issues)
            if "_pos" in name:
                res["variant_pos"]=variant_pos_rules(root, base, issues)

            # EVENT integrity on val/test (subset-aware: intersection) and train
            ev = {sp: build_events(root, sp, ignore_augmented=True) for sp in ("val","test")}
            ev_out = {}
            ds_dir = out_root/name
            ds_dir.mkdir(parents=True, exist_ok=True)
            ev_diff_lines = []

            for sp in ("val","test"):
                mode = "intersection" if is_subset else "union"
                mism, details = compare_events(base_ev[sp], ev[sp], mode=mode)
                ev_out[sp]={"mismatch_videos": mism}
                if mism:
                    issues["errors"].append(f"{name}:{sp}: event spans differ from base in {mism} videos")
                    ev_diff_lines.append(f"[{sp}] mismatched videos: {mism}\n" + "\n".join(details[:200]))

            # train comparison
            ev_train = build_events(root, "train", ignore_augmented=True)
            mode_train = "intersection" if is_subset else "union"
            mism_train, train_details = compare_events(base_ev["train"], ev_train, mode=mode_train)
            if "_neg" in name:
                if mism_train:
                    issues["errors"].append(f"{name}:train: positive event spans differ from base in {mism_train} videos")
                    ev_diff_lines.append(f"[train] mismatched videos: {mism_train}\n" + "\n".join(train_details[:200]))
            else:
                if mism_train:
                    issues["errors"].append(f"{name}:train: base positive event spans differ (ignoring duplicates) in {mism_train} videos")
                    ev_diff_lines.append(f"[train] mismatched videos: {mism_train}\n" + "\n".join(train_details[:200]))

            if ev_diff_lines:
                with (ds_dir/"event_diffs.txt").open("w", encoding="utf-8") as f:
                    f.write(f"# Event differences for {name}\n\n")
                    for block in ev_diff_lines:
                        f.write(block + "\n\n")
            res["events"]=ev_out

            # Visual galleries (few curated modes)
            gdir = ds_dir/"visual"
            if any(tag in name for tag in ("_i1","_i2","_i3","_i4","_1_3")):
                make_gallery(base, root, "train", gdir/"common_pos_train", mode="common_pos", n=24, seed=42)
                make_gallery(base, root, "val",   gdir/"common_pos_val",   mode="common_pos", n=12, seed=42)
            if "_neg" in name:
                make_gallery(base, root, "train", gdir/"variant_only_neg_train", mode="variant_only_neg", n=24, seed=42)
            if "_pos" in name:
                make_gallery(base, root, "train", gdir/"variant_duplicates_train", mode="variant_duplicates", n=24, seed=42)

        res["errors"]=issues["errors"]; res["warnings"]=issues["warnings"]; res["ok"]= (len(issues["errors"])==0)
        if issues["errors"]: hard_fail=True
        summary[name]=res
        print(f"[{'OK' if res['ok'] else 'FAIL'}] {name} | warnings={len(issues['warnings'])}")

    # write summary files
    (out_root/"report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md=["# REAL-Colon Audit Summary\n"]
    for name,res in summary.items():
        md.append(f"## {name}\n- Root: `{res['root']}`\n- Status: **{'OK' if res['ok'] else 'FAIL'}**")
        for sp, s in res["schema"]["splits"].items():
            md.append(f"  - {sp}: {s['n_img']} images, {s['n_lbl']} labels")
        if res["warnings"]:
            md.append(f"- Warnings ({len(res['warnings'])}):\n  - " + "\n  - ".join(res["warnings"][:10]))
            if len(res["warnings"])>10: md.append(f"  - ... and {len(res['warnings'])-10} more")
        if res["errors"]:
            md.append(f"- Errors ({len(res['errors'])}):\n  - " + "\n  - ".join(res["errors"][:10]))
            if len(res["errors"])>10: md.append(f"  - ... and {len(res['errors'])-10} more")
        md.append("")
    (out_root/"report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Output folder: {out_root.resolve()}")
    if hard_fail: sys.exit(1)

if __name__ == "__main__":
    main()
