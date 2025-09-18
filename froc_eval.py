import pathlib, re, numpy as np
from collections import defaultdict

IMG_SIZE = 640
BASE     = pathlib.Path("realColon_thesis")
VAL_LABS = pathlib.Path("/data/local/aschwab/data/realColon_640x640/labels/val")

def vid_from_stem(stem: str) -> str:
    m = re.match(r"([A-Za-z0-9]+-[A-Za-z0-9]+)_", stem)  # e.g., 002-011_XXXXX -> 002-011
    return m.group(1) if m else stem.split("_")[0]

def yolo_to_xyxy(xc, yc, w, h, s=IMG_SIZE):
    x1=(xc-w/2)*s; y1=(yc-h/2)*s; x2=(xc+w/2)*s; y2=(yc+h/2)*s
    return np.array([x1,y1,x2,y2], dtype=float)

def iou(a,b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    iw=max(0.0,xx2-xx1); ih=max(0.0,yy2-yy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua>0 else 0.0

# --- load GT once ---
gt_by_img = {}
size_bin  = {}
for lab in sorted(VAL_LABS.glob("*.txt")):
    stem = lab.stem
    lines = lab.read_text().strip().splitlines()
    G = []
    B = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5: continue
        _,xc,yc,w,h = map(float, parts[:5])
        G.append(yolo_to_xyxy(xc,yc,w,h))
        area = (w*IMG_SIZE)*(h*IMG_SIZE)
        B.append('S' if area < 32*32 else 'M' if area < 96*96 else 'L')
    gt_by_img[stem] = np.array(G) if G else np.zeros((0,4))
    size_bin[stem]  = B

def find_pred_dir(run: pathlib.Path):
    d1 = run / "labels"
    d2 = run / "pred" / "labels"
    if d1.exists(): return d1
    if d2.exists(): return d2
    return None

def parse_thr(name: str):
    m = re.search(r"iou([0-9.]+)_c([0-9.]+)$", name)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

runs = sorted(p for p in BASE.glob("val_sweep_iou*/"))
print("run,IoU,conf,Sens,Sens_S,Sens_M,FP_per_video")

for r in runs:
    pred_dir = find_pred_dir(r)
    if not pred_dir: 
        continue
    iou_thr, conf_thr = parse_thr(r.name)
    if iou_thr is None: 
        continue

    tp=fp=tot=tpS=tpM=totS=totM=0
    vids=set(); fp_by_vid=defaultdict(int)

    for pf in sorted(pred_dir.glob("*.txt")):
        stem = pf.stem
        G  = gt_by_img.get(stem, np.zeros((0,4)))
        Bn = size_bin.get(stem, [])
        tot += len(G)
        for b in Bn:
            if b=='S': totS += 1
            elif b=='M': totM += 1

        preds=[]
        txt = pf.read_text().strip()
        if txt:
            for ln in txt.splitlines():
                parts = ln.split()
                if len(parts) < 6: continue
                # cls cx cy w h conf
                _,xc,yc,w,h,conf = parts[:6]
                if float(conf) >= conf_thr:
                    preds.append((yolo_to_xyxy(float(xc),float(yc),float(w),float(h)), float(conf)))
        preds.sort(key=lambda z:-z[1])

        matched = np.zeros(len(G), dtype=bool)
        for box,_ in preds:
            best_iou, best_j = 0.0, -1
            for j,g in enumerate(G):
                if matched[j]: continue
                v = iou(box,g)
                if v > best_iou: best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                matched[best_j] = True
                tp += 1
                sz = Bn[best_j] if best_j < len(Bn) else 'L'
                if   sz=='S': tpS += 1
                elif sz=='M': tpM += 1
            else:
                fp += 1
                vids.add(vid_from_stem(stem))
        vids.add(vid_from_stem(stem))

    videos = max(len(vids), 1)
    sens   = tp / tot if tot else 0.0
    sensS  = tpS / max(totS,1)
    sensM  = tpM / max(totM,1)
    print(f"{r.name},{iou_thr:.2f},{conf_thr:.2f},{sens:.3f},{sensS:.3f},{sensM:.3f},{fp/videos:.2f}")
