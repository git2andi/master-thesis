import os, random, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import cv2


RESIZED_ROOT = "/data/local/aschwab/data/realColon_640x640"
ORIGINAL_ROOT = "/data/local/aschwab/data/realColon"
SPLIT = "train"

IMG_DIR = Path(RESIZED_ROOT)/"images"/SPLIT
LBL_DIR = Path(RESIZED_ROOT)/"labels"/SPLIT
OUT_DIR = Path.home()/ "master-thesis" / "test" / "sanity_check"
N_SAMPLES = 10
OUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO labels with boxes
label_files = [f for f in LBL_DIR.glob("*.txt") if f.stat().st_size > 0]

by_video = {}
for lbl in label_files:
    vid = lbl.stem.split("_")[0]
    by_video.setdefault(vid, []).append(lbl)

chosen = []
for vid in random.sample(list(by_video.keys()), min(N_SAMPLES, len(by_video))):
    lbl = random.choice(by_video[vid])
    img_resized = IMG_DIR / f"{lbl.stem}.jpg"
    if img_resized.exists():
        chosen.append((vid, img_resized, lbl))

print(f"Selected {len(chosen)} frames from {len(set(v for v,_,_ in chosen))} videos")


# --- Helpers for plotting ---
def plot_yolo(img_path, yolo_path, out_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(yolo_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)  # red
    cv2.imwrite(str(out_path), img)


def plot_voc(img_path, xml_path, out_path):
    img = cv2.imread(str(img_path))
    if not os.path.exists(xml_path):
        return
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            if b is None: continue
            xmin = int(float(b.findtext("xmin")))
            ymin = int(float(b.findtext("ymin")))
            xmax = int(float(b.findtext("xmax")))
            ymax = int(float(b.findtext("ymax")))
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)  # green
    except Exception as e:
        print(f"[WARN] Failed to parse {xml_path}: {e}")
    cv2.imwrite(str(out_path), img)


# --- Copy + Plot ---
for vid, img_resized, lbl_resized in chosen:
    base = lbl_resized.stem

    orig_img = Path(ORIGINAL_ROOT)/"frames"/f"{vid}_frames"/f"{base}.jpg"
    orig_xml = Path(ORIGINAL_ROOT)/"annotations"/f"{vid}_annotations"/f"{base}.xml"

    shutil.copy(img_resized, OUT_DIR / f"{base}_resized.jpg")
    shutil.copy(lbl_resized, OUT_DIR / f"{base}_resized.txt")

    if orig_img.exists():
        shutil.copy(orig_img, OUT_DIR / f"{base}_orig.jpg")
    if orig_xml.exists():
        shutil.copy(orig_xml, OUT_DIR / f"{base}_orig.xml")

    plot_yolo(img_resized, lbl_resized, OUT_DIR / f"{base}_resized_plot.jpg")
    if orig_img.exists() and orig_xml.exists():
        plot_voc(orig_img, orig_xml, OUT_DIR / f"{base}_orig_plot.jpg")

    print(f"Copied {base}")