#!/usr/bin/env python3
import os, json, random, glob, xml.etree.ElementTree as ET
from pathlib import Path
import cv2

# ---- CONFIG (edit if paths differ) -----------------------------------------
ORIG_FRAMES = Path("/data/local/aschwab/data/realColon/frames")       # 001-001_frames/...
ORIG_ANNOTS = Path("/data/local/aschwab/data/realColon/annotations")  # 001-001_annotations/...
NEW_ROOT    = Path("/data/local/aschwab/data/realColon_full")
NEW_IMAGES  = NEW_ROOT / "images"     # images/{train,val,test}/...
NEW_LABELS  = NEW_ROOT / "labels"     # labels/{train,val,test}/...
COCO_JSONS  = {
    "train": NEW_ROOT / "annotations_coco_train.json",
    "val":   NEW_ROOT / "annotations_coco_val.json",
    "test":  NEW_ROOT / "annotations_coco_test.json",
}
OUT_DIR     = Path("./viz_check_min")
N_SAMPLES   = 3
# ----------------------------------------------------------------------------

def yolo_to_xyxy(cx, cy, w, h, W, H):
    cx, cy, w, h = cx*W, cy*H, w*W, h*H
    x1, y1 = int(round(cx - w/2)), int(round(cy - h/2))
    x2, y2 = int(round(cx + w/2)), int(round(cy + h/2))
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def draw(img, boxes, color, tag):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, tag, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img

def parse_voc(xml_path: Path):
    boxes = []
    root = ET.parse(str(xml_path)).getroot()
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1 = int(float(bb.findtext("xmin"))); y1 = int(float(bb.findtext("ymin")))
        x2 = int(float(bb.findtext("xmax"))); y2 = int(float(bb.findtext("ymax")))
        boxes.append((x1,y1,x2,y2))
    return boxes

def load_coco_index(json_path: Path):
    if not json_path.is_file(): return {}, {}
    data = json.loads(json_path.read_text())
    name2id = {os.path.basename(im["file_name"]): im["id"] for im in data.get("images", [])}
    id2boxes = {}
    for a in data.get("annotations", []):
        x,y,w,h = a["bbox"]
        id2boxes.setdefault(a["image_id"], []).append(
            (int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h)))
        )
    return name2id, id2boxes

def split_of(p: Path):
    for s in ("train","val","test"):
        if f"/{s}/" in f"/{p.as_posix()}/": return s
    return "train"

def orig_paths(basename: str):
    stem = Path(basename).stem           # e.g., 001-001_6928
    series = stem.split("_",1)[0]        # e.g., 001-001
    return (ORIG_FRAMES/f"{series}_frames"/basename,
            ORIG_ANNOTS/f"{series}_annotations"/f"{stem}.xml")

def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # grab non-empty YOLO label files across splits
    yolo_files = []
    for s in ("train","val","test"):
        yolo_files += [p for p in (NEW_LABELS/s).rglob("*.txt") if p.stat().st_size > 0]
    if not yolo_files:
        raise SystemExit("No non-empty YOLO labels found.")
    picks = random.sample(yolo_files, min(N_SAMPLES, len(yolo_files)))

    # preload COCO indices per split (simple dicts)
    coco_idx = {s: load_coco_index(COCO_JSONS[s]) for s in ("train","val","test")}

    for txt in picks:
        s = split_of(txt)
        rel = txt.relative_to(NEW_LABELS/s).with_suffix(".jpg")
        img_path = (NEW_IMAGES/s/rel)
        if not img_path.is_file():
            img_path = img_path.with_suffix(".png")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot read {img_path}"); continue
        H,W = img.shape[:2]
        base = img_path.name; stem = img_path.stem

        # 2) YOLO on new image
        yolo_boxes = []
        for line in txt.read_text().splitlines():
            if not line.strip(): continue
            parts = line.split()
            if len(parts) < 5: continue
            _, cx, cy, w, h = parts[:5]
            yolo_boxes.append(yolo_to_xyxy(float(cx),float(cy),float(w),float(h), W,H))
        yolo_img = draw(img.copy(), yolo_boxes, (0,255,0), "YOLO")
        cv2.imwrite(str(OUT_DIR/f"{stem}_new_YOLO.jpg"), yolo_img)

        # 3) COCO on new image
        name2id, id2boxes = coco_idx[s]
        coco_boxes = id2boxes.get(name2id.get(base, -1), [])
        coco_img = draw(img.copy(), coco_boxes, (0,0,255), "COCO")
        cv2.imwrite(str(OUT_DIR/f"{stem}_new_COCO.jpg"), coco_img)

        # 1) VOC on original image
        orig_img_path, orig_xml_path = orig_paths(base)
        oimg = cv2.imread(str(orig_img_path))
        if oimg is not None and orig_xml_path.is_file():
            voc_boxes = parse_voc(orig_xml_path)
            voc_img = draw(oimg, voc_boxes, (0,165,255), "VOC")
            cv2.imwrite(str(OUT_DIR/f"{stem}_orig_VOC.jpg"), voc_img)
        else:
            print(f"[warn] missing original for {base}: {orig_img_path} / {orig_xml_path}")

        print(f"[ok] wrote triplet for {base} -> {OUT_DIR}")

if __name__ == "__main__":
    main()
