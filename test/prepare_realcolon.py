import os
import random
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from shutil import rmtree
from typing import Tuple, List

from PIL import Image, ImageOps  # pip install pillow

# === CONFIGURATION ===
SRC = "/data/local/aschwab/data/realColon"
DST = "/data/local/aschwab/data/realColon_640x640"
SPLIT = {"train": 48, "val": 6, "test": 6}
CLASS_MAP = {"lesion": 0}  # keep single class for SOTA comparability
IMG_SIZE = 640
PADDING_COLOR = (114, 114, 114)
random.seed(42)  # reproducible shuffle

# === CLEAN & CREATE OUTPUT STRUCTURE ===
if os.path.exists(DST):
    print(f"Removing existing folder: {DST}")
    rmtree(DST)

for split in SPLIT:
    os.makedirs(os.path.join(DST, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DST, 'labels', split), exist_ok=True)

# === FIND VIDEO FOLDERS ===
frame_folders = sorted(glob(os.path.join(SRC, "frames", "*_frames")))
anno_folders = sorted(glob(os.path.join(SRC, "annotations", "*_annotations")))
video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])
assert len(video_ids) == 60, f"Expected 60 videos, found {len(video_ids)}"
print(f"Found {len(video_ids)} videos.")

# === RANDOMIZED VIDEO SPLIT (by video ID) ===
random.shuffle(video_ids)  # avoid center-wise bias
train_ids = video_ids[:SPLIT["train"]]
val_ids = video_ids[SPLIT["train"]:SPLIT["train"] + SPLIT["val"]]
test_ids = video_ids[SPLIT["train"] + SPLIT["val"]:]

split_map = {vid: "train" for vid in train_ids}
split_map.update({vid: "val" for vid in val_ids})
split_map.update({vid: "test" for vid in test_ids})

print(f"Split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

def letterbox_pil(img: Image.Image, new_shape: int = 640, color=(114,114,114)) -> Tuple[Image.Image, float, int, int]:
    """Resize image with unchanged aspect ratio using padding (like YOLOv5). Returns (image, scale, pad_left, pad_top)."""
    w, h = img.size  # (width, height)
    r = min(new_shape / w, new_shape / h)
    new_unpad = (max(int(round(w * r)), 1), max(int(round(h * r)), 1))
    img = img.resize(new_unpad, Image.BILINEAR)
    pad_w = new_shape - new_unpad[0]
    pad_h = new_shape - new_unpad[1]
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=color)
    return img, r, pad_left, pad_top

def voc_box_to_yolo_after_letterbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
    scale: float, pad_left: int, pad_top: int, img_size_out: int = 640
) -> Tuple[float, float, float, float]:
    """Transform VOC (pixel) box using scale+pad, then convert to normalized YOLO (cx, cy, w, h)."""
    x1 = xmin * scale + pad_left
    y1 = ymin * scale + pad_top
    x2 = xmax * scale + pad_left
    y2 = ymax * scale + pad_top

    # clip to image bounds [0, img_size_out]
    x1 = max(0.0, min(float(img_size_out), x1))
    y1 = max(0.0, min(float(img_size_out), y1))
    x2 = max(0.0, min(float(img_size_out), x2))
    y2 = max(0.0, min(float(img_size_out), y2))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    # normalize
    return cx / img_size_out, cy / img_size_out, bw / img_size_out, bh / img_size_out

processed = 0
skipped_xml = 0
skipped_parse = 0

for vid in video_ids:
    frame_dir = os.path.join(SRC, "frames", f"{vid}_frames")
    anno_dir = os.path.join(SRC, "annotations", f"{vid}_annotations")
    split = split_map[vid]

    img_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
    if not img_paths:
        continue

    for img_path in img_paths:
        base = Path(img_path).stem
        xml_path = os.path.join(anno_dir, f"{base}.xml")
        if not os.path.exists(xml_path):
            skipped_xml += 1
            # still convert image and write empty label (negative frame)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            img640, r, pad_l, pad_t = letterbox_pil(img, new_shape=IMG_SIZE, color=PADDING_COLOR)
            out_img = os.path.join(DST, "images", split, f"{base}.jpg")
            img640.save(out_img, format="JPEG", quality=95, subsampling=0)

            out_lbl = os.path.join(DST, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("")  # empty label file
            processed += 1
            continue

        # parse XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            skipped_parse += 1
            # still convert image and write empty label
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            img640, r, pad_l, pad_t = letterbox_pil(img, new_shape=IMG_SIZE, color=PADDING_COLOR)
            out_img = os.path.join(DST, "images", split, f"{base}.jpg")
            img640.save(out_img, format="JPEG", quality=95, subsampling=0)
            out_lbl = os.path.join(DST, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("")
            processed += 1
            continue

        # open image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # letterbox
        img640, r, pad_l, pad_t = letterbox_pil(img, new_shape=IMG_SIZE, color=PADDING_COLOR)

        # collect boxes
        lines: List[str] = []
        for obj in root.findall("object"):
            cls_name = obj.findtext("name", default="").strip().lower()
            if cls_name not in CLASS_MAP:
                # skip unknown classes
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            try:
                xmin = float(b.findtext("xmin"))
                ymin = float(b.findtext("ymin"))
                xmax = float(b.findtext("xmax"))
                ymax = float(b.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            # transform & normalize
            cx, cy, bw, bh = voc_box_to_yolo_after_letterbox(
                xmin, ymin, xmax, ymax, r, pad_l, pad_t, IMG_SIZE
            )

            # drop degenerate boxes after transform
            if bw <= 0 or bh <= 0:
                continue

            cls_id = CLASS_MAP[cls_name]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # save image
        out_img = os.path.join(DST, "images", split, f"{base}.jpg")
        img640.save(out_img, format="JPEG", quality=95, subsampling=0)

        # save label (empty if no objects)
        out_lbl = os.path.join(DST, "labels", split, f"{base}.txt")
        with open(out_lbl, "w") as f:
            f.write("\n".join(lines))

        processed += 1

    print(f"✅ Processed video {vid} → {split} ({len(img_paths)} frames)")

print(f"\nDone. Converted {processed} frames. Missing-XML frames: {skipped_xml}, XML-parse skips: {skipped_parse}")

# === CREATE data.yaml FILE ===
yaml_path = os.path.join(DST, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""\
train: {os.path.join(DST, 'images/train')}
val: {os.path.join(DST, 'images/val')}
test: {os.path.join(DST, 'images/test')}

nc: 1
names: ["lesion"]
""")
print("\n📦 Dataset ready at:", DST)
