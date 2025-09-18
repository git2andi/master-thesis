#!/usr/bin/env python3
import os
import re
import json
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from shutil import rmtree
from typing import Tuple, List, Dict

from PIL import Image, ImageOps  # pip install pillow

# =========================
# CONFIG
# =========================
SRC = "/data/local/aschwab/data/realColon"                  # root with frames/ and annotations/
DST = "/data/local/aschwab/data/realColon_224x224"          # output root
IMG_SIZE = 224
PADDING_COLOR = (114, 114, 114)

CLASS_MAP = {"lesion": 0}  # VOC "lesion" → YOLO class 0 → COCO category 0


# =========================
# HELPERS
# =========================
def letterbox_pil(img: Image.Image, new_shape: int, color=(114, 114, 114)) -> Tuple[Image.Image, float, int, int]:
    """Resize with unchanged aspect ratio using padding (like YOLO)."""
    w, h = img.size
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


def voc_box_to_resized(
    xmin: float, ymin: float, xmax: float, ymax: float,
    scale: float, pad_left: int, pad_top: int, img_size_out: int
) -> Tuple[float, float, float, float]:
    """Transform VOC (pixel) to resized + padded pixel coords (x1,y1,x2,y2)."""
    x1 = xmin * scale + pad_left
    y1 = ymin * scale + pad_top
    x2 = xmax * scale + pad_left
    y2 = ymax * scale + pad_top

    # clip
    x1 = max(0.0, min(float(img_size_out), x1))
    y1 = max(0.0, min(float(img_size_out), y1))
    x2 = max(0.0, min(float(img_size_out), x2))
    y2 = max(0.0, min(float(img_size_out), y2))

    return x1, y1, x2, y2


def voc_box_to_yolo(x1: float, y1: float, x2: float, y2: float, img_size: int) -> Tuple[float, float, float, float]:
    """Convert pixel coords to normalized YOLO (cx,cy,w,h)."""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / img_size, cy / img_size, bw / img_size, bh / img_size


def parse_vid_id(path_stem: str) -> tuple:
    m = re.match(r"^(\d{3})-(\d{3})$", path_stem)
    if not m:
        raise ValueError(f"Unexpected video id format: {path_stem}")
    return int(m.group(1)), int(m.group(2))


def build_split(video_ids: List[str]) -> Dict[str, str]:
    by_cohort: Dict[int, List[int]] = {}
    for vid in video_ids:
        sss, vvv = parse_vid_id(vid)
        by_cohort.setdefault(sss, []).append(vvv)
    split_map = {}
    for sss, vvvs in by_cohort.items():
        vvvs = sorted(vvvs)
        for v in vvvs:
            if 1 <= v <= 10:
                tag = "train"
            elif 11 <= v <= 12:
                tag = "val"
            else:
                tag = "test"
            split_map[f"{sss:03d}-{v:03d}"] = tag
    return split_map


def write_yaml_files(dst_root: str):
    yaml_path = os.path.join(dst_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# REAL-Colon YOLO data
train: {os.path.join(dst_root, 'images/train')}
val:   {os.path.join(dst_root, 'images/val')}
test:  {os.path.join(dst_root, 'images/test')}

nc: 1
names: ["polyp"]
""")
    print(f"✔ Wrote {yaml_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Clean output
    if os.path.exists(DST):
        print(f"Removing existing folder: {DST}")
        rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", sp), exist_ok=True)

    # Prepare COCO dicts
    coco_by_split = {
        "train": {"images": [], "annotations": [], "categories": [{"id": 0, "name": "polyp"}]},
        "val":   {"images": [], "annotations": [], "categories": [{"id": 0, "name": "polyp"}]},
        "test":  {"images": [], "annotations": [], "categories": [{"id": 0, "name": "polyp"}]},
    }
    ann_id_counter = {"train": 0, "val": 0, "test": 0}

    # Discover videos
    frame_folders = sorted(glob(os.path.join(SRC, "frames", "*_frames")))
    anno_folders = sorted(glob(os.path.join(SRC, "annotations", "*_annotations")))
    video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])

    assert len(video_ids) == 60, f"Expected 60 videos, found {len(video_ids)}"
    print(f"Found {len(video_ids)} videos.")

    split_map = build_split(video_ids)

    processed = 0
    skipped_xml = 0
    skipped_parse = 0

    for vid in video_ids:
        split = split_map.get(vid, "test")
        frame_dir = os.path.join(SRC, "frames", f"{vid}_frames")
        anno_dir = os.path.join(SRC, "annotations", f"{vid}_annotations")

        img_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if not img_paths:
            print(f"[WARN] No frames for {vid}")
            continue

        for img_path in img_paths:
            base = Path(img_path).stem
            xml_path = os.path.join(anno_dir, f"{base}.xml")

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_out, r, pad_l, pad_t = letterbox_pil(img, new_shape=IMG_SIZE, color=PADDING_COLOR)
            out_img = os.path.join(DST, "images", split, f"{base}.jpg")

            # COCO bookkeeping
            img_id = len(coco_by_split[split]["images"]) + 1
            coco_by_split[split]["images"].append({
                "id": img_id,
                "file_name": f"{base}.jpg",
                "height": IMG_SIZE,
                "width": IMG_SIZE
            })

            lines: List[str] = []
            if os.path.exists(xml_path):
                try:
                    root = ET.parse(xml_path).getroot()
                    for obj in root.findall("object"):
                        cls_name = (obj.findtext("name") or "").strip().lower()
                        if cls_name not in CLASS_MAP:
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

                        # transform box
                        x1, y1, x2, y2 = voc_box_to_resized(xmin, ymin, xmax, ymax, r, pad_l, pad_t, IMG_SIZE)
                        cx, cy, bw, bh = voc_box_to_yolo(x1, y1, x2, y2, IMG_SIZE)
                        if bw <= 0 or bh <= 0:
                            continue

                        # YOLO line
                        cls_id = CLASS_MAP[cls_name]
                        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                        # COCO annotation
                        w = x2 - x1
                        h = y2 - y1
                        ann_id = ann_id_counter[split]
                        coco_by_split[split]["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id_counter[split] += 1

                except ET.ParseError:
                    skipped_parse += 1
                except Exception:
                    skipped_parse += 1
            else:
                skipped_xml += 1

            # save image + YOLO label
            img_out.save(out_img, format="JPEG", quality=95, subsampling=0)
            out_lbl = os.path.join(DST, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("\n".join(lines))

            processed += 1

        print(f"✅ {vid} → {split} ({len(img_paths)} frames)")

    # Write YOLO configs
    write_yaml_files(DST)

    # Dump COCO JSONs
    for sp in ("train", "val", "test"):
        out_json = os.path.join(DST, f"annotations_coco_{sp}.json")
        with open(out_json, "w") as f:
            json.dump(coco_by_split[sp], f, indent=2)
        print(f"✔ Wrote {out_json}")

    print(f"\nDone. Converted {processed} frames.")
    print(f"Missing-XML frames: {skipped_xml}, XML-parse skips: {skipped_parse}")
    print(f"Dataset ready at: {DST}")
