import os
import re
import csv
import json
import random
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
DST = "/data/local/aschwab/data/realColon_600x600"  # output root
IMG_SIZE = 600
PADDING_COLOR = (114, 114, 114)

# Keep a single class for detection (polyp vs background), but the source XML uses "lesion".
# We map to class 0 and keep YOLO names=["polyp"] for readability.
CLASS_MAP = {"lesion": 0}

# If present, these CSVs can be handy later (not required to run):
VIDEO_INFO = os.path.join(SRC, "video_info.csv")  # optional
LESION_INFO = os.path.join(SRC, "lesion_info.csv")  # optional

# =========================
# HELPERS
# =========================
def letterbox_pil(img: Image.Image, new_shape: int, color=(114,114,114)) -> Tuple[Image.Image, float, int, int]:
    """Resize with unchanged aspect ratio using padding (like YOLO). Returns (image, scale, pad_left, pad_top)."""
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

def voc_box_to_yolo_after_letterbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
    scale: float, pad_left: int, pad_top: int, img_size_out: int
) -> Tuple[float, float, float, float]:
    """Transform VOC (pixel) to padded/resized, then convert to normalized YOLO (cx, cy, w, h)."""
    x1 = xmin * scale + pad_left
    y1 = ymin * scale + pad_top
    x2 = xmax * scale + pad_left
    y2 = ymax * scale + pad_top

    # clip
    x1 = max(0.0, min(float(img_size_out), x1))
    y1 = max(0.0, min(float(img_size_out), y1))
    x2 = max(0.0, min(float(img_size_out), x2))
    y2 = max(0.0, min(float(img_size_out), y2))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / img_size_out, cy / img_size_out, bw / img_size_out, bh / img_size_out

def parse_vid_id(path_stem: str) -> tuple:
    """
    Parse 'SSS-VVV' from a vid id like '001-012'.
    Returns (cohort_int, video_int). Raises on bad format.
    """
    m = re.match(r"^(\d{3})-(\d{3})$", path_stem)
    if not m:
        raise ValueError(f"Unexpected video id format: {path_stem}")
    return int(m.group(1)), int(m.group(2))

def build_split(video_ids: List[str]) -> Dict[str, str]:
    """
    Build deterministic split per paper:
      For each cohort SSS ∈ {001..004}:
        Train: SSS-001..SSS-010
        Val:   SSS-011..SSS-012
        Test:  SSS-013..SSS-015
    """
    # group by cohort
    by_cohort: Dict[int, List[int]] = {}
    for vid in video_ids:
        sss, vvv = parse_vid_id(vid)
        by_cohort.setdefault(sss, []).append(vvv)
    # sanity & sort
    split_map = {}
    for sss, vvvs in by_cohort.items():
        vvvs = sorted(vvvs)
        # Expect exactly 15 videos per cohort
        if len(vvvs) != 15 or vvvs[0] != 1 or vvvs[-1] != 15:
            print(f"[WARN] Cohort {sss} has {len(vvvs)} videos (expected 15, 001..015). Using available sorted videos.")
        for v in vvvs:
            tag = None
            if 1 <= v <= 10:
                tag = "train"
            elif 11 <= v <= 12:
                tag = "val"
            else:
                tag = "test"  # 13..15 (or the remainder)
            split_map[f"{sss:03d}-{v:03d}"] = tag
    return split_map

def write_yaml_files(dst_root: str):
    # data.yaml for YOLO
    yaml_path = os.path.join(dst_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# REAL-Colon YOLO data
train: {os.path.join(dst_root, 'images/train')}
val:   {os.path.join(dst_root, 'images/val')}
test:  {os.path.join(dst_root, 'images/test')}

nc: 1
names: ["polyp"]  # source XML uses 'lesion'; we map it to class 0
""")
    print(f"✔ Wrote {yaml_path}")

    # Light COCO-style augmentation (training-time) – aligns with the paper spirit
    # Use with YOLOv5: --hyp hyp_endoscopy.yaml
    # Use with YOLOv8: pass equivalent CLI overrides (see commands below)
    hyp_path = os.path.join(dst_root, "hyp_endoscopy.yaml")
    with open(hyp_path, "w") as f:
        f.write("""# Light, realistic augs for endoscopy (no mosaic/mixup; no vertical flip)
hsv_h: 0.015
hsv_s: 0.60
hsv_v: 0.40
degrees: 5.0
translate: 0.10
scale: 0.20
shear: 1.0
perspective: 0.0
fliplr: 0.5
flipud: 0.0
mosaic: 0.0
mixup: 0.0
""")
    print(f"✔ Wrote {hyp_path}")

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

    # Discover videos
    frame_folders = sorted(glob(os.path.join(SRC, "frames", "*_frames")))
    anno_folders = sorted(glob(os.path.join(SRC, "annotations", "*_annotations")))
    video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])

    assert len(video_ids) == 60, f"Expected 60 videos, found {len(video_ids)}"
    print(f"Found {len(video_ids)} videos.")

    # Build deterministic split by cohort/video number (paper protocol)
    split_map = build_split(video_ids)

    # Track stats
    processed = 0
    skipped_xml = 0
    skipped_parse = 0

    # Iterate videos
    for vid in video_ids:
        split = split_map.get(vid, "test")  # default to test if unexpected numbering
        frame_dir = os.path.join(SRC, "frames", f"{vid}_frames")
        anno_dir  = os.path.join(SRC, "annotations", f"{vid}_annotations")

        img_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if not img_paths:
            print(f"[WARN] No frames for {vid}")
            continue

        for img_path in img_paths:
            base = Path(img_path).stem
            xml_path = os.path.join(anno_dir, f"{base}.xml")

            # open image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # letterbox to 600x600
            img_out, r, pad_l, pad_t = letterbox_pil(img, new_shape=IMG_SIZE, color=PADDING_COLOR)
            out_img = os.path.join(DST, "images", split, f"{base}.jpg")

            # Parse XML (if exists), collect boxes, write YOLO label (or empty if none)
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

                        cx, cy, bw, bh = voc_box_to_yolo_after_letterbox(
                            xmin, ymin, xmax, ymax, r, pad_l, pad_t, IMG_SIZE
                        )
                        if bw <= 0 or bh <= 0:
                            continue
                        cls_id = CLASS_MAP[cls_name]
                        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                except ET.ParseError:
                    skipped_parse += 1
                except Exception:
                    skipped_parse += 1
            else:
                skipped_xml += 1

            # save image + label (empty for negatives)
            img_out.save(out_img, format="JPEG", quality=95, subsampling=0)
            out_lbl = os.path.join(DST, "labels", split, f"{base}.txt")
            with open(out_lbl, "w") as f:
                f.write("\n".join(lines))

            processed += 1

        print(f"✅ {vid} → {split} ({len(img_paths)} frames)")

    # Write YOLO configs
    write_yaml_files(DST)

    print(f"\nDone. Converted {processed} frames.")
    print(f"Missing-XML frames: {skipped_xml}, XML-parse skips: {skipped_parse}")
    print(f"Dataset ready at: {DST}")
