#!/usr/bin/env python

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Adjust if needed
BASE_ORIG = Path("/data/local/aschwab/data/piccolo")
BASE_SPLIT = Path("/data/local/aschwab/data/piccolo_split")

NUM_SAMPLES = 8
RANDOM_SEED = 42


def apply_void_crop(img: Image.Image,
                    polyp_mask: Image.Image,
                    void_mask: Image.Image):
    """
    Apply the same void cropping as in prepare_piccolo.py:
    - find bounding box of valid FOV (void mask == 0),
    - crop image and mask to this rectangle,
    - black-out outside-FOV pixels in both image and mask inside that rect.
    """
    void_gray = void_mask.convert("L")
    v = np.array(void_gray)
    valid = (v == 0)

    if not np.any(valid):
        # fallback: no valid region detected, return originals
        return img, polyp_mask

    ys, xs = np.where(valid)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # Rectangular crop
    box = (x_min, y_min, x_max + 1, y_max + 1)

    img_cropped = img.crop(box)
    mask_cropped = polyp_mask.crop(box)
    v_cropped = v[y_min:y_max + 1, x_min:x_max + 1]

    # Black-out outside-FOV in image
    img_arr = np.array(img_cropped)
    img_arr[v_cropped != 0] = 0
    img_cropped = Image.fromarray(img_arr)

    # Black-out outside-FOV in mask
    mask_arr = np.array(mask_cropped.convert("L"))
    mask_arr[v_cropped != 0] = 0
    mask_cropped = Image.fromarray(mask_arr).convert("L")

    return img_cropped, mask_cropped


def yolo_to_abs_boxes(label_path: Path, img_width: int, img_height: int):
    """
    Convert YOLO labels (cls cx cy w h, normalized) to absolute boxes.
    """
    if not label_path.exists():
        return []

    txt = label_path.read_text().strip()
    if not txt:
        return []

    boxes = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        # cls = int(parts[0])  # unused here
        _, cx, cy, w, h = parts[:5]
        cx = float(cx) * img_width
        cy = float(cy) * img_height
        w = float(w) * img_width
        h = float(h) * img_height

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes.append((x1, y1, x2, y2))

    return boxes


def draw_boxes_on_image(img: Image.Image, boxes, color="red", width=3):
    """
    Draw list of boxes [(x1,y1,x2,y2), ...] on a copy of img.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def overlay_mask(base_img: Image.Image, mask_img: Image.Image):
    """
    Overlay a binary mask onto base_img by marking mask>0 pixels in red.
    """
    base = base_img.copy()
    arr = np.array(base)
    m = np.array(mask_img.convert("L"))

    # red overlay where mask is positive
    arr[m > 0] = [255, 0, 0]
    return Image.fromarray(arr)


def collect_polyp_candidates():
    """
    Collect all images in piccolo_split that have >= 1 annotation in COCO
    (train, val, test).
    Returns list of tuples: (split_name, img_info, img_anns).
    """
    candidates = []
    split_defs = [
        ("train", "coco_annotations_train.json"),
        ("val",   "coco_annotations_val.json"),
        ("test",  "coco_annotations_test.json"),
    ]

    for split_name, coco_file in split_defs:
        coco_path = BASE_SPLIT / coco_file
        if not coco_path.exists():
            continue

        with coco_path.open("r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        anns = coco.get("annotations", [])

        ann_by_img = {}
        for ann in anns:
            img_id = ann["image_id"]
            ann_by_img.setdefault(img_id, []).append(ann)

        for img_info in images:
            img_id = img_info["id"]
            img_anns = ann_by_img.get(img_id, [])
            if len(img_anns) > 0:
                candidates.append((split_name, img_info, img_anns))

    return candidates


def main():
    candidates = collect_polyp_candidates()
    print(f"Found {len(candidates)} positive frames in piccolo_split.")

    if len(candidates) == 0:
        raise RuntimeError("No positive frames found in piccolo_split.")

    random.seed(RANDOM_SEED)
    samples = random.sample(candidates, min(NUM_SAMPLES, len(candidates)))

    out_dir = Path.cwd() / "verify_piccolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving verification images to: {out_dir}")

    # mapping split_name -> original split folder name
    split_to_orig = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    for idx, (split_name, img_info, img_anns) in enumerate(samples, start=1):
        fname = img_info["file_name"]  # e.g. "071_VP29_frame0359.jpg"
        stem = Path(fname).stem        # "071_VP29_frame0359"

        orig_split = split_to_orig[split_name]

        # Original paths
        orig_root = BASE_ORIG / orig_split
        orig_img_path = orig_root / "polyps" / f"{stem}.png"
        orig_mask_path = orig_root / "masks" / f"{stem}_Corrected.tif"
        orig_void_path = orig_root / "void" / f"{stem}_Void.tif"

        if not orig_img_path.exists():
            print(f"[MISS] Original image missing: {orig_img_path}")
            continue
        if not orig_mask_path.exists():
            print(f"[MISS] Original mask missing: {orig_mask_path}")
            continue
        if not orig_void_path.exists():
            print(f"[MISS] Original void mask missing: {orig_void_path}")
            continue

        # Split paths
        split_img_path = BASE_SPLIT / "images" / split_name / fname
        split_lbl_path = BASE_SPLIT / "labels" / split_name / f"{stem}.txt"

        if not split_img_path.exists():
            print(f"[MISS] Split image missing: {split_img_path}")
            continue

        # Load originals
        orig_img = Image.open(orig_img_path).convert("RGB")
        orig_mask = Image.open(orig_mask_path).convert("L")
        orig_void = Image.open(orig_void_path).convert("L")

        # Apply the same void cropping as preprocessing to original image + mask
        orig_img_c, orig_mask_c = apply_void_crop(orig_img, orig_mask, orig_void)

        # Load split image (already cropped)
        split_img_coco = Image.open(split_img_path).convert("RGB")
        split_img_yolo = split_img_coco.copy()
        w, h = split_img_coco.size

        # COCO boxes for this image
        split_coco_boxes = []
        for ann in img_anns:
            x, y, bw, bh = ann["bbox"]
            x1 = x
            y1 = y
            x2 = x + bw
            y2 = y + bh
            split_coco_boxes.append((x1, y1, x2, y2))

        # YOLO boxes for this image
        split_yolo_boxes = yolo_to_abs_boxes(split_lbl_path, w, h)

        # Visualizations:
        # 1) original cropped image with original mask overlay
        orig_vis = overlay_mask(orig_img_c, orig_mask_c)

        # 2) split image with COCO boxes
        split_coco_vis = draw_boxes_on_image(split_img_coco, split_coco_boxes,
                                             color="lime", width=3)

        # 3) split image with YOLO boxes
        split_yolo_vis = draw_boxes_on_image(split_img_yolo, split_yolo_boxes,
                                             color="blue", width=3)

        base_tag = f"piccolo_sample{idx}_{stem}"

        out_orig = out_dir / f"{base_tag}_orig_mask.jpg"
        out_coco = out_dir / f"{base_tag}_split_coco.jpg"
        out_yolo = out_dir / f"{base_tag}_split_yolo.jpg"

        orig_vis.save(out_orig)
        split_coco_vis.save(out_coco)
        split_yolo_vis.save(out_yolo)

        print(f"Saved for {fname}:")
        print(f"  {out_orig.name}")
        print(f"  {out_coco.name}")
        print(f"  {out_yolo.name}")

    print("Done.")


if __name__ == "__main__":
    main()
