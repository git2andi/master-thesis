# PICCOLO Dataset Preprocessing


import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

from helper.geometry import clip_box, yolo_norm_from_abs
from helper.coco import coco_init, add_coco_image, add_coco_annotation
from helper.yolo import write_yolo_label_file


# YOLO / COCO metadata
YOLO_ID = 0
YOLO_CLASS = "lesion"

COCO_INFO = {
    "description": "PICCOLO polyp dataset (detection-ready)",
    "url": "https://www.mdpi.com/2076-3417/10/23/8501",
    "version": "1.0",
    "year": 2020,
    "contributor": " Dr. Luisa F. Sánchez-Peralta et al.",
    "date_created": "2020/10/30",
}

COCO_LICENSES = [{
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
}]

COCO_ID = 1
COCO_CATEGORIES = [{"supercategory": "lesion", "id": COCO_ID, "name": "lesion"}]


def apply_void_crop(
    img: Image.Image,
    polyp_mask: Image.Image,
    void_mask: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the void mask to remove the white border and crop to the valid FOV.
      - void_mask: black region = valid image area, white region = border to discard.
    """
    void_gray = void_mask.convert("L")
    v = np.array(void_gray)

    # valid pixels: where mask is black (value 0)
    valid = (v == 0)

    if not np.any(valid):
        # Fallback: no valid region detected, return originals
        return img, polyp_mask

    ys, xs = np.where(valid)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # Crop box in PIL coordinates: (left, upper, right, lower)
    box = (x_min, y_min, x_max + 1, y_max + 1)

    img_cropped = img.crop(box)
    mask_cropped = polyp_mask.crop(box)

    # Also remove void outside FOV inside the cropped mask
    v_cropped = v[y_min:y_max + 1, x_min:x_max + 1]
    mask_arr = np.array(mask_cropped.convert("L"))
    mask_arr[v_cropped != 0] = 0  # zero out outside-FOV remnants
    mask_cropped = Image.fromarray(mask_arr).convert("L")


    return img_cropped, mask_cropped


def boxes_from_mask(mask_img: Image.Image, min_pixels=20):
    """
    Extract bounding boxes from mask.
    Filters out lesions < min_pixels (same as PICCOLO authors).
    """
    m = np.array(mask_img.convert("L"))
    ys, xs = np.where(m > 0)
    if ys.size < min_pixels:        # filtering threshold
        return []                   # → classify frame as negative

    # bounding box
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    return [(x_min, y_min, x_max, y_max)]




def draw_case_id_from_stem(stem: str) -> str:
    """
    PICCOLO filenames look like: 071_VP29_frame0359

    <video> > 071_VP29 > unique_id in COCO annotations
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return "_".join(parts[:2])  # 71_VP29


def convert_split(
    src_root: Path,
    dst_images_root: Path,
    dst_labels_root: Path,
    coco_json_path: Path,
    split_name: str,
) -> None:
    data = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    dst_images_root.mkdir(parents=True, exist_ok=True)
    dst_labels_root.mkdir(parents=True, exist_ok=True)

    polyps_dir = src_root / "polyps"
    masks_dir = src_root / "masks"
    void_dir = src_root / "void"

    image_id_counter = 0
    ann_id_counter = 0

    polyp_files = sorted(
        [p for p in polyps_dir.glob("*.png") if p.is_file()],
        key=lambda p: p.name,
    )

    print(f"\n--- Converting {split_name}: {len(polyp_files)} polyp images ---")

    for img_path in polyp_files:
        stem = img_path.stem  # e.g. "071_VP29_frame0359"

        mask_path = masks_dir / f"{stem}_Corrected.tif"
        void_path = void_dir / f"{stem}_Void.tif"

        if not mask_path.exists():
            print(f"  [WARN] Missing mask for {img_path.name}: {mask_path.name}")
            continue
        if not void_path.exists():
            print(f"  [WARN] Missing void mask for {img_path.name}: {void_path.name}")
            continue

        # Load images
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] Cannot open image {img_path}: {e}")
            continue

        try:
            polyp_mask = Image.open(mask_path)
        except Exception as e:
            print(f"  [ERROR] Cannot open mask {mask_path}: {e}")
            continue

        try:
            void_mask = Image.open(void_path)
        except Exception as e:
            print(f"  [ERROR] Cannot open void mask {void_path}: {e}")
            continue

        # Apply void cropping (remove white border, restrict to FOV)
        img_cropped, mask_cropped = apply_void_crop(img, polyp_mask, void_mask)
        width, height = img_cropped.size

        # Derive bounding boxes from mask
        boxes = boxes_from_mask(mask_cropped)

        # Save image as JPEG in destination
        new_img_name = f"{stem}.jpg"
        dst_img_path = dst_images_root / new_img_name
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        img_cropped.save(dst_img_path, format="JPEG", quality=100, subsampling=0)

        # COCO image entry
        add_coco_image(
            data,
            image_id=image_id_counter,
            file_name=new_img_name,
            width=width,
            height=height,
            license_id=1,
        )

        # COCO annotations + YOLO labels
        yolo_boxes = []
        unique_id = draw_case_id_from_stem(stem)

        for (x1_raw, y1_raw, x2_raw, y2_raw) in boxes:
            # Clip to image bounds
            x1, y1, x2, y2 = clip_box(x1_raw, y1_raw, x2_raw, y2_raw, width, height)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # COCO annotation
            add_coco_annotation(
                data,
                ann_id=ann_id_counter,
                image_id=image_id_counter,
                x=x1,
                y=y1,
                w=bw,
                h=bh,
                category_id=COCO_ID,
                extra_fields={"unique_id": unique_id},
            )

            # YOLO (normalized)
            cx, cy, ww, hh = yolo_norm_from_abs(x1, y1, x2, y2, w=width, h=height)
            yolo_boxes.append((YOLO_ID, cx, cy, ww, hh))
            ann_id_counter += 1

        # YOLO label file (empty if no boxes)
        yolo_label_path = dst_labels_root / f"{stem}.txt"
        yolo_label_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_label_file(str(yolo_label_path), yolo_boxes)

        image_id_counter += 1

    # Save COCO JSON
    coco_json_path.parent.mkdir(parents=True, exist_ok=True)
    with coco_json_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(
        f"{split_name}: {image_id_counter} images, "
        f"{ann_id_counter} annotations -> {coco_json_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert PICCOLO dataset to COCO/YOLO formats."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/data/local/aschwab/data/piccolo",
        help="Path to PICCOLO root (contains train/, validation/, test/).",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="/data/local/aschwab/data",
        help="Root output folder. Creates 'piccolo_split/' inside.",
    )

    args = parser.parse_args()
    base_dataset_folder = Path(args.src)
    dst_root = Path(args.dst_root)

    # Source splits
    train_root = base_dataset_folder / "train"
    val_root = base_dataset_folder / "validation"
    test_root = base_dataset_folder / "test"

    for p in [train_root, val_root, test_root]:
        if not p.is_dir():
            print(f"Error: expected split folder missing: {p}")
            return

    # Destination layout
    split_output_folder = dst_root / "piccolo_split"
    images_root = split_output_folder / "images"
    labels_root = split_output_folder / "labels"

    train_images_folder = images_root / "train"
    val_images_folder = images_root / "val"
    test_images_folder = images_root / "test"

    train_labels_folder = labels_root / "train"
    val_labels_folder = labels_root / "val"
    test_labels_folder = labels_root / "test"

    split_output_folder.mkdir(parents=True, exist_ok=True)

    # Convert each split
    convert_split(
        src_root=train_root,
        dst_images_root=train_images_folder,
        dst_labels_root=train_labels_folder,
        coco_json_path=split_output_folder / "coco_annotations_train.json",
        split_name="train",
    )

    convert_split(
        src_root=val_root,
        dst_images_root=val_images_folder,
        dst_labels_root=val_labels_folder,
        coco_json_path=split_output_folder / "coco_annotations_val.json",
        split_name="val",
    )

    convert_split(
        src_root=test_root,
        dst_images_root=test_images_folder,
        dst_labels_root=test_labels_folder,
        coco_json_path=split_output_folder / "coco_annotations_test.json",
        split_name="test",
    )

    # Write YOLO data.yaml
    data_yaml_path = split_output_folder / "data.yaml"
    with data_yaml_path.open("w") as f:
        f.write(
            f"""path: {split_output_folder}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["lesion"]
"""
        )

    print(f"\nWrote YOLO data.yaml to {data_yaml_path}")


if __name__ == "__main__":
    main()
