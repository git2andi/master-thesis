# SUN Dataset Preprocessing

import os
import json
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict
import random
from typing import List, Tuple, Dict, Any
import sys

from natsort import natsorted
from PIL import Image

from helper.geometry import clip_box, yolo_norm_from_abs
from helper.coco import add_coco_image, add_coco_annotation
from helper.yolo import write_yolo_label_file

# YOLO / COCO metadata
YOLO_ID = 0
YOLO_CLASS = "lesion"

COCO_INFO = {
    "description": "SUN dataset (Skin Ulceration Network)",
    "url": "http://sun-dataset.org",
    "version": "1.0",
    "year": 2024,
    "contributor": "SUN Developers",
    "date_created": "2024/01/01",
}

COCO_LICENSES = [{
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
}]
COCO_ID = 1
COCO_CATEGORIES = [{"supercategory": "lesion", "id": COCO_ID, "name": "lesion"}]


def parse_sun_case_annotations(annotation_path: str, video_case_name: str) -> Dict[str, list]:
    frame_annotations = defaultdict(list)
    if not os.path.exists(annotation_path):
        return {}
    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            box_strings = parts[1:]
            for box_str in box_strings:
                try:
                    coords = [int(float(c)) for c in box_str.split(",")[:4]]
                    x1, y1, x2, y2 = coords
                except ValueError:
                    continue
                if len(coords) != 4:
                    continue
                box_info = {"box_ltrb": [x1, y1, x2, y2], "unique_id": video_case_name}
                frame_annotations[filename].append(box_info)
    return dict(frame_annotations)


def convert_case_list(
    base_dataset_folder: str,
    case_list: List[Dict[str, Any]],
    frames_output_folder: str,
    labels_output_folder: str,
    json_output_file: str,
    imgsz: int,
) -> None:
    data = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_uniq_id_cnt = 0
    image_uniq_box_cnt = 0

    annotation_root = os.path.join(
        base_dataset_folder, "sundatabase_positive_part1", "annotation_txt"
    )

    for case_idx, case_info in enumerate(case_list):
        case_name = case_info["case_name"]
        data_folder = case_info["data_folder"]
        case_type = case_info["case_type"]
        new_case_id = case_info["new_case_id"]
        case_prefix = f"case{new_case_id}"

        print(f"Processing case {case_idx+1}/{len(case_list)} ({case_type}): {case_name} -> {case_prefix}")

        frames_dir = os.path.join(base_dataset_folder, data_folder, case_name)
        ann_file_path = os.path.join(annotation_root, f"{case_name}.txt")
        annotations = parse_sun_case_annotations(ann_file_path, case_name)
        frames_wbox_names = set(annotations.keys())

        all_images = sorted(
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(frames_dir, f))
        )

        for img_name in all_images:
            src_img_path = os.path.join(frames_dir, img_name)
            if not os.path.exists(src_img_path):
                continue

            # rename: case<id>_<original-filename>
            new_img_name = f"{case_prefix}_{img_name}"
            dst_img_path = os.path.join(frames_output_folder, new_img_name)

            try:
                img = Image.open(src_img_path).convert("RGB")
            except Exception:
                continue

            out_W, out_H = img.size

            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            img.save(dst_img_path, format="JPEG", quality=100, subsampling=0)

            add_coco_image(
                data,
                image_id=image_uniq_id_cnt,
                file_name=new_img_name,
                width=out_W,
                height=out_H,
                license_id=1,
            )

            yolo_boxes = []
            if img_name in frames_wbox_names:
                for cbox in annotations[img_name]:
                    l_raw, t_raw, r_raw, b_raw = cbox["box_ltrb"]

                    x1 = l_raw
                    y1 = t_raw
                    x2 = r_raw
                    y2 = b_raw

                    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, out_W, out_H)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue

                    add_coco_annotation(
                        data,
                        ann_id=image_uniq_box_cnt,
                        image_id=image_uniq_id_cnt,
                        x=x1,
                        y=y1,
                        w=bw,
                        h=bh,
                        category_id=COCO_ID,
                        extra_fields={"unique_id": cbox["unique_id"]},
                    )

                    cx, cy, ww, hh = yolo_norm_from_abs(
                        x1, y1, x2, y2, w=out_W, h=out_H
                    )
                    yolo_boxes.append((YOLO_ID, cx, cy, ww, hh))
                    image_uniq_box_cnt += 1

            yolo_out = os.path.join(
                labels_output_folder, Path(new_img_name).with_suffix(".txt").name
            )
            os.makedirs(os.path.dirname(yolo_out), exist_ok=True)
            write_yolo_label_file(yolo_out, yolo_boxes)

            image_uniq_id_cnt += 1

    with open(json_output_file, "w") as f:
        json.dump(data, f, indent=2)


def collect_all_cases(base_dataset_folder: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    negative_cases: List[Dict[str, Any]] = []
    positive_cases: List[Dict[str, Any]] = []

    data_parts = glob(os.path.join(base_dataset_folder, "sundatabase_*_part*"))

    for part_path in data_parts:
        part_name = os.path.basename(part_path)

        if "negative" in part_name:
            case_type = "negative"
        elif "positive" in part_name:
            case_type = "positive"
        else:
            continue

        case_subfolders = [
            d for d in os.listdir(part_path)
            if os.path.isdir(os.path.join(part_path, d)) and d != "annotation_txt"
        ]
        case_names = natsorted(case_subfolders)

        for case_name in case_names:
            case_info: Dict[str, Any] = {
                "case_name": case_name,
                "data_folder": part_name,
                "case_type": case_type,
            }
            if case_type == "negative":
                negative_cases.append(case_info)
            else:
                positive_cases.append(case_info)

    positive_cases = natsorted(positive_cases, key=lambda x: x["case_name"])
    negative_cases = natsorted(negative_cases, key=lambda x: x["case_name"])

    # assign numeric IDs: pos 1.., neg 101..
    for i, c in enumerate(positive_cases):
        c["new_case_id"] = i + 1
    NEG_START = 101
    for j, c in enumerate(negative_cases):
        c["new_case_id"] = NEG_START + j

    return positive_cases, negative_cases


def execute_split_task(
    pos_cases: List[Dict[str, Any]],
    neg_cases: List[Dict[str, Any]],
    dst_root: str,
    imgsz: int,
    is_full_test: bool,
    base_dataset_folder: str,
) -> None:
    random.seed(1000)

    if is_full_test:
        train_cases: List[Dict[str, Any]] = []
        val_cases: List[Dict[str, Any]] = []
        test_cases = pos_cases.copy()
        for neg_case in neg_cases:
            test_cases.insert(random.randint(0, len(test_cases)), neg_case)
    else:
        num_pos = len(pos_cases)
        pos_train_len = int(num_pos * 0.70)
        pos_val_len = int(num_pos * 0.10)

        pos_train = pos_cases[:pos_train_len]
        pos_val = pos_cases[pos_train_len:pos_train_len + pos_val_len]
        pos_test = pos_cases[pos_train_len + pos_val_len:]

        random.shuffle(neg_cases)
        neg_train_len, neg_val_len, neg_test_len = 7, 2, 4

        neg_train = neg_cases[:min(neg_train_len, len(neg_cases))]
        neg_val = neg_cases[neg_train_len:min(neg_train_len + neg_val_len, len(neg_cases))]
        neg_test = neg_cases[neg_train_len + neg_val_len:min(neg_train_len + neg_val_len + neg_test_len, len(neg_cases))]

        train_cases = pos_train.copy()
        for nc in neg_train:
            train_cases.insert(random.randint(0, len(train_cases)), nc)

        val_cases = pos_val.copy()
        for nc in neg_val:
            val_cases.insert(random.randint(0, len(val_cases)), nc)

        test_cases = pos_test.copy()
        for nc in neg_test:
            test_cases.insert(random.randint(0, len(test_cases)), nc)

        print(f"\nDataset Split Summary for {os.path.basename(dst_root)}:")
        print(f"Train Cases: {len(pos_train)} Pos, {len(neg_train)} Neg ({len(train_cases)} Total)")
        print(f"Val Cases:   {len(pos_val)} Pos, {len(neg_val)} Neg ({len(val_cases)} Total)")
        print(f"Test Cases:  {len(pos_test)} Pos, {len(neg_test)} Neg ({len(test_cases)} Total)")

    images_root = os.path.join(dst_root, "images")
    labels_root = os.path.join(dst_root, "labels")

    train_images_folder = os.path.join(images_root, "train")
    val_images_folder = os.path.join(images_root, "val")
    test_images_folder = os.path.join(images_root, "test")

    train_labels_folder = os.path.join(labels_root, "train")
    val_labels_folder = os.path.join(labels_root, "val")
    test_labels_folder = os.path.join(labels_root, "test")

    json_output_file_train = os.path.join(dst_root, "coco_annotations_train.json")
    json_output_file_val = os.path.join(dst_root, "coco_annotations_val.json")
    json_output_file_test = os.path.join(dst_root, "coco_annotations_test.json")

    for folder in [
        train_images_folder, val_images_folder, test_images_folder,
        train_labels_folder, val_labels_folder, test_labels_folder,
    ]:
        os.makedirs(folder, exist_ok=True)

    if not is_full_test:
        if train_cases:
            print("\n--- Converting Training subset ---")
            convert_case_list(
                base_dataset_folder, train_cases,
                train_images_folder, train_labels_folder,
                json_output_file_train, imgsz=imgsz,
            )
        if val_cases:
            print("\n--- Converting Validation subset ---")
            convert_case_list(
                base_dataset_folder, val_cases,
                val_images_folder, val_labels_folder,
                json_output_file_val, imgsz=imgsz,
            )
    if test_cases:
        print("\n--- Converting Testing subset ---")
        convert_case_list(
            base_dataset_folder, test_cases,
            test_images_folder, test_labels_folder,
            json_output_file_test, imgsz=imgsz,
        )

    data_yaml_path = os.path.join(dst_root, "data.yaml")
    train_line = "train: images/train" if not is_full_test else ""
    val_line = "val: images/val" if not is_full_test else ""

    with open(data_yaml_path, "w") as f:
        f.write(
            f"""path: {dst_root}
{train_line}
{val_line}
test: images/test

nc: 1
names: ["lesion"]
"""
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SUN dataset cases to COCO/YOLO formats with split and full test options."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/data/local/aschwab/data/sun",
        help="Path to the folder of the original SUN dataset (unzipped root).",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="/data/local/aschwab/data",
        help="Root output folder. Creates 'sun_split/' and 'sun_full/' inside.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=0,  # ignored, kept for compatibility
        help="(Ignored) Was used for resizing; kept for CLI compatibility.",
    )

    args = parser.parse_args()
    base_dataset_folder = args.src
    dst_root = args.dst_root
    imgsz = args.imgsz

    if not os.path.isdir(base_dataset_folder):
        print(f"Error: Base dataset folder {base_dataset_folder} not found.")
        sys.exit(1)

    positive_cases, negative_cases = collect_all_cases(base_dataset_folder)
    print(f"Found {len(positive_cases)} Positive Cases.")
    print(f"Found {len(negative_cases)} Negative Cases.")

    # sun_split
    split_output_folder = os.path.join(dst_root, "sun_split")
    print("\n" + "=" * 50)
    print(f"| TASK 1: Creating SPLIT dataset in {split_output_folder}")
    print("=" * 50)
    try:
        os.makedirs(split_output_folder, exist_ok=False)
        execute_split_task(
            positive_cases,
            negative_cases,
            split_output_folder,
            imgsz,
            is_full_test=False,
            base_dataset_folder=base_dataset_folder,
        )
    except FileExistsError:
        print(f"Skipping TASK 1: {split_output_folder} already exists.")

    # sun_full
    full_output_folder = os.path.join(dst_root, "sun_full")
    print("\n" + "=" * 50)
    print(f"| TASK 2: Creating FULL TEST dataset in {full_output_folder}")
    print("=" * 50)
    try:
        os.makedirs(full_output_folder, exist_ok=False)
        execute_split_task(
            positive_cases,
            negative_cases,
            full_output_folder,
            imgsz,
            is_full_test=True,
            base_dataset_folder=base_dataset_folder,
        )
    except FileExistsError:
        print(f"Skipping TASK 2: {full_output_folder} already exists.")
