import os
import glob
import json
import numpy as np
import argparse
from pathlib import Path

def get_unpadded_name(padded_name):
    """
    Converts '001-013_000010.txt' -> '001-013_10.txt'
    """
    stem = Path(padded_name).stem # 001-013_000010
    extension = Path(padded_name).suffix # .txt or .jpg
    
    parts = stem.split("_")
    video_prefix = "_".join(parts[:-1])
    frame_num = int(parts[-1]) # automatically strips leading zeros
    
    return f"{video_prefix}_{frame_num}{extension}"

def compare_yolo_dir(new_root, old_root, split="train", tolerance=0.01):
    print(f"\n--- Checking YOLO Labels: {split} ---")
    
    new_labels_path = os.path.join(new_root, "labels", split)
    old_labels_path = os.path.join(old_root, "labels", split)
    
    # Get all new files
    new_files = glob.glob(os.path.join(new_labels_path, "*.txt"))
    
    matches = 0
    mismatches = 0
    missing = 0
    
    print(f"Found {len(new_files)} files in NEW dataset.")

    for new_f in new_files:
        filename = os.path.basename(new_f)
        old_filename = get_unpadded_name(filename)
        old_f = os.path.join(old_labels_path, old_filename)
        
        if not os.path.exists(old_f):
            # This might happen if your old set had subsampling and new one doesn't
            # or just strict missing file
            # print(f"Missing in old set: {old_filename}") 
            missing += 1
            continue

        # Load boxes
        with open(new_f, 'r') as nf, open(old_f, 'r') as of:
            new_data = [list(map(float, line.split())) for line in nf.readlines()]
            old_data = [list(map(float, line.split())) for line in of.readlines()]

        if len(new_data) != len(old_data):
            print(f"[FAIL] Box count mismatch: {filename} (New: {len(new_data)}, Old: {len(old_data)})")
            mismatches += 1
            continue

        # Compare Coordinates
        # Sort by x coordinate to ensure we compare same boxes
        new_data.sort(key=lambda x: x[1])
        old_data.sort(key=lambda x: x[1])
        
        is_match = True
        for nb, ob in zip(new_data, old_data):
            # nb = [class, cx, cy, w, h]
            if int(nb[0]) != int(ob[0]): # Class check
                is_match = False
                break
            
            # Coordinate check (cx, cy, w, h)
            # We use a small epsilon because of the geometry fix
            if not np.allclose(nb[1:], ob[1:], atol=tolerance):
                is_match = False
                break
        
        if is_match:
            matches += 1
        else:
            print(f"[FAIL] Values differ: {filename}")
            # print(f"   New: {new_data}")
            # print(f"   Old: {old_data}")
            mismatches += 1

    print(f"Summary {split}: {matches} Matches | {mismatches} Value Failures | {missing} Missing in Old Set")

def compare_coco_json(new_json_path, old_json_path, tolerance=2.0):
    print(f"\n--- Checking COCO JSON: {os.path.basename(new_json_path)} ---")
    
    if not os.path.exists(old_json_path):
        print(f"Skipping: Old JSON not found at {old_json_path}")
        return

    with open(new_json_path, 'r') as f: new_data = json.load(f)
    with open(old_json_path, 'r') as f: old_data = json.load(f)

    # Index Old Data by "Unpadded Filename" for fast lookup
    # Map: '001-013_10.jpg' -> [bboxes...]
    old_lookup = {}
    
    # 1. Map Image IDs to Filenames
    old_img_id_to_name = {img['id']: img['file_name'] for img in old_data['images']}
    
    # 2. Build Box Lookup
    for ann in old_data['annotations']:
        img_name = old_img_id_to_name[ann['image_id']]
        if img_name not in old_lookup: old_lookup[img_name] = []
        old_lookup[img_name].append(ann['bbox']) # [x,y,w,h]

    matches = 0
    mismatches = 0
    
    # Iterate New Data
    new_img_id_to_name = {img['id']: img['file_name'] for img in new_data['images']}
    
    for ann in new_data['annotations']:
        new_name = new_img_id_to_name[ann['image_id']]
        old_name = get_unpadded_name(new_name) # Convert 000010 -> 10
        
        new_box = ann['bbox']
        
        if old_name not in old_lookup:
            # Maybe frame didn't exist in old set
            continue
            
        # Find matching box in old list (greedy match)
        found = False
        old_boxes = old_lookup[old_name]
        
        for idx, ob in enumerate(old_boxes):
            # Check absolute pixel difference
            # Tolerance is in pixels (e.g., 2.0 pixels diff allowed)
            if np.allclose(new_box, ob, atol=tolerance):
                found = True
                break
        
        if found:
            matches += 1
        else:
            mismatches += 1
            # print(f"Box mismatch in {new_name}: {new_box}")

    print(f"JSON Check: {matches} Boxes Matched | {mismatches} Boxes Failed/Missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str, required=True, help="Root of new dataset (padded)")
    parser.add_argument("--old", type=str, required=True, help="Root of old dataset (unpadded)")
    args = parser.parse_args()

    # Check YOLO Labels
    for split in ['train', 'val', 'test']:
        if os.path.exists(os.path.join(args.new, "labels", split)):
            compare_yolo_dir(args.new, args.old, split)

    # Check COCO JSONs
    # Assuming standard naming from your script
    pairs = [
        ("coco_annotations_train.json", "coco_annotations_train.json"), # Names might differ?
        ("coco_annotations_val.json",   "coco_annotations_val.json"),
        ("coco_annotations_test.json",  "coco_annotations_test.json"),
    ]
    
    # If your old files were named differently (e.g. train_ann.json), update mapping here manually or rename them
    # Based on your first upload, old names were 'train_ann.json', etc.
    # New names in prepare_realColon.py are 'coco_annotations_train.json'
    
    pairs = [
        ("coco_annotations_train.json", "train_ann.json"),
        ("coco_annotations_val.json",   "validation_ann.json"),
        ("coco_annotations_test.json",  "test_ann.json"),
    ]

    for new_name, old_name in pairs:
        compare_coco_json(
            os.path.join(args.new, new_name),
            os.path.join(args.old, old_name)
        )



# python check_consistency.py \
#  --old /data/local/aschwab/data/realColon_640x640 \
#  --new /data/local/aschwab/data/realColon_coco_640_all