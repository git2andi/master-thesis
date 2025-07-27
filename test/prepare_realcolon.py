import os
import random
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from shutil import copy2

# === CONFIGURATION ===
SRC = "/mnt/data/aschwab/data/realColon"
DST = "/mnt/data/aschwab/data/realColon_final"
SPLIT = {"train": 48, "val": 6, "test": 6}
CLASS_MAP = {"lesion": 0}
random.seed(42)  # Reproducible shuffle

# === CREATE OUTPUT STRUCTURE ===
for split in SPLIT:
    os.makedirs(os.path.join(DST, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DST, 'labels', split), exist_ok=True)

# === FIND VIDEO FOLDERS ===
frame_folders = sorted(glob(os.path.join(SRC, "frames", "*_frames")))
anno_folders = sorted(glob(os.path.join(SRC, "annotations", "*_annotations")))
video_ids = sorted([Path(f).stem.replace("_frames", "") for f in frame_folders])
assert len(video_ids) == 60, "Expected 60 videos in realColon dataset"

# === RANDOMIZED VIDEO SPLIT ===
random.shuffle(video_ids)  # Shuffle to avoid center-wise bias
train_ids = video_ids[:SPLIT["train"]]
val_ids = video_ids[SPLIT["train"]:SPLIT["train"] + SPLIT["val"]]
test_ids = video_ids[SPLIT["train"] + SPLIT["val"]:]

split_map = {vid: "train" for vid in train_ids}
split_map.update({vid: "val" for vid in val_ids})
split_map.update({vid: "test" for vid in test_ids})

# === FUNCTION TO CONVERT BBOX TO YOLO FORMAT ===
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

# === PROCESS EACH VIDEO ===
for vid in video_ids:
    frame_dir = os.path.join(SRC, "frames", f"{vid}_frames")
    anno_dir = os.path.join(SRC, "annotations", f"{vid}_annotations")
    split = split_map[vid]
    
    frame_files = sorted(glob(os.path.join(frame_dir, "*.jpg")))
    for img_path in frame_files:
        base_name = Path(img_path).stem
        xml_file = os.path.join(anno_dir, f"{base_name}.xml")
        if not os.path.exists(xml_file):
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            continue

        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        label_lines = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls]
            bndbox = obj.find("bndbox")
            bbox = (
                int(bndbox.find("xmin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("ymax").text)
            )
            yolo_box = convert_bbox((w, h), bbox)
            label_lines.append(f"{cls_id} {' '.join(f'{v:.6f}' for v in yolo_box)}")

        # Write label .txt file
        label_out = os.path.join(DST, "labels", split, f"{base_name}.txt")
        with open(label_out, "w") as f:
            f.write("\n".join(label_lines))

        # Copy image to YOLO directory
        img_out = os.path.join(DST, "images", split, f"{base_name}.jpg")
        copy2(img_path, img_out)

    print(f"✅ Processed video folder: {vid} → {split}")

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

print("\nDone! dataset ready at:", DST)
