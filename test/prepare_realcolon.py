import os
import xml.etree.ElementTree as ET
from glob import glob
import random
import shutil
from pathlib import Path

# INPUT Paths
SRC = "/mnt/data/aschwab/data/realColon"
FRAME_ROOT = os.path.join(SRC, "frames")
ANN_ROOT = os.path.join(SRC, "annotations")

# OUTPUT Paths (updated)
DST = "/mnt/data/aschwab/data/realColon_yolo"
IMG_BASE = os.path.join(DST, "images")
LBL_BASE = os.path.join(DST, "labels")

# Set seed
random.seed(42)

# Make output folders
for subset in ['train', 'val']:
    os.makedirs(os.path.join(IMG_BASE, subset), exist_ok=True)
    os.makedirs(os.path.join(LBL_BASE, subset), exist_ok=True)

# Get all video annotation folders
ann_folders = sorted(glob(os.path.join(ANN_ROOT, "*_annotations")))
random.shuffle(ann_folders)

split_idx = int(0.8 * len(ann_folders))
train_folders = ann_folders[:split_idx]
val_folders = ann_folders[split_idx:]

def gather_xmls(folders):
    xmls = []
    for folder in folders:
        xmls.extend(sorted(glob(os.path.join(folder, "*.xml"))))
    return xmls

train_xmls = gather_xmls(train_folders)
val_xmls = gather_xmls(val_folders)


def get_frame_path_from_annotation(xml_file):
    basename = Path(xml_file).name.replace('.xml', '.jpg')
    video_id = basename.split('_')[0]  # e.g., 001-001
    frame_dir = os.path.join(FRAME_ROOT, f"{video_id}_frames")
    return os.path.join(frame_dir, basename)

def convert_and_save(xml_list, subset):
    for xml_file in xml_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        label_file = os.path.join(LBL_BASE, subset, filename.replace('.jpg', '.txt'))
        lines = []

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

        src_img = get_frame_path_from_annotation(xml_file)
        dst_img = os.path.join(IMG_BASE, subset, filename)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

convert_and_save(train_xmls, 'train')
convert_and_save(val_xmls, 'val')

print(f"Saved {len(train_xmls)} training and {len(val_xmls)} validation samples in {DST}")
