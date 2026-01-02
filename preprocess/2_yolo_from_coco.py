""" 
Script to generate a clean YOLO dataset structure with zero-padding names (to ensure ordering) and symlinks.
"""

import json
import os
from tqdm import tqdm # pip install tqdm

def get_padded_filename(original_name, padding=6):
    """
    001-001_1.jpg > 001-001_000001.jpg
    assume: institution_videoID.frameNum
    """
    base, ext = os.path.splitext(original_name)
    parts = base.rsplit('_', 1)
    
    if len(parts) == 2:
        video_id = parts[0]
        frame_num = parts[1]
        if frame_num.isdigit():
            new_name = f"{video_id}_{int(frame_num):0{padding}d}{ext}"
            return new_name
    return original_name

def convert_to_yolo_bbox(bbox, img_w, img_h):
    # Converts COCO bbox (xmin, ymin, w, h) to normalized YOLO (xc, yc, w, h)
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w /= img_w
    h /= img_h
    return max(0, min(1, x_center)), max(0, min(1, y_center)), max(0, min(1, w)), max(0, min(1, h))

def process_subset(root_path, json_filename, source_images_folder, subset_name):
    """
    takes root, json file ({train,validation,test}_ann.json), source image folder ({train,validation,test}_images), and destination folder (train,test,val)
    """
    json_path = os.path.join(root_path, json_filename)
    source_img_dir = os.path.join(root_path, source_images_folder)
    
    dest_img_dir = os.path.join(root_path, "images", subset_name)
    dest_lbl_dir = os.path.join(root_path, "labels", subset_name)
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    print(f"processing {subset_name}")
    print(f"  source: {source_img_dir}")
    print(f"  dest images: {dest_img_dir}")
    print(f"  dest labels: {dest_lbl_dir}")

    if not os.path.exists(json_path):
        print(f"{json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    images_info = {img['id']: img for img in data['images']}
    anns_map = {img_id: [] for img_id in images_info.keys()}
    if 'annotations' in data:
        for ann in data['annotations']:
            anns_map[ann['image_id']].append(ann)

    for img_id, img_data in tqdm(images_info.items()):
        original_fname = img_data['file_name']
        
        # zero pad
        new_filename = get_padded_filename(original_fname)
        new_txt_name = os.path.splitext(new_filename)[0] + ".txt"

        # syslink
        current_symlink_path = os.path.join(source_img_dir, original_fname)
        target_dest_path = os.path.join(dest_img_dir, new_filename)

        if os.path.exists(current_symlink_path):
            real_source_path = os.path.realpath(current_symlink_path)
            
            if not os.path.exists(target_dest_path):
                os.symlink(real_source_path, target_dest_path)
        else:
            print(f"{original_fname} not found in {source_images_folder}")
            continue

        # label
        txt_path = os.path.join(dest_lbl_dir, new_txt_name)
        with open(txt_path, 'w') as f_out:
            annotations = anns_map.get(img_id, [])
            for ann in annotations:
                class_id = 0
                xc, yc, w, h = convert_to_yolo_bbox(ann['bbox'], img_data['width'], img_data['height'])
                f_out.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"finished and {len(images_info)} images processed for {subset_name}.\n")

# use train2, ... directly as in script 3) syslinks are made actual files within train2, ... folders
# Then data.yaml does not need manual change :)
def create_data_yaml(root_path):
    yaml_path = os.path.join(root_path, "data.yaml")
    yaml_content = f"""
path: {root_path}
train: images/train2
val: images/val2
test: images/test2

# Classes
nc: 1
names:
  0: lesion
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YOLO config: {yaml_path}")


if __name__ == "__main__":
    DATASET_ROOT = "/data/local/aschwab/data/real_colon_allPos_fraction1ofNeg_onlyPatient" # root
    process_subset(DATASET_ROOT, "train_ann.json", "train_images", "train2") # orig coco annotation, orig image path, dest path
    process_subset(DATASET_ROOT, "validation_ann.json", "validation_images", "val2")
    process_subset(DATASET_ROOT, "test_ann.json", "test_images", "test2")
    create_data_yaml(DATASET_ROOT)

    print("done")