# RealColon Preporcesing
# Adapted from https://github.com/cosmoimd/real-colon-dataset
# add resizing and YOLO labels
# dropped Pos Frames subsampling as not required

import os
import json
import argparse
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

from helper.geometry import letterbox_pil, clip_box, yolo_norm_from_abs
from helper.sampling import select_pos_neg_frames
from helper.coco import add_coco_image, add_coco_annotation
from helper.yolo import write_yolo_label_file

# YOLO metadata
YOLO_ID = 0 #id for 'lesion' (0 to match Ultralytics)
YOLO_CLASS = "lesion"

# COCO metadata from https://github.com/cosmoimd/real-colon-dataset
COCO_INFO = {
    "description": "Cosmo data",
    "url": "http://cosmoimd.com",
    "version": "1.0",
    "year": 2023,
    "contributor": "CosmoIMD",
    "date_created": "2023/02/28",
}

COCO_LICENSES = [{
    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
}]
COCO_ID = 1 # id for 'lesion' (1 to match original & detectron2)
COCO_CATEGORIES = [{"supercategory": "lesion", "id": COCO_ID, "name": "lesion"}]


def parsevocfile(annotation_file):
    if not os.path.exists(annotation_file):
        raise Exception("Cannot find bounding box file %s" % (annotation_file))
    try:
        tree = ET.parse(annotation_file)
    except Exception as e:
        print(e)
        raise Exception("Failed to open annotation file %s" % annotation_file)

    img = {}
    cboxes = []
    filename = None

    for elem in tree.iter():
        if 'filename' in elem.tag:
            filename = elem.text
        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'depth' in elem.tag:
            img['depth'] = int(elem.text)


        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}
            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text
                if 'unique_id' in attr.tag:
                    obj['unique_id'] = attr.text

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            l = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            t = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            r = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            b = int(round(float(dim.text)))
                    obj["box_ltrb"] = [l, t, r, b]

            cboxes.append(obj)
    
    img_shape = (img["height"], img["width"], img["depth"])
    return {
        "boxes": cboxes, 
        "img_shape": img_shape, 
        "img_name": filename 
    }


def convert_video_list(base_dataset_folder: str,
                       video_list: list[str],
                       annotation_list: list[str],
                       frames_output_folder: str,
                       labels_output_folder: str,
                       json_output_file: str,
                       imgsz: int,
                       negative_ratio: float = 0.0):

    """
    Args:
        base_dataset_folder: Base folder for the unzipped REAL-Colon dataset. All in Root.
        video_list:          List of video folder names (['001-001_frames', ...]).
        annotation_list:     Matching list of annotation folder names (['001-001_annotations', ...]).
        frames_output_folder: Output folder for frames (or symlinks), relative to `base_dataset_folder`.
        labels_output_folder: Output folder for YOLO labels, relative to `base_dataset_folder`.
        json_output_file:    Path to the COCO JSON file to write.
        imgsz:               Target square size for letterbox resizing. Set to -1 to use original size.
        negative_ratio:      Fraction of negative frames to keep per video, in [0,1].
                             - 0.0 > no negatives
                             - 1.0 > keep all negatives
                             - -1 > match num of negatives with positives (so 1:1 ratio)
    """
    data = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    images_uniq_id = {}
    image_uniq_id_cnt = 0
    image_uniq_box_cnt = 0
    uniq_box_to_lesion_association = {}
    for video_idx, (curr_video_folder, curr_ann_folder) in enumerate(zip(video_list, annotation_list)):
        print(f"Processing video {video_idx}: {curr_video_folder}")

        frames_dir = os.path.join(base_dataset_folder, curr_video_folder)
        ann_dir = os.path.join(base_dataset_folder, curr_ann_folder)

        all_images = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        all_xmls = sorted(
            [f for f in os.listdir(ann_dir) if f.endswith(".xml")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        if len(all_images) != len(all_xmls):
            raise Exception(
                f"Image and annotations must have same length for {curr_video_folder}: "
                f"{len(all_images)} images vs {len(all_xmls)} xmls"
            )

        all_datas = []
        num_boxes_indexes = []
        for c_xml in all_xmls:
            xml_path = os.path.join(ann_dir, c_xml)
            c_data = parsevocfile(xml_path)
            all_datas.append(c_data)
            num_boxes_indexes.append(len(c_data["boxes"]))        
        num_frames = len(all_datas)

        frames_wbox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v > 0]
        frames_nobox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v == 0]

        per_lesion_dict = {}
        for cidx, c_data in enumerate(all_datas):
            for cbox in c_data['boxes']:
                cname = cbox['unique_id']
                if cname not in per_lesion_dict:
                    per_lesion_dict[cname] = []
                per_lesion_dict[cname].append(cidx)
        
        # Debug: how many lesions and how many frames per lesion
        lesion_lengths = [len(per_lesion_dict[x]) for x in per_lesion_dict.keys()]
        print(
            f"Found {len(per_lesion_dict)} lesions with "
            + " - ".join(str(n) for n in lesion_lengths)
            + " frames each"
        )     

        selected_pos_idxs, selected_neg_idxs, info = select_pos_neg_frames(
            frames_wbox_indexes,
            frames_nobox_indexes,
            negative_ratio=negative_ratio,
            seed=1000,
        )

        print(
            f"[{info['mode']}] Using {info['num_pos']} positives and "
            f"{info['num_neg_kept']} / {info['num_neg']} negatives "
            f"(negative_ratio={info['negative_ratio']})"
        )

        # Combine positives + negatives
        selected_frames = sorted(selected_pos_idxs + selected_neg_idxs)
        xml_to_be_used = [all_xmls[i] for i in selected_frames]


        for c_xml in xml_to_be_used:
            xml_path = os.path.join(ann_dir, c_xml)
            c_data = parsevocfile(xml_path)

            img_name = c_data["img_name"]
            base_name = os.path.splitext(c_xml)[0]  # works for 001-012 (all files tailing .0) or others (no .0)
            frame_part = base_name.split("_")[-1]   # 2.0
            frame_num = int(float(frame_part))      # 2.0 > 2
            img_name = base_name + ".jpg"           # 001-012_2.0.jpg
            video_prefix = "_".join(base_name.split("_")[:-1])
            new_img_name = f"{video_prefix}_{frame_num:06d}.jpg" # 001-012_000002.jpg
            H_orig, W_orig, _ = c_data["img_shape"]

            # Resizing 
            src_img_path = os.path.join(frames_dir, img_name)
            img = Image.open(src_img_path).convert("RGB")
            img_sq, scale_w, scale_h, pad_l, pad_t = letterbox_pil(
                img,
                new_shape=imgsz,
                color=(114, 114, 114),
            )
            out_W = imgsz
            out_H = imgsz
            
            dst_img_path = os.path.join(frames_output_folder, new_img_name)
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            img_sq.save(dst_img_path, format="JPEG", quality=100, subsampling=0)

            add_coco_image(
                data,
                image_id=image_uniq_id_cnt,
                file_name=new_img_name,
                width=out_W,
                height=out_H,
                license_id=1,
            )
            images_uniq_id[image_uniq_id_cnt] = img_name

            yolo_boxes = []

            for cbox in c_data["boxes"]:
                l_raw, t_raw, r_raw, b_raw = cbox["box_ltrb"]

                # Map original box to square coords
                x1 = l_raw * scale_w + pad_l
                y1 = t_raw * scale_h + pad_t
                x2 = r_raw * scale_w + pad_l
                y2 = b_raw * scale_h + pad_t

                # clip oob
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, imgsz, imgsz)
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0.0 or bh <= 0.0:
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
        
                # lesion & box association
                uid = cbox["unique_id"]
                if uid not in uniq_box_to_lesion_association:
                    uniq_box_to_lesion_association[uid] = []
                uniq_box_to_lesion_association[uid].append(image_uniq_box_cnt)

                cx, cy, ww, hh = yolo_norm_from_abs(x1, y1, x2, y2, w=imgsz, h=imgsz)
                yolo_boxes.append((YOLO_ID, cx, cy, ww, hh))

                image_uniq_box_cnt += 1


            # Write YOLO labels
            yolo_out = os.path.join(
                labels_output_folder,
                    Path(new_img_name).with_suffix(".txt").name,
                )
            os.makedirs(os.path.dirname(yolo_out), exist_ok=True)
            write_yolo_label_file(yolo_out, yolo_boxes)

            image_uniq_id_cnt += 1
        
    # Write COCO labels
    with open(json_output_file, "w") as f:
        json.dump(data, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/data/local/aschwab/data/realColon"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/data/local/aschwab/data/realColon_coco_resized"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    base_dataset_folder = args.src
    output_folder = args.dst
    negative_ratio = args.negative_ratio
    imgsz = args.imgsz
    os.makedirs(output_folder, exist_ok=False)

    # Split
    NUM_TRAIN_VIDEOS_PER_SET = 10  # first 10 videos per set > train
    NUM_VALID_VIDEOS_PER_SET = 2   # next 2 > val, remaining 3 > test

    video_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_frames")])
    annotation_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_annotations")])


    images_root = os.path.join(output_folder, "images")
    labels_root = os.path.join(output_folder, "labels")
    os.makedirs(images_root, exist_ok=False)
    os.makedirs(labels_root, exist_ok=False)

    train_images_folder = os.path.join(images_root, "train")
    val_images_folder   = os.path.join(images_root, "val")
    test_images_folder  = os.path.join(images_root, "test")
    
    train_labels_folder = os.path.join(labels_root, "train")
    val_labels_folder   = os.path.join(labels_root, "val")
    test_labels_folder  = os.path.join(labels_root, "test")

    json_output_file_train = os.path.join(output_folder, "coco_annotations_train.json")
    json_output_file_val   = os.path.join(output_folder, "coco_annotations_val.json")
    json_output_file_test  = os.path.join(output_folder, "coco_annotations_test.json")
    
    
    
    # Train
    video_list_train = [
        x for x in video_list
        if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET
    ]
    annotation_list_train = [
        x for x in annotation_list
        if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET
    ]

    convert_video_list(
        base_dataset_folder,
        video_list_train,
        annotation_list_train,
        train_images_folder,
        train_labels_folder,
        json_output_file_train,
        imgsz=imgsz,
        negative_ratio=negative_ratio,
    )
    print("Training subset conversion completed")

    # Val
    video_list_val = [
        x for x in video_list
        if (NUM_TRAIN_VIDEOS_PER_SET
            < int(x.split("-")[1].split("_")[0])
            <= NUM_TRAIN_VIDEOS_PER_SET + NUM_VALID_VIDEOS_PER_SET)
    ]
    annotation_list_val = [
        x for x in annotation_list
        if (NUM_TRAIN_VIDEOS_PER_SET
            < int(x.split("-")[1].split("_")[0])
            <= NUM_TRAIN_VIDEOS_PER_SET + NUM_VALID_VIDEOS_PER_SET)
    ]

    convert_video_list(
        base_dataset_folder,
        video_list_val,
        annotation_list_val,
        val_images_folder,
        val_labels_folder,
        json_output_file_val,
        imgsz=imgsz,
        negative_ratio=negative_ratio,
    )
    print("Validation subset conversion completed")

    # Test
    video_list_test = [
        x for x in video_list
        if int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET + NUM_VALID_VIDEOS_PER_SET
    ]
    annotation_list_test = [
        x for x in annotation_list
        if int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET + NUM_VALID_VIDEOS_PER_SET
    ]

    convert_video_list(
        base_dataset_folder,
        video_list_test,
        annotation_list_test,
        test_images_folder,
        test_labels_folder,
        json_output_file_test,
        imgsz=imgsz,
        negative_ratio=negative_ratio,
    )
    print("Testing subset conversion completed")

    # Write YOLO data.yaml
    data_yaml_path = os.path.join(output_folder, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(
            f"""path: {output_folder}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["lesion"]
"""
        )


'''
## Run via
python prepare_realColon.py \
  --src /data/local/aschwab/data/realColon \
  --dst /data/local/aschwab/data/realColon_xxx \
  --imgsz 640 \
  --negative_ratio 1.0
'''