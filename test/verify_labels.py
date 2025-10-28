#!/usr/bin/env python3
import os
import re
import cv2
import json
import math
import time
import glob
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def read_yaml_data_paths(yaml_path: str) -> Dict[str, str]:
    paths = {}
    with open(yaml_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("train:"):
                paths["train"] = line.split(":", 1)[1].strip()
            elif line.startswith("val:"):
                paths["val"] = line.split(":", 1)[1].strip()
            elif line.startswith("test:"):
                paths["test"] = line.split(":", 1)[1].strip()
    return paths

def parse_coco(ann_path: str) -> Tuple[Dict[str, List[List[float]]], Dict[str, int]]:
    with open(ann_path, "r") as f:
        j = json.load(f)
    id2name = {im["id"]: im["file_name"] for im in j["images"]}
    id2size = {im["file_name"]: (im["width"], im["height"]) for im in j["images"]}
    boxes = defaultdict(list)
    for a in j["annotations"]:
        if a.get("iscrowd", 0) != 0:
            continue
        fn = id2name[a["image_id"]]
        boxes[fn].append(a["bbox"])
    return boxes, id2size

def parse_yolo_label(txt_path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.isfile(txt_path):
        return []
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cls, cx, cy, w, h))
    return out

def yolo_to_coco_pixels(yolo_boxes, img_w, img_h) -> List[List[float]]:
    out = []
    for cls, cx, cy, w, h in yolo_boxes:
        x = (cx - w/2.0) * img_w
        y = (cy - h/2.0) * img_h
        pw = w * img_w
        ph = h * img_h
        out.append([x, y, pw, ph])
    return out

def voc_parse(xml_path: str) -> Tuple[int, int, List[Tuple[float,float,float,float]]]:
    tree = ET.parse(xml_path)
    r = tree.getroot()
    w = int(r.find("size").find("width").text)
    h = int(r.find("size").find("height").text)
    boxes = []
    for obj in r.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
    return w, h, boxes

def letterbox_params(orig_w: int, orig_h: int, new_shape: int) -> Tuple[float, int, int]:
    r = min(new_shape / float(orig_w), new_shape / float(orig_h))
    new_unpad_w = max(int(round(orig_w * r)), 1)
    new_unpad_h = max(int(round(orig_h * r)), 1)
    pad_w = new_shape - new_unpad_w
    pad_h = new_shape - new_unpad_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return r, pad_left, pad_top

def xyxy_to_coco(x1, y1, x2, y2):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]

def transform_voc_box_to_resized(vb, r, pad_left, pad_top, out_size):
    x1 = vb[0]*r + pad_left
    y1 = vb[1]*r + pad_top
    x2 = vb[2]*r + pad_left
    y2 = vb[3]*r + pad_top
    x1 = max(0.0, min(float(out_size), x1))
    y1 = max(0.0, min(float(out_size), y1))
    x2 = max(0.0, min(float(out_size), x2))
    y2 = max(0.0, min(float(out_size), y2))
    return xyxy_to_coco(x1, y1, x2, y2)

def iou_xywh(a, b, eps=1e-6):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = aw*ah + bw*bh - inter + eps
    return inter / ua

def almost_equal_boxes(a, b, tol=0.5):
    return all(abs(a[i] - b[i]) <= tol for i in range(4))

def draw_boxes(img, boxes, color, label_text):
    for (x, y, w, h) in boxes:
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(img, p1, p2, color, 2)
    if label_text:
        cv2.putText(img, label_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def basename(fn: str) -> str:
    return os.path.basename(fn)

def video_id_from_basename(fn: str) -> Optional[str]:
    base = os.path.splitext(basename(fn))[0]
    parts = base.split("_", 1)
    if len(parts) < 2:
        return None
    return parts[0]

def locate_xml(base_root: str, fn: str) -> Optional[str]:
    vid = video_id_from_basename(fn)
    if vid is None:
        return None
    xml_path = os.path.join(base_root, "annotations", f"{vid}_annotations", os.path.splitext(basename(fn))[0] + ".xml")
    return xml_path if os.path.isfile(xml_path) else None

def locate_original_image(base_root: str, fn: str) -> Optional[str]:
    vid = video_id_from_basename(fn)
    if vid is None:
        return None
    img_path = os.path.join(base_root, "frames", f"{vid}_frames", basename(fn))
    if not os.path.isfile(img_path):
        alt = os.path.splitext(img_path)[0] + ".png"
        if os.path.isfile(alt):
            img_path = alt
    return img_path if os.path.isfile(img_path) else None

def verify_dataset_pair(base_root: str, dst_root: str, out_dir: str, sample_basenames: List[str]) -> Dict[str, Dict]:
    report = {
        "dst": dst_root,
        "checked_images": 0,
        "ok_strict_equal": 0,
        "ok_iou999": 0,
        "ok_letterbox_match": 0,
        "mismatch_counts": 0,
        "missing_yolo": 0,
        "missing_coco": 0,
        "missing_xml": 0,
        "fail_list": []
    }

    yaml_path = os.path.join(dst_root, "data.yaml")
    paths = read_yaml_data_paths(yaml_path)

    coco_paths = {
        "train": os.path.join(dst_root, "annotations_coco_train.json"),
        "val": os.path.join(dst_root, "annotations_coco_val.json"),
        "test": os.path.join(dst_root, "annotations_coco_test.json"),
    }
    coco_boxes = {}
    coco_sizes = {}
    for sp, ap in coco_paths.items():
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"Missing COCO file: {ap}")
        boxes, sizes = parse_coco(ap)
        coco_boxes[sp] = boxes
        coco_sizes[sp] = sizes

    for split in ["train", "val", "test"]:
        img_dir = paths[split]
        lbl_dir = os.path.join(dst_root, "labels", split)

        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fn in img_files:
            report["checked_images"] += 1

            # Load YOLO (normalized), convert to pixels
            yolo_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            yolo_exists = os.path.isfile(yolo_path)
            yolo_recs = parse_yolo_label(yolo_path) if yolo_exists else []
            if not yolo_exists:
                report["missing_yolo"] += 1

            img_w, img_h = coco_sizes[split].get(fn, (None, None))
            if img_w is None:
                report["missing_coco"] += 1
                continue
            yolo_boxes_px = yolo_to_coco_pixels(yolo_recs, img_w, img_h)

            # COCO boxes
            coco_list = coco_boxes[split].get(fn, [])

            # Additional accounting for negatives/positives vs YOLO file presence
            if len(coco_list) == 0 and yolo_exists and len(yolo_recs) == 0:
                report.setdefault("negatives_with_empty_yolo", 0)
                report["negatives_with_empty_yolo"] += 1
            if len(coco_list) > 0 and (not yolo_exists or len(yolo_recs) == 0):
                report.setdefault("positives_missing_yolo", 0)
                report["positives_missing_yolo"] += 1

            # Strict compare: same count
            if len(yolo_boxes_px) != len(coco_list):
                report["mismatch_counts"] += 1
                report["fail_list"].append((split, fn, "count_mismatch", len(yolo_boxes_px), len(coco_list)))
                continue

            y_sorted = sorted(yolo_boxes_px, key=lambda b: (round(b[0],2), round(b[1],2), round(b[2],2), round(b[3],2)))
            c_sorted = sorted(coco_list, key=lambda b: (round(b[0],2), round(b[1],2), round(b[2],2), round(b[3],2)))

            if all(almost_equal_boxes(a, b, tol=0.5) for a, b in zip(y_sorted, c_sorted)):
                report["ok_strict_equal"] += 1
            else:
                if all(iou_xywh(a, b) >= 0.999 for a, b in zip(y_sorted, c_sorted)):
                    report["ok_iou999"] += 1
                else:
                    report["fail_list"].append((split, fn, "coord_mismatch", y_sorted, c_sorted))

            # Letterbox match check against VOC
            xml_path = locate_xml(base_root, fn)
            if not xml_path or not os.path.isfile(xml_path):
                report["missing_xml"] += 1
            else:
                ow, oh, voc_boxes = voc_parse(xml_path)
                m = re.search(r"_(\d{3,4})x\1$", os.path.basename(dst_root))
                out_size = int(m.group(1)) if m else img_w
                r, pad_left, pad_top = letterbox_params(ow, oh, out_size)
                voc2res = [transform_voc_box_to_resized(vb, r, pad_left, pad_top, out_size) for vb in voc_boxes]
                # Round to same precision as exporter (2 decimals) to avoid false mismatches
                voc2res_rounded = [[round(b[0],2), round(b[1],2), round(b[2],2), round(b[3],2)] for b in voc2res]
                voc_sorted = sorted(voc2res_rounded, key=lambda b: (b[0], b[1], b[2], b[3]))
                if len(voc_sorted) == len(c_sorted) and all(almost_equal_boxes(a, b, tol=0.01) for a, b in zip(voc_sorted, c_sorted)):
                    report["ok_letterbox_match"] += 1
                else:
                    # Fall back to a very tight IoU check
                    if len(voc_sorted) == len(c_sorted) and all(iou_xywh(a, b) >= 0.995 for a, b in zip(voc_sorted, c_sorted)):
                        report["ok_letterbox_match"] += 1
                    else:
                        report["fail_list"].append((split, fn, "letterbox_mismatch", voc_sorted, c_sorted))

    os.makedirs(out_dir, exist_ok=True)
    m = re.search(r"_(\d{3,4})x\1$", os.path.basename(dst_root))
    size_tag = m.group(1) if m else "X"
    for split in ["train", "val", "test"]:
        img_dir = read_yaml_data_paths(os.path.join(dst_root, "data.yaml"))[split]
        lbl_dir = os.path.join(dst_root, "labels", split)
        coco_path = os.path.join(dst_root, f"annotations_coco_{split}.json")
        if not os.path.isfile(coco_path):
            continue
        cboxes, csizes = parse_coco(coco_path)

        for fn in sample_basenames:
            img_path = os.path.join(img_dir, fn)
            if not os.path.isfile(img_path):
                continue
            im = cv2.imread(img_path)
            if im is None:
                continue

            yolo_txt = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            yolo_recs = parse_yolo_label(yolo_txt) if os.path.isfile(yolo_txt) else []
            w,h = csizes.get(fn, (im.shape[1], im.shape[0]))
            ypx = yolo_to_coco_pixels(yolo_recs, w, h)
            cpx = cboxes.get(fn, [])

            vis = im.copy()
            draw_boxes(vis, cpx, (0,255,0), f"COCO ({size_tag})")
            draw_boxes(vis, ypx, (0,128,255), f"YOLO ({size_tag})")
            out_path = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}__resized_{size_tag}.jpg")
            cv2.imwrite(out_path, vis)

    for fn in sample_basenames:
        orig_img = locate_original_image(base_root, fn)
        xml_path = locate_xml(base_root, fn)
        if not orig_img or not os.path.isfile(orig_img) or not xml_path:
            continue
        im = cv2.imread(orig_img)
        if im is None:
            continue
        ow, oh, voc_boxes = voc_parse(xml_path)
        boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in voc_boxes]
        vis = im.copy()
        draw_boxes(vis, boxes_xywh, (255,0,0), "VOC original")
        out_path = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}__original.jpg")
        cv2.imwrite(out_path, vis)

    return report

def main():
    ap = argparse.ArgumentParser(description="Verify YOLO vs COCO vs VOC (letterbox) and create visual checks.")
    ap.add_argument("--base", required=True, help="REAL-Colon original root (has annotations/ and frames/)")
    ap.add_argument("--dst", action="append", required=True,
                    help="Resized dataset root, repeat for 224x224 and 640x640 (expects data.yaml, images/, labels/).")
    ap.add_argument("--out", default=None, help="Output folder for visual samples and report JSON.")
    ap.add_argument("--samples", type=int, default=20, help="Number of basenames to sample for visualization.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    args = ap.parse_args()

    dst_roots = args.dst
    if len(dst_roots) < 1:
        raise SystemExit("Provide at least one --dst; usually two: 224x224 and 640x640.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.expanduser(f"~/master-thesis/test/verify_labels_output_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    split = "train"
    per_dst_files = []
    for dst in dst_roots:
        yml = os.path.join(dst, "data.yaml")
        paths = read_yaml_data_paths(yml)
        train_dir = paths[split]
        files = {f for f in os.listdir(train_dir) if f.lower().endswith((".jpg",".jpeg",".png"))}
        per_dst_files.append(files)
    common = set.intersection(*per_dst_files) if len(per_dst_files) > 1 else per_dst_files[0]

    first_dst = dst_roots[0]
    coco_train_path = os.path.join(first_dst, "annotations_coco_train.json")
    cboxes, _ = parse_coco(coco_train_path)
    positive = [fn for fn in common if len(cboxes.get(fn, [])) > 0]
    random.seed(args.seed)
    chosen = random.sample(positive, min(args.samples, len(positive))) if positive else random.sample(list(common), min(args.samples, len(common)))

    reports = []
    for dst in dst_roots:
        rep = verify_dataset_pair(args.base, dst, out_dir, chosen)
        reports.append(rep)

    report_path = os.path.join(out_dir, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Wrote report: {report_path}")
    print(f"Visual samples saved under: {out_dir}")

if __name__ == "__main__":
    main()
