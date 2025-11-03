#!/usr/bin/env python3
"""
verify_realcolon_labels.py

Cross-check that the YOLO and COCO labels in your processed REAL-Colon datasets
(**full**, **640x640**, **224x224**) match the **original VOC XML** labels under
`/data/local/aschwab/data/realColon/{annotations,frames}`.

It verifies, per image:
  • Original VOC → list of GT boxes (xyxy) and image size (W,H)
  • Variant YOLO label (.txt) → boxes (xyxy absolute), compare vs expected
  • Variant COCO JSON → boxes (xywh), compare vs expected

For square variants (224/640), expected boxes are obtained from original by a
**letterbox transform** (scale + center padding) to the target size.

Output:
  • A compact **summary table** on stdout per variant & subset (train/val/test)
  • CSV logs under `meta/verify/` with any mismatches, degenerate boxes, or
    missing files (one CSV per variant+subset for YOLO and COCO each)

Assumptions:
  • Filenames retain the original stem: `SSS-VVV_FFFFF.jpg` (e.g., 001-001_44456)
  • Original VOC XML is located in `/annotations/SSS-VVV_annotations/SSS-VVV_FFFFF.xml`.
  • Processed COCO JSONs sit at the variant root as `annotations_coco_{split}.json`.

Usage example
-------------
python verify_realcolon_labels.py \
  --orig-root /data/local/aschwab/data/realColon \
  --variant /data/local/aschwab/data/realColon_full:full:0 \
  --variant /data/local/aschwab/data/realColon_640x640:640:1 \
  --variant /data/local/aschwab/data/realColon_224x224:224:1

Where each `--variant` is `ROOT:TAG:IS_SQUARE`, with IS_SQUARE 0/1
(0 = original-res/no letterbox, 1 = letterbox square; TAG is just a short name).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET
from PIL import Image

SUBSETS = ("train", "val", "test")

# Deterministic per-institution split (SSS-VVV):
# For each institution SSS in {001,002,003,004}:
#   VVV 001-010 -> train
#   VVV 011-012 -> val
#   VVV 013-015 -> test
SPLIT_RULE = {
    "train":  range(1, 11),   # 001..010
    "val":    range(11, 13),  # 011..012
    "test":   range(13, 16),  # 013..015
}


# -------------------- helpers --------------------

def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[float,float,float,float]]]:
    """Return (W,H, boxes_xyxy). Single-class dataset; ignore <name>.
    If multiple <object>, return all boxes.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(size.findtext("width"))
    H = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1 = float(bb.findtext("xmin"))
        y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax"))
        y2 = float(bb.findtext("ymax"))
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        boxes.append((x1,y1,x2,y2))
    return W, H, boxes


def yolo_txt_to_xyxy(txt_path: Path, W: int, H: int) -> List[Tuple[float,float,float,float]]:
    """Parse YOLO txt lines: '0 cx cy w h' normalized. Return abs xyxy.
    Empty or nonexistent file => empty list.
    """
    if not txt_path.exists():
        return []
    lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    boxes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            # tolerate extra spaces or malformed
            try:
                cls, cx, cy, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            except Exception:
                continue
        cls, cx, cy, w, h = parts
        try:
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        except ValueError:
            continue
        bw = w * W
        bh = h * H
        x1 = (cx * W) - bw/2.0
        y1 = (cy * H) - bh/2.0
        x2 = x1 + bw
        y2 = y1 + bh
        # do NOT clamp; we want to see mismatches
        boxes.append((x1,y1,x2,y2))
    return boxes


def coco_load_index(coco_json: Path) -> Tuple[Dict[str,int], Dict[int, List[Tuple[float,float,float,float]]]]:
    """Index COCO by file_name and map image_id -> list of boxes (xywh)."""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    # image_id -> (file_name, W, H)
    id2fn: Dict[int, Tuple[str,int,int]] = {}
    for im in data.get("images", []):
        id2fn[int(im["id"])]= (im["file_name"], int(im["width"]), int(im["height"]))
    # file_name -> image_id
    fn2id: Dict[str,int] = {fn: iid for iid,(fn,_,_) in id2fn.items()}
    # image_id -> list[xywh]
    ann_map: Dict[int, List[Tuple[float,float,float,float]]] = {}
    for a in data.get("annotations", []):
        iid = int(a["image_id"]) ; bbox = a.get("bbox", [0,0,0,0])
        ann_map.setdefault(iid, []).append(tuple(map(float, bbox)))
    return fn2id, ann_map


def stem_to_case_and_xml_stem(stem: str) -> Tuple[str,str]:
    """From '001-001_44456' -> ('001-001', '001-001_44456')."""
    parts = stem.split("_")
    case = parts[0]
    return case, stem


def sss_vvv_from_case(case: str) -> Tuple[int,int]:
    """From '001-001' -> (1,1). Robust to leading zeros."""
    sss, vvv = case.split("-")
    return int(sss), int(vvv)


def expected_subset_for_case(case: str) -> str:
    """Return 'train'|'val'|'test' according to the deterministic 10/2/3 rule."""
    _, vvv = sss_vvv_from_case(case)
    for subset, rng in SPLIT_RULE.items():
        if vvv in rng:
            return subset
    return "unknown"


def expected_box_xyxy_from_letterbox(box_xyxy: Tuple[float,float,float,float], Wo: int, Ho: int, Wt: int, Ht: int) -> Tuple[float,float,float,float]:
    """Apply the **exact** letterbox used in your pipeline:
    r = min(new_shape / w, new_shape / h)
    new_unpad = (round(w*r), round(h*r))  # with int(round(...)) and min 1
    pad_left = (new_shape - new_unpad_w) // 2
    pad_top  = (new_shape - new_unpad_h) // 2
    This function assumes square targets: Wt == Ht == new_shape.
    """
    assert Wt == Ht, "Expected square target for letterbox variants"
    new_shape = Wt
    r = min(new_shape / float(Wo), new_shape / float(Ho))
    new_unpad_w = max(int(round(Wo * r)), 1)
    new_unpad_h = max(int(round(Ho * r)), 1)
    pad_w = new_shape - new_unpad_w
    pad_h = new_shape - new_unpad_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    x1, y1, x2, y2 = box_xyxy
    return (
        x1 * r + pad_left,
        y1 * r + pad_top,
        x2 * r + pad_left,
        y2 * r + pad_top,
    )


def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    iw = max(0.0, min(ax2,bx2) - max(ax1,bx1))
    ih = max(0.0, min(ay2,by2) - max(ay1,by1))
    inter = iw*ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0,(ax2-ax1)) * max(0.0,(ay2-ay1))
    area_b = max(0.0,(bx2-bx1)) * max(0.0,(by2-by1))
    denom = area_a + area_b - inter
    return inter/denom if denom>0 else 0.0


def compare_boxes(voc_boxes: List[Tuple[float,float,float,float]], cand_boxes: List[Tuple[float,float,float,float]], tol_px: float=2.0, tol_iou: float=0.995) -> Tuple[int,int,List[Tuple[int,Tuple[float,float,float,float],Tuple[float,float,float,float],float]]]:
    """Greedy match on IoU, count matches (IoU>=tol_iou) and record mismatches.
    Returns (n_expected, n_matched, mismatches)
    mismatches items: (idx, expected_box, candidate_box, iou)
    """
    used = [False]*len(cand_boxes)
    n_match = 0
    mismatches = []
    for i, eb in enumerate(voc_boxes):
        best_j = -1
        best_iou = -1.0
        for j, cb in enumerate(cand_boxes):
            if used[j]:
                continue
            iou = iou_xyxy(eb, cb)
            if iou > best_iou:
                best_iou = iou ; best_j = j
        if best_j>=0 and best_iou >= tol_iou:
            used[best_j]=True
            n_match += 1
        else:
            # keep the best candidate (even if none)
            cb = cand_boxes[best_j] if best_j>=0 else (math.nan,)*4
            mismatches.append((i, eb, cb, best_iou))
    return len(voc_boxes), n_match, mismatches


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------- main verification --------------------

def verify_variant(orig_root: Path, variant_root: Path, tag: str, is_square: bool, out_dir: Path):
    print(f"== Variant: {tag} ==")

    # Load COCO JSON indices once per subset
    coco_idx: Dict[str, Tuple[Dict[str,int], Dict[int, List[Tuple[float,float,float,float]]]]] = {}
    for subset in SUBSETS:
        cj = variant_root / f"annotations_coco_{subset}.json"
        if cj.exists():
            coco_idx[subset] = coco_load_index(cj)
        else:
            coco_idx[subset] = ({}, {})

    # Split placement audit CSV
    split_csv = out_dir / f"verify_{tag}_split_placement.csv"
    ensure_dir(out_dir)
    split_f = split_csv.open("w", newline="", encoding="utf-8")
    sw = csv.writer(split_f)
    sw.writerow(["file","case","current_subset","expected_subset","status"]) 

    for subset in SUBSETS:
        img_dir = variant_root / "images" / subset
        lbl_dir = variant_root / "labels" / subset
        if not img_dir.is_dir():
            print(f"[WARN] no images dir for {subset} at {img_dir}")
            continue
        # Logs
        yolo_csv = out_dir / f"verify_{tag}_{subset}_yolo.csv"
        coco_csv = out_dir / f"verify_{tag}_{subset}_coco.csv"
        ensure_dir(out_dir)
        yolo_f = yolo_csv.open("w", newline="", encoding="utf-8")
        coco_f = coco_csv.open("w", newline="", encoding="utf-8")
        yw = csv.writer(yolo_f) ; cw = csv.writer(coco_f)
        yw.writerow(["file","expected_boxes","matched","status","detail_iou"])
        cw.writerow(["file","expected_boxes","matched","status","detail_iou"])

        n_imgs = n_ok_yolo = n_ok_coco = 0
        fn2id, ann_map = coco_idx.get(subset, ({}, {}))

        for img_path in sorted(img_dir.rglob("*.jpg")):
            n_imgs += 1
            stem = img_path.stem  # SSS-VVV_FFFFF
            case, xml_stem = stem_to_case_and_xml_stem(stem)

            # Split placement check
            expected_subset = expected_subset_for_case(case)
            status_split = "OK" if expected_subset == subset else "WRONG_SPLIT"
            sw.writerow([img_path.name, case, subset, expected_subset, status_split])

            xml_path = orig_root / "annotations" / f"{case}_annotations" / f"{xml_stem}.xml"
            if not xml_path.exists():
                yw.writerow([img_path.name, 0, 0, "missing_xml", "-"])
                cw.writerow([img_path.name, 0, 0, "missing_xml", "-"])
                continue
            Wo, Ho, voc_boxes = parse_voc_xml(xml_path)

            with Image.open(img_path) as im:
                Wv, Hv = im.size

            # Expected target boxes
            if is_square:
                exp_boxes = [expected_box_xyxy_from_letterbox(b, Wo, Ho, Wv, Hv) for b in voc_boxes]
            else:
                exp_boxes = list(voc_boxes)

            # YOLO compare
            yolo_txt = lbl_dir / f"{stem}.txt"
            yolo_boxes = yolo_txt_to_xyxy(yolo_txt, Wv, Hv)
            n_exp, n_match, mism = compare_boxes(exp_boxes, yolo_boxes)
            status = "OK" if n_exp==n_match else ("MISMATCH" if yolo_boxes else ("NEGATIVE_OK" if n_exp==0 else "MISSING_YOLO"))
            best_iou = min([m[3] for m in mism], default=1.0)
            if status=="OK":
                n_ok_yolo += 1
            yw.writerow([img_path.name, n_exp, n_match, status, f"{best_iou:.6f}"])

            # COCO compare
            iid = fn2id.get(img_path.name, fn2id.get(f"images/{subset}/{img_path.name}", None))
            coco_boxes_xyxy = []
            if iid is not None:
                for (x,y,w,h) in ann_map.get(iid, []):
                    coco_boxes_xyxy.append((x, y, x+w, y+h))
            n_exp_c, n_match_c, mism_c = compare_boxes(exp_boxes, coco_boxes_xyxy)
            status_c = "OK" if n_exp_c==n_match_c else ("MISMATCH" if coco_boxes_xyxy else ("NEGATIVE_OK" if n_exp_c==0 else "MISSING_COCO"))
            best_iou_c = min([m[3] for m in mism_c], default=1.0)
            if status_c=="OK":
                n_ok_coco += 1
            cw.writerow([img_path.name, n_exp_c, n_match_c, status_c, f"{best_iou_c:.6f}"])

        yolo_f.close(); coco_f.close()
        print(f"  [{tag}:{subset}] images={n_imgs}  YOLO_OK={n_ok_yolo}  COCO_OK={n_ok_coco}  -> logs: {yolo_csv.name}, {coco_csv.name}")


def main():
    ap = argparse.ArgumentParser(description="Verify REAL-Colon processed labels against original VOC.")
    ap.add_argument("--orig-root", type=Path, required=True, help="/data/local/aschwab/data/realColon")
    ap.add_argument("--variant", action="append", required=True,
                    help="Variant spec ROOT:TAG:IS_SQUARE  (IS_SQUARE 0/1)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write CSV logs (default: <orig-root>/meta/verify)")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.orig_root / "meta" / "verify")
    for spec in args.variant:
        try:
            root_str, tag, is_sq = spec.split(":")
            var_root = Path(root_str)
            is_square = bool(int(is_sq))
        except Exception:
            raise SystemExit(f"Bad --variant spec: {spec}  (expected ROOT:TAG:IS_SQUARE)")
        verify_variant(args.orig_root, var_root, tag, is_square, out_dir)

    print("\n✅ Verification finished. Inspect CSVs under:", out_dir)


if __name__ == "__main__":
    main()
