#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("/home/stud/aschwab/master-thesis/code/pred_cache.db")

GT_JSON = Path("/data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/test_ann2.json")
IMAGES_DIR = Path("/data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/images/test2")

OUT_DIR = Path("viz_random_samples")

MODELS = ["RTDETR", "YOLOv11", "YOLOv8", "FasterRCNN"]

CONF_THR = 0.2
TOPK_DRAW = 2          # draw up to N predictions
TOPK_MATCH = 50        # use up to N predictions for TP/FP checks (fast enough)

# IoU threshold for calling a prediction "true detection" vs "false positive"
IOU_THR_TRUE = 0.1

# Manual selection: paste Ultralytics-style key (or None)
# examples: "001-013_010005.jpg" or "001-013_10005"
MANUAL_FRAME_KEY = None

# Sampling pool: "any", "pos", "neg"
SAMPLE_GT = "any"

# Constraints (apply only when MANUAL_FRAME_KEY is None)
REQUIRE_PRED_MODELS: List[str] = ["YOLOv11"]   # models that must have >=1 pred
FORBID_PRED_MODELS:  List[str] = []   # models that must have 0 pred

REQUIRE_TP_MODELS:   List[str] = []   # models that must have >=1 TP (needs GT)
FORBID_TP_MODELS:    List[str] = []   # models that must have 0 TP

REQUIRE_FP_MODELS:   List[str] = []   # models that must have >=1 FP
FORBID_FP_MODELS:    List[str] = []   # models that must have 0 FP

MIN_TOTAL_TP = None   # e.g. 1
MIN_TOTAL_FP = None   # e.g. 2

MAX_TRIES = 200000
RANDOM_SEED = None

# Text sizes (no header anymore)
FONT_LABEL = 18
FONT_SCORE = 28

# ============================================================


def canonical_realcolon_stem_id(name: str) -> str:
    stem = Path(name).stem
    if "_" not in stem:
        raise ValueError(f"Cannot canonicalize frame key (missing '_'): {name}")
    p, s = stem.rsplit("_", 1)
    return f"{p}_{int(s)}"


def xywh_to_xyxy(b):
    x, y, w, h = b
    return x, y, x + w, y + h


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def get_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ------------------------------------------------------------
# Load GT + mappings
# ------------------------------------------------------------

print("[INFO] loading GT...")

with GT_JSON.open("r") as f:
    coco = json.load(f)

# image_id (string) -> canonical frame_id
imgid_to_canonical: Dict[str, str] = {
    str(img["id"]): canonical_realcolon_stem_id(img["file_name"])
    for img in coco["images"]
}

# canonical frame_id -> image filename
frame_to_file: Dict[str, str] = {
    canonical_realcolon_stem_id(img["file_name"]): Path(img["file_name"]).name
    for img in coco["images"]
}

# canonical frame_id -> raw COCO image_id (string) (needed for FasterRCNN DB)
canonical_to_imgid: Dict[str, str] = {}
for imgid_str, fid in imgid_to_canonical.items():
    canonical_to_imgid.setdefault(fid, imgid_str)

# canonical frame_id -> GT boxes
frame_to_gt_boxes: Dict[str, List[List[float]]] = defaultdict(list)
for ann in coco["annotations"]:
    fid = imgid_to_canonical.get(str(ann["image_id"]))
    if fid is not None:
        frame_to_gt_boxes[fid].append(ann["bbox"])

# has GT?
frame_has_gt: Dict[str, bool] = {fid: (len(frame_to_gt_boxes.get(fid, [])) > 0) for fid in frame_to_file}

all_frames = list(frame_to_file.keys())
pos_frames = [f for f in all_frames if frame_has_gt[f]]
neg_frames = [f for f in all_frames if not frame_has_gt[f]]

print(f"[INFO] frames total={len(all_frames)}  GT+={len(pos_frames)}  GT-={len(neg_frames)}")


# ------------------------------------------------------------
# DB helpers (on-the-fly mapping)
# ------------------------------------------------------------

def get_preds(cur, model: str, canonical_frame_id: str, limit: int) -> List[Tuple[float, float, float, float, float]]:
    if model == "FasterRCNN":
        raw_id = canonical_to_imgid.get(canonical_frame_id)
        if raw_id is None:
            return []
        cur.execute(
            """
            SELECT x, y, w, h, score
            FROM preds
            WHERE model = ?
              AND frame_id = ?
              AND score >= ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (model, raw_id, CONF_THR, limit),
        )
        return cur.fetchall()

    cur.execute(
        """
        SELECT x, y, w, h, score
        FROM preds
        WHERE model = ?
          AND frame_id = ?
          AND score >= ?
        ORDER BY score DESC
        LIMIT ?
        """,
        (model, canonical_frame_id, CONF_THR, limit),
    )
    return cur.fetchall()


def analyze_model(preds_xywh: List[List[float]], gt_xywh: List[List[float]]) -> Tuple[bool, bool, bool, int, int]:
    """
    Returns:
      has_pred, has_tp, has_fp, n_tp, n_fp
    TP if matches any GT at IoU>=IOU_THR_TRUE, FP otherwise.
    """
    if not preds_xywh:
        return False, False, False, 0, 0

    if not gt_xywh:
        # all preds are FP on GT-negative frames
        return True, False, True, 0, len(preds_xywh)

    n_tp = 0
    n_fp = 0
    for p in preds_xywh:
        ok = any(iou_xywh(p, g) >= IOU_THR_TRUE for g in gt_xywh)
        if ok:
            n_tp += 1
        else:
            n_fp += 1
    return True, (n_tp > 0), (n_fp > 0), n_tp, n_fp


# ------------------------------------------------------------
# Rendering (no header; always draw GT if present)
# ------------------------------------------------------------

def render(img, gt_boxes, preds, model_name):
    im = img.copy()
    d = ImageDraw.Draw(im)

    font_label = get_font(FONT_LABEL)
    font_score = get_font(FONT_SCORE)

    # GT (green)
    for b in gt_boxes:
        x1, y1, x2, y2 = xywh_to_xyxy(b)
        d.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        d.text((x1 + 3, y1 + 3), "GT", fill="lime", font=font_label)

    # Predictions (red)
    for x, y, w, h, s in preds[:TOPK_DRAW]:
        x1, y1, x2, y2 = xywh_to_xyxy([x, y, w, h])
        d.rectangle([x1, y1, x2, y2], outline="red", width=3)
        d.text((x1 + 3, max(0, y1 - (FONT_SCORE + 4))), f"{s:.2f}", fill="red", font=font_score)

    return im


# ------------------------------------------------------------
# Frame selection with constraints (if MANUAL_FRAME_KEY is None)
# ------------------------------------------------------------

def candidate_list():
    if SAMPLE_GT == "pos":
        return pos_frames
    if SAMPLE_GT == "neg":
        return neg_frames
    return all_frames


def pick_frame_with_constraints(cur) -> str:
    cand = candidate_list()
    if not cand:
        raise RuntimeError("No frames available after SAMPLE_GT filtering.")

    for t in range(1, MAX_TRIES + 1):
        fid = random.choice(cand)
        gt_boxes = frame_to_gt_boxes.get(fid, [])

        per_model = {}
        total_tp = 0
        total_fp = 0

        for m in MODELS:
            preds = get_preds(cur, m, fid, TOPK_MATCH)
            preds_xywh = [list(p[:4]) for p in preds]
            has_pred, has_tp, has_fp, n_tp, n_fp = analyze_model(preds_xywh, gt_boxes)
            per_model[m] = (has_pred, has_tp, has_fp, n_tp, n_fp)
            total_tp += n_tp
            total_fp += n_fp

        # require/forbid any prediction
        if REQUIRE_PRED_MODELS and not all(per_model[m][0] for m in REQUIRE_PRED_MODELS):
            continue
        if FORBID_PRED_MODELS and any(per_model[m][0] for m in FORBID_PRED_MODELS):
            continue

        # require/forbid TP
        if REQUIRE_TP_MODELS and not all(per_model[m][1] for m in REQUIRE_TP_MODELS):
            continue
        if FORBID_TP_MODELS and any(per_model[m][1] for m in FORBID_TP_MODELS):
            continue

        # require/forbid FP
        if REQUIRE_FP_MODELS and not all(per_model[m][2] for m in REQUIRE_FP_MODELS):
            continue
        if FORBID_FP_MODELS and any(per_model[m][2] for m in FORBID_FP_MODELS):
            continue

        if MIN_TOTAL_TP is not None and total_tp < MIN_TOTAL_TP:
            continue
        if MIN_TOTAL_FP is not None and total_fp < MIN_TOTAL_FP:
            continue

        if t % 2000 == 0:
            print(f"[INFO] tried {t} frames...")

        return fid

    raise RuntimeError("No frame found that satisfies the constraints.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    OUT_DIR.mkdir(exist_ok=True)
    run_id = len(list(OUT_DIR.iterdir())) + 1
    run_dir = OUT_DIR / f"{run_id:04d}"
    run_dir.mkdir()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if MANUAL_FRAME_KEY is not None:
        fid = canonical_realcolon_stem_id(MANUAL_FRAME_KEY)
        if fid not in frame_to_file:
            conn.close()
            raise ValueError(f"Manual frame '{MANUAL_FRAME_KEY}' -> '{fid}' not found in GT images.")
        print(f"[INFO] manual frame: {fid}  GT={int(frame_has_gt[fid])}")
    else:
        fid = pick_frame_with_constraints(cur)
        print(f"[INFO] selected frame: {fid}  GT={int(frame_has_gt[fid])}")

    img = Image.open(IMAGES_DIR / frame_to_file[fid]).convert("RGB")
    gt_boxes = frame_to_gt_boxes.get(fid, [])

    for model in MODELS:
        preds = get_preds(cur, model, fid, TOPK_DRAW)  # only need TOPK_DRAW for visualization
        out = render(img, gt_boxes, preds, model)
        out.save(run_dir / f"{model}_{fid}.png")

    conn.close()
    print(f"[DONE] saved to {run_dir}")


if __name__ == "__main__":
    main()
