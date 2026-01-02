import json
import random
from pathlib import Path
from glob import glob
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from PIL import Image, ImageDraw
from natsort import natsorted

BASE_ORIG = Path("/data/local/aschwab/data/sun") 
BASE_SPLIT = Path("/data/local/aschwab/data/sun_split")
BASE_FULL  = Path("/data/local/aschwab/data/sun_full")

NUM_SAMPLES = 5
RANDOM_SEED = 42


def collect_all_cases(base_dataset_folder: Path) -> Dict[int, Dict[str, Any]]:
    """
    Rebuild the mapping from new_case_id -> case_info as in the SUN script.
    """
    negative_cases: List[Dict[str, Any]] = []
    positive_cases: List[Dict[str, Any]] = []

    data_parts = glob(str(base_dataset_folder / "sundatabase_*_part*"))

    for part_path in data_parts:
        part_path = Path(part_path)
        part_name = part_path.name

        if "negative" in part_name:
            case_type = "negative"
        elif "positive" in part_name:
            case_type = "positive"
        else:
            continue

        case_subfolders = [
            d.name
            for d in part_path.iterdir()
            if d.is_dir() and d.name != "annotation_txt"
        ]
        case_names = natsorted(case_subfolders)

        for case_name in case_names:
            case_info: Dict[str, Any] = {
                "case_name": case_name,
                "data_folder": part_name,  # e.g. "sundatabase_positive_part1"
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

    id2case: Dict[int, Dict[str, Any]] = {}
    for c in positive_cases + negative_cases:
        id2case[c["new_case_id"]] = c

    return id2case


def parse_sun_case_annotations(annotation_root: Path, case_name: str) -> Dict[str, List[Tuple[int,int,int,int]]]:
    """
    Original SUN parser: one txt per case in annotation_txt, lines like:
      <filename> x1,y1,x2,y2[,...] x1,y1,x2,y2[,...] ...
    Returns: dict[filename] = list of (x1,y1,x2,y2).
    """
    ann_path = annotation_root / f"{case_name}.txt"
    frame_annotations: Dict[str, List[Tuple[int,int,int,int]]] = defaultdict(list)
    if not ann_path.exists():
        return {}

    with ann_path.open("r") as f:
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
                frame_annotations[filename].append((x1, y1, x2, y2))
    return dict(frame_annotations)


def draw_boxes_on_image(img: Image.Image, boxes, color="red", width=3):
    """
    Draw list of boxes [(x1,y1,x2,y2), ...] on a copy of img.
    Returns a new Image.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def yolo_to_abs_boxes(label_path: Path, img_width: int, img_height: int):
    """
    Convert YOLO labels (cls cx cy w h, normalized) to absolute boxes.
    Returns list of (x1, y1, x2, y2).
    """
    if not label_path.exists():
        return []

    text = label_path.read_text().strip()
    if not text:
        return []

    boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # cls = int(parts[0])
        _, cx, cy, w, h = parts[:5]
        cx = float(cx) * img_width
        cy = float(cy) * img_height
        w = float(w) * img_width
        h = float(h) * img_height
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes.append((x1, y1, x2, y2))
    return boxes


# -----------------------------
# Helpers for derived SUN COCO
# -----------------------------

def load_full_coco_by_filename() -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Load sun_full COCO and index annotations by file_name.
    Returns dict[file_name] -> list of (x1,y1,x2,y2).
    """
    coco_full_path = BASE_FULL / "coco_annotations_test.json"
    if not coco_full_path.exists():
        print(f"WARNING: {coco_full_path} not found; full COCO boxes will be empty.")
        return {}

    with coco_full_path.open("r") as f:
        coco = json.load(f)

    anns = coco.get("annotations", [])
    imgs = coco.get("images", [])

    # map image_id -> file_name
    id2fname = {img["id"]: img["file_name"] for img in imgs}

    boxes_by_fname: Dict[str, List[Tuple[float,float,float,float]]] = defaultdict(list)
    for ann in anns:
        img_id = ann["image_id"]
        fname = id2fname.get(img_id)
        if fname is None:
            continue
        x, y, w, h = ann["bbox"]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes_by_fname[fname].append((x1, y1, x2, y2))

    return boxes_by_fname


def collect_polyp_candidates_split():
    """
    Collect all images in sun_split that have >=1 annotation in COCO
    (over train/val/test splits).
    Returns list of tuples: (split_name, img_info, img_anns).
    """
    candidates = []
    split_defs = [
        ("train", "coco_annotations_train.json"),
        ("val",   "coco_annotations_val.json"),
        ("test",  "coco_annotations_test.json"),
    ]

    for split_name, coco_file in split_defs:
        coco_path = BASE_SPLIT / coco_file
        if not coco_path.exists():
            continue

        with coco_path.open("r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        anns = coco.get("annotations", [])

        anns_by_img = defaultdict(list)
        for ann in anns:
            anns_by_img[ann["image_id"]].append(ann)

        for img_info in images:
            img_id = img_info["id"]
            img_anns = anns_by_img.get(img_id, [])
            if len(img_anns) > 0:
                candidates.append((split_name, img_info, img_anns))

    return candidates


def main():
    # Build case-id mapping for original SUN
    id2case = collect_all_cases(BASE_ORIG)
    print(f"Rebuilt mapping for {len(id2case)} cases in original SUN.")

    # Polyp frames from sun_split
    candidates = collect_polyp_candidates_split()
    print(f"Found {len(candidates)} polyp frames in sun_split.")

    if len(candidates) < NUM_SAMPLES:
        raise RuntimeError("Not enough polyp frames to sample from.")

    random.seed(RANDOM_SEED)
    samples = random.sample(candidates, NUM_SAMPLES)

    # Load COCO for sun_full, indexed by file_name
    full_boxes_by_fname = load_full_coco_by_filename()

    # Output folder: verify_sun/ under current working directory
    out_dir = Path.cwd() / "verify_sun"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualization images to: {out_dir}")

    # Positive annotations for SUN: all in positive part's annotation_txt
    ann_root = BASE_ORIG / "sundatabase_positive_part1" / "annotation_txt"

    for idx, (split_name, img_info, img_anns) in enumerate(samples, start=1):
        fname = img_info["file_name"]  # e.g. "case12_004-001_000123.jpg"
        stem = Path(fname).stem

        # Parse "case<id>_<orig_name>"
        try:
            prefix, orig_name = fname.split("_", 1)  # keep extension in orig_name
        except ValueError:
            print(f"Skipping unexpected filename format: {fname}")
            continue

        if not prefix.startswith("case"):
            print(f"Skipping unexpected prefix in filename: {fname}")
            continue

        new_case_id = int(prefix.replace("case", ""))
        case_info = id2case.get(new_case_id)
        if case_info is None:
            print(f"No case mapping found for {fname} (case_id={new_case_id})")
            continue

        case_name = case_info["case_name"]
        data_folder = case_info["data_folder"]  # e.g. "sundatabase_positive_part1"

        # Original image path
        orig_img_path = BASE_ORIG / data_folder / case_name / orig_name
        if not orig_img_path.exists():
            print(f"Original image missing: {orig_img_path}")
            continue

        # Original boxes
        orig_ann_by_frame = parse_sun_case_annotations(ann_root, case_name)
        orig_boxes = orig_ann_by_frame.get(orig_name, [])

        # sun_split paths
        split_img_path = BASE_SPLIT / "images" / split_name / fname
        split_lbl_path = BASE_SPLIT / "labels" / split_name / (Path(fname).with_suffix(".txt").name)

        if not split_img_path.exists():
            print(f"Split image missing: {split_img_path}")
            continue

        # sun_full paths
        full_img_path = BASE_FULL / "images" / "test" / fname
        full_lbl_path = BASE_FULL / "labels" / "test" / (Path(fname).with_suffix(".txt").name)

        # Load images
        orig_img = Image.open(orig_img_path).convert("RGB")
        split_img_for_coco = Image.open(split_img_path).convert("RGB")
        split_img_for_yolo = split_img_for_coco.copy()
        full_img_for_coco = split_img_for_coco.copy()  # same content
        full_img_for_yolo = split_img_for_coco.copy()

        w, h = split_img_for_coco.size

        # COCO boxes (sun_split)
        split_coco_boxes = []
        for ann in img_anns:
            x, y, bw, bh = ann["bbox"]
            x1 = x
            y1 = y
            x2 = x + bw
            y2 = y + bh
            split_coco_boxes.append((x1, y1, x2, y2))

        # YOLO boxes (sun_split)
        split_yolo_boxes = yolo_to_abs_boxes(split_lbl_path, w, h)

        # COCO boxes (sun_full)
        full_coco_boxes = full_boxes_by_fname.get(fname, [])

        # YOLO boxes (sun_full)
        full_yolo_boxes = yolo_to_abs_boxes(full_lbl_path, w, h)

        # Draw variants
        orig_vis = draw_boxes_on_image(orig_img, orig_boxes, color="red", width=3)
        split_coco_vis = draw_boxes_on_image(split_img_for_coco, split_coco_boxes, color="green", width=3)
        split_yolo_vis = draw_boxes_on_image(split_img_for_yolo, split_yolo_boxes, color="blue", width=3)
        full_coco_vis = draw_boxes_on_image(full_img_for_coco, full_coco_boxes, color="yellow", width=3)
        full_yolo_vis = draw_boxes_on_image(full_img_for_yolo, full_yolo_boxes, color="cyan", width=3)

        # Save
        base_tag = f"sun_sample{idx}_{case_name}_{Path(orig_name).stem}"

        out_orig       = out_dir / f"{base_tag}_orig.jpg"
        out_split_coco = out_dir / f"{base_tag}_split_coco.jpg"
        out_split_yolo = out_dir / f"{base_tag}_split_yolo.jpg"
        out_full_coco  = out_dir / f"{base_tag}_full_coco.jpg"
        out_full_yolo  = out_dir / f"{base_tag}_full_yolo.jpg"

        orig_vis.save(out_orig)
        split_coco_vis.save(out_split_coco)
        split_yolo_vis.save(out_split_yolo)
        full_coco_vis.save(out_full_coco)
        full_yolo_vis.save(out_full_yolo)

        print(f"Saved for {fname}:")
        print(f"  {out_orig.name}")
        print(f"  {out_split_coco.name}")
        print(f"  {out_split_yolo.name}")
        print(f"  {out_full_coco.name}")
        print(f"  {out_full_yolo.name}")

    print("Done. Check verify_sun/ for all visualization images.")


if __name__ == "__main__":
    main()

