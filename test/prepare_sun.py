#!/usr/bin/env python3
"""
Prepare SUN dataset into:
  1) sun_test/  (full test-only package)
  2) sun_split/ (random 70/15/15 case-wise split)

Both include:
  - images/{...}/{pos,neg}/caseXX/*.jpg  (symlinks; falls back to copy)
  - labels/{...}/{pos,neg}/caseXX/*.txt  (YOLO; single class 'lesion', id 0)
  - instances_{train,val,test}.json (COCO; single class 'lesion', id 0)
  - data.yaml
  - sun_case_split_map.csv (for sun_split)

Usage:
  python prepare_sun.py --sun-root /data/local/aschwab/data/sun --seed 42
Options:
  --no-test / --no-split : skip building one of the outputs
  --test-out / --split-out: override output dirs (default are under sun-root)
  --copy                 : copy files instead of symlinking
  --class-name           : default 'lesion'
  --yolo-class-id        : default 0
  --coco-class-id        : default 0
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, random
from pathlib import Path

# Pillow just for image sizes (fast: reads headers)
try:
    from PIL import Image, ImageFile
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pillow"])
    from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly truncated JPEGs

# ---------------------------- utils ----------------------------
def ensure_link(src: Path, dst: Path, copy_fallback: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        if copy_fallback:
            shutil.copy2(src, dst)
        else:
            raise

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def img_size(p: Path, cache: dict[Path, tuple[int,int]]) -> tuple[int,int]:
    if p in cache:
        return cache[p]
    with Image.open(p) as im:
        cache[p] = im.size
    return cache[p]

def write_yaml(path: Path, dataset_root: Path, class_name: str, split_paths: dict[str,str]):
    text = [
        f"# auto-generated: SUN ({class_name})",
        f"path: {dataset_root.as_posix()}",
        f"train: {split_paths['train']}",
        f"val: {split_paths['val']}",
        f"test: {split_paths['test']}",
        "names:",
        f"  0: {class_name}",
        "",
    ]
    path.write_text("\n".join(text))

def coco_dump(path: Path, images, annotations, class_id: int, class_name: str, description: str):
    j = {
        "info": {"description": description},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": class_id, "name": class_name}],
    }
    path.write_text(json.dumps(j))

# ---------------------------- indexing & parsing ----------------------------
def discover_sun(sun_root: Path):
    """Return (pos_case_to_dir, neg_case_to_dir, ann_dir)"""
    pos_case_to_dir = {}
    for c in range(1, 101):
        name = f"case{c}"
        hits = list(sun_root.glob(f"sundatabase_positive_part*/{name}"))
        if hits:
            pos_case_to_dir[name] = hits[0]
    ann_dir = sun_root / "sundatabase_positive_part1" / "annotation_txt"
    neg_case_to_dir = {}
    for d in sorted(sun_root.glob("sundatabase_negative_part*/case*")):
        neg_case_to_dir[d.name] = d
    if not ann_dir.exists():
        raise SystemExit(f"Annotation dir not found: {ann_dir}")
    return pos_case_to_dir, neg_case_to_dir, ann_dir

def parse_annotations(ann_dir: Path, pos_case_to_dir: dict[str, Path]):
    """Return (pos_case_to_imgs, img2boxes) using SIGNED coords, merged across lines."""
    line_pat = re.compile(r"^(\S+\.jpg)\s+(.*)$", re.IGNORECASE)
    box_pat  = re.compile(r"(-?\d+),(-?\d+),(-?\d+),(-?\d+),\d+")
    pos_case_to_imgs: dict[str, list[Path]] = {}
    img2boxes: dict[Path, list[tuple[int,int,int,int]]] = {}
    missing = 0

    for ann in sorted(ann_dir.glob("case*.txt"), key=lambda p: int(p.stem[4:])):
        caseN = ann.stem
        imgdir = pos_case_to_dir.get(caseN)
        if not imgdir:
            continue
        imgs = []
        with ann.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                m = line_pat.match(raw.strip())
                if not m:
                    continue
                fname, rest = m.group(1), m.group(2)
                p = imgdir / fname
                if not p.exists():
                    missing += 1
                    continue
                imgs.append(p)
                boxes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in box_pat.findall(rest)]
                if boxes:
                    img2boxes.setdefault(p, []).extend(boxes)
        if imgs:
            # de-dup in case an image is listed multiple times
            pos_case_to_imgs[caseN] = sorted(set(imgs))
    return pos_case_to_imgs, img2boxes, missing

# ---------------------------- builders ----------------------------
def build_sun_test(
    sun_root: Path,
    out_root: Path,
    pos_case_to_imgs: dict[str, list[Path]],
    neg_case_to_dir: dict[str, Path],
    img2boxes: dict[Path, list[tuple[int,int,int,int]]],
    class_name: str,
    yolo_class_id: int,
    coco_class_id: int,
    copy_files: bool,
):
    """Builds test-only package at out_root (images/test, labels/test, instances_test.json, data.yaml)."""
    if out_root.exists():
        shutil.rmtree(out_root)
    img_root = out_root / "images" / "test"
    lbl_root = out_root / "labels" / "test"
    (img_root).mkdir(parents=True, exist_ok=True)
    (lbl_root / "pos").mkdir(parents=True, exist_ok=True)
    (lbl_root / "neg").mkdir(parents=True, exist_ok=True)

    size_cache = {}

    # Positives: link images and write YOLO labels; COCO records images+boxes
    images, annotations = [], []
    image_id, ann_id = 1, 1
    pos_imgs = 0
    for caseN in sorted(pos_case_to_imgs.keys(), key=lambda s: int(s[4:])):
        for src in pos_case_to_imgs[caseN]:
            ensure_link(src, img_root / "pos" / caseN / src.name, copy_fallback=copy_files)
            W, H = img_size(src, size_cache)
            # YOLO label
            lbl = lbl_root / "pos" / caseN / (src.stem + ".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for (x1, y1, x2, y2) in img2boxes.get(src, []):
                x1 = clamp(x1, 0, W - 1); x2 = clamp(x2, 0, W - 1)
                y1 = clamp(y1, 0, H - 1); y2 = clamp(y2, 0, H - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = ((x1 + x2) / 2) / W
                yc = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                lines.append(f"{yolo_class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": coco_class_id,
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0
                })
                ann_id += 1
            if lines:
                lbl.write_text("\n".join(lines))
            else:
                lbl.touch()  # rare
            images.append({
                "id": image_id,
                "file_name": f"pos/{caseN}/{src.name}",
                "width": W, "height": H
            })
            image_id += 1
            pos_imgs += 1

    # Negatives: link images and create empty YOLO labels; COCO has images only
    neg_imgs = 0
    for caseN, d in sorted(neg_case_to_dir.items(), key=lambda kv: int(kv[0][4:])):
        for src in sorted(d.glob("*.jpg")):
            ensure_link(src, img_root / "neg" / caseN / src.name, copy_fallback=copy_files)
            lbl = lbl_root / "neg" / caseN / (src.stem + ".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            if not lbl.exists():
                lbl.touch()
            W, H = img_size(src, size_cache)
            images.append({
                "id": image_id,
                "file_name": f"neg/{caseN}/{src.name}",
                "width": W, "height": H
            })
            image_id += 1
            neg_imgs += 1

    coco_dump(out_root / "instances_test.json", images, annotations, coco_class_id, class_name,
              "SUN test-only (single class)")

    write_yaml(out_root / "data.yaml", out_root, class_name,
               {"train": "images/test", "val": "images/test", "test": "images/test"})

    print(f"[sun_test] Pos images: {pos_imgs}  Neg images: {neg_imgs}  COCO anns: {len(annotations)}")

def build_sun_split(
    sun_root: Path,
    out_root: Path,
    pos_case_to_imgs: dict[str, list[Path]],
    neg_case_to_dir: dict[str, Path],
    img2boxes: dict[Path, list[tuple[int,int,int,int]]],
    class_name: str,
    yolo_class_id: int,
    coco_class_id: int,
    seed: int,
    copy_files: bool,
):
    """Builds sun_split/ with random 70/15/15 (pos: 70/15/15 cases; neg: 9/2/2 cases)."""
    if out_root.exists():
        shutil.rmtree(out_root)
    img_root = out_root / "images"
    lbl_root = out_root / "labels"
    for s in ("train", "val", "test"):
        (img_root / s / "pos").mkdir(parents=True, exist_ok=True)
        (img_root / s / "neg").mkdir(parents=True, exist_ok=True)
        (lbl_root / s / "pos").mkdir(parents=True, exist_ok=True)
        (lbl_root / s / "neg").mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Choose case IDs
    pos_cases = sorted(pos_case_to_imgs.keys(), key=lambda s: int(s[4:]))  # case1..case100
    train_pos = set(random.sample(pos_cases, 70))
    rem = [c for c in pos_cases if c not in train_pos]
    val_pos = set(random.sample(rem, 15))
    test_pos = set([c for c in pos_cases if c not in train_pos | val_pos])

    neg_cases = sorted(neg_case_to_dir.keys(), key=lambda s: int(s[4:]))  # case1..case13
    train_neg = set(random.sample(neg_cases, 9))
    remn = [c for c in neg_cases if c not in train_neg]
    val_neg = set(random.sample(remn, 2))
    test_neg = set([c for c in neg_cases if c not in train_neg | val_neg])

    splits = {
        "train": {"pos": train_pos, "neg": train_neg},
        "val":   {"pos": val_pos,   "neg": val_neg},
        "test":  {"pos": test_pos,  "neg": test_neg},
    }

    # Build contents
    size_cache = {}
    yolo_pos_written = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        # POS
        for caseN in sorted(splits[split]["pos"], key=lambda s: int(s[4:])):
            for src in pos_case_to_imgs[caseN]:
                ensure_link(src, img_root / split / "pos" / caseN / src.name, copy_fallback=copy_files)
                W, H = img_size(src, size_cache)
                lines = []
                for (x1, y1, x2, y2) in img2boxes.get(src, []):
                    x1 = clamp(x1, 0, W - 1); x2 = clamp(x2, 0, W - 1)
                    y1 = clamp(y1, 0, H - 1); y2 = clamp(y2, 0, H - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    xc = ((x1 + x2) / 2) / W
                    yc = ((y1 + y2) / 2) / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    lines.append(f"{yolo_class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                lbl = lbl_root / split / "pos" / caseN / (src.stem + ".txt")
                lbl.parent.mkdir(parents=True, exist_ok=True)
                if lines:
                    lbl.write_text("\n".join(lines))
                    yolo_pos_written[split] += 1
                else:
                    lbl.touch()

        # NEG
        for caseN in sorted(splits[split]["neg"], key=lambda s: int(s[4:])):
            for src in sorted(neg_case_to_dir[caseN].glob("*.jpg")):
                ensure_link(src, img_root / split / "neg" / caseN / src.name, copy_fallback=copy_files)
                lbl = lbl_root / split / "neg" / caseN / (src.stem + ".txt")
                lbl.parent.mkdir(parents=True, exist_ok=True)
                if not lbl.exists():
                    lbl.touch()

    # data.yaml
    write_yaml(out_root / "data.yaml", out_root, class_name,
               {"train": "images/train", "val": "images/val", "test": "images/test"})

    # COCO per split
    def build_coco_for(split: str):
        images, anns = [], []
        image_id, ann_id = 1, 1
        # POS
        for case_dir in sorted((img_root / split / "pos").glob("case*"), key=lambda p: int(p.name[4:])):
            caseN = case_dir.name
            for p in sorted(case_dir.glob("*.jpg")):
                # resolve src (to read original boxes)
                src = next((s for s in pos_case_to_imgs[caseN] if s.name == p.name), None)
                W, H = img_size(src or p, size_cache)
                images.append({"id": image_id,
                               "file_name": f"pos/{caseN}/{p.name}",
                               "width": W, "height": H})
                for (x1, y1, x2, y2) in img2boxes.get(src, []):
                    x1 = clamp(x1, 0, W - 1); x2 = clamp(x2, 0, W - 1)
                    y1 = clamp(y1, 0, H - 1); y2 = clamp(y2, 0, H - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    w, h = (x2 - x1), (y2 - y1)
                    anns.append({"id": ann_id, "image_id": image_id, "category_id": coco_class_id,
                                 "bbox": [int(x1), int(y1), int(w), int(h)],
                                 "area": float(w * h), "iscrowd": 0})
                    ann_id += 1
                image_id += 1
        # NEG
        for case_dir in sorted((img_root / split / "neg").glob("case*"), key=lambda p: int(p.name[4:])):
            caseN = case_dir.name
            for p in sorted(case_dir.glob("*.jpg")):
                src = next((s for s in neg_case_to_dir[caseN].glob("*.jpg") if s.name == p.name), None)
                W, H = img_size(src or p, size_cache)
                images.append({"id": image_id,
                               "file_name": f"neg/{caseN}/{p.name}",
                               "width": W, "height": H})
                image_id += 1
        return images, anns

    for split in ("train", "val", "test"):
        images, anns = build_coco_for(split)
        coco_dump(out_root / f"instances_{split}.json", images, anns, coco_class_id, class_name,
                  f"SUN {split} (random seed {seed})")

    # Write CSV manifest (case -> split)
    csv = ["type,case_id,split,images"]
    for typ in ("pos", "neg"):
        for split in ("train", "val", "test"):
            for d in sorted((img_root / split / typ).glob("case*"), key=lambda p: int(p.name[4:])):
                n = sum(1 for _ in d.rglob("*.jpg"))
                csv.append(f"{typ},{d.name},{split},{n}")
    (out_root / "sun_case_split_map.csv").write_text("\n".join(csv))

    # Summary
    def count_imgs(root): return sum(1 for _ in (root).rglob("*.jpg"))
    def count_lbls(root): return sum(1 for _ in (root).rglob("*.txt"))
    print("[sun_split] seed:", seed)
    for s in ("train", "val", "test"):
        print(f"  {s:5s} imgs: {count_imgs(img_root/s):6d} | lbls: {count_lbls(lbl_root/s):6d}")

# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-root", type=Path, required=True,
                    help="Path containing sundatabase_positive_part{1,2} and sundatabase_negative_part{1..4}")
    ap.add_argument("--test-out", type=Path, default=None, help="Output dir for sun_test/")
    ap.add_argument("--split-out", type=Path, default=None, help="Output dir for sun_split/")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SUN_SPLIT_SEED", 42)))
    ap.add_argument("--class-name", default="lesion")
    ap.add_argument("--yolo-class-id", type=int, default=0)
    ap.add_argument("--coco-class-id", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument("--no-test", action="store_true", help="Skip building sun_test/")
    ap.add_argument("--no-split", action="store_true", help="Skip building sun_split/")
    args = ap.parse_args()

    sun_root = args.sun_root.resolve()
    test_out = (args.test_out or (sun_root / "sun_test")).resolve()
    split_out = (args.split_out or (sun_root / "sun_split")).resolve()

    # Discover & parse
    pos_case_to_dir, neg_case_to_dir, ann_dir = discover_sun(sun_root)
    pos_case_to_imgs, img2boxes, missing = parse_annotations(ann_dir, pos_case_to_dir)
    if missing:
        print(f"[warn] {missing} referenced images not found on disk (skipped)")

    # Build
    if not args.no_test:
        build_sun_test(
            sun_root, test_out,
            pos_case_to_imgs, neg_case_to_dir, img2boxes,
            class_name=args.class_name,
            yolo_class_id=args.yolo_class_id,
            coco_class_id=args.coco_class_id,
            copy_files=args.copy,
        )
    if not args.no_split:
        build_sun_split(
            sun_root, split_out,
            pos_case_to_imgs, neg_case_to_dir, img2boxes,
            class_name=args.class_name,
            yolo_class_id=args.yolo_class_id,
            coco_class_id=args.coco_class_id,
            seed=args.seed,
            copy_files=args.copy,
        )

    print("Done.")

if __name__ == "__main__":
    main()
