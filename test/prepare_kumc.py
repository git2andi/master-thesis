#!/usr/bin/env python3
"""
Build KUMC 'kumc_clean' (one-class 'lesion') on eridanus.

- Source:  /data/local/aschwab/data/kumc/PolypsSet/{train2019,val2019,test2019}
- Output:  /data/local/aschwab/data/kumc/kumc_clean
- Mapping rules:
    train2019 (FLAT) →   Annotation/<id>.xml  ↔  Image/<id>.<ext>
    val/test   (NESTED) → Annotation/<seq>/<n>.xml ↔ Image/<seq>/<n>.<ext>

- No resizing on disk; boxes are scaled from XML <size> to ACTUAL image size (they match but we stay robust).
- Images are symlinked by default to save space (safe for training). Flip COPY_IMAGES=True to hard-copy instead.

Author: you :)
"""

from __future__ import annotations
import os
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Tuple, Dict, List
import xml.etree.ElementTree as ET

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Pillow (PIL) is required. Try: pip install pillow") from e


# ---------- CONFIG (eridanus paths) ----------
SRC_ROOT  = Path("/data/local/aschwab/data/kumc/PolypsSet")
OUT_ROOT  = Path("/data/local/aschwab/data/kumc/kumc_clean")
ONE_CLASS = "lesion"
COPY_IMAGES = False          # False = symlink (recommended locally); True = copy files
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------- utils ----------
def safe_link_or_copy(src: Path, dst: Path, copy: bool = COPY_IMAGES) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.as_posix(), dst.as_posix())
        except OSError:
            shutil.copy2(src, dst)  # fallback if FS disallows symlinks


def parse_xml(xf: Path) -> Tuple[int, int, List[Tuple[float, float, float, float]]]:
    """Return (W,H, boxes) from an XML. Boxes are (xmin,ymin,xmax,ymax). Class is ignored → lesion."""
    r = ET.parse(xf).getroot()
    W = r.findtext("size/width") or "0"
    H = r.findtext("size/height") or "0"
    try:
        W = int(W); H = int(H)
    except Exception:
        W = 0; H = 0
    boxes = []
    for o in r.findall("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float((bb.findtext("xmin") or "0").strip())
            y1 = float((bb.findtext("ymin") or "0").strip())
            x2 = float((bb.findtext("xmax") or "0").strip())
            y2 = float((bb.findtext("ymax") or "0").strip())
        except Exception:
            continue
        boxes.append((x1, y1, x2, y2))
    return W, H, boxes


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    x1 = max(0.0, min(float(w), x1)); x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1)); y2 = max(0.0, min(float(h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def yolo_line_from_abs(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    # Clamp to [0,1]
    cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0); bh = min(max(bh, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"  # class 0 = lesion


def find_img_by_stem(folder: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_coco(out_json: Path, images: List[dict], annots: List[dict]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annots,
        "categories": [{"id": 1, "name": ONE_CLASS}],
    }
    out_json.write_text(json.dumps(coco))


def write_yaml(out_yaml: Path) -> None:
    content = f"""# KUMC PolypsSet (clean, one-class '{ONE_CLASS}')
path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test
nc: 1
names: [{ONE_CLASS}]
"""
    out_yaml.write_text(content)


def natural_key(name: str):
    # sort numeric stems numerically, strings lexically
    return (int(name), "") if name.isdigit() else (10**12, name)


# ---------- builders ----------
def build_train_train2019() -> Tuple[int, int, int]:
    split = "train2019"
    img_dir = SRC_ROOT / split / "Image"
    ann_dir = SRC_ROOT / split / "Annotation"

    out_img_dir = OUT_ROOT / "images" / "train"
    out_lbl_dir = OUT_ROOT / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images_by_stem: Dict[str, Path] = {
        p.stem: p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    }

    coco_images: List[dict] = []
    coco_annots: List[dict] = []
    ann_id = 1
    n_pos = 0
    n_neg = 0

    # manifest
    man_path = OUT_ROOT / "manifests" / "train_pairs.csv"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_rows = [["key", "src_image", "src_xml", "status", "num_boxes"]]

    for stem in sorted(images_by_stem.keys(), key=natural_key):
        src_img = images_by_stem[stem]
        src_xml = ann_dir / f"{stem}.xml"
        dst_img = out_img_dir / src_img.name
        dst_lbl = out_lbl_dir / f"{stem}.txt"

        safe_link_or_copy(src_img, dst_img)
        with Image.open(src_img) as im:
            iw, ih = im.size

        boxes_out: List[Tuple[float, float, float, float]] = []
        if src_xml.exists():
            W, H, boxes = parse_xml(src_xml)
            sx = iw / float(W) if W else 1.0
            sy = ih / float(H) if H else 1.0
            seen = set()
            for x1, y1, x2, y2 in boxes:
                X1, Y1, X2, Y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                c = clip_box(X1, Y1, X2, Y2, iw, ih)
                if not c:
                    continue
                # dedup by coarse rounding to avoid exact duplicates
                k = (round(c[0], 1), round(c[1], 1), round(c[2], 1), round(c[3], 1))
                if k in seen:
                    continue
                seen.add(k)
                boxes_out.append(c)

        # YOLO label
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if boxes_out:
            with open(dst_lbl, "w") as f:
                for X1, Y1, X2, Y2 in boxes_out:
                    f.write(yolo_line_from_abs(X1, Y1, X2, Y2, iw, ih) + "\n")
            n_pos += 1
        else:
            dst_lbl.touch()
            n_neg += 1

        # COCO
        img_id = len(coco_images) + 1
        coco_images.append({
            "id": img_id,
            "file_name": dst_img.relative_to(out_img_dir).as_posix(),
            "width": iw, "height": ih,
        })
        for X1, Y1, X2, Y2 in boxes_out:
            x, y, w, h = float(X1), float(Y1), float(X2 - X1), float(Y2 - Y1)
            coco_annots.append({
                "id": ann_id, "image_id": img_id,
                "category_id": 1, "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0
            })
            ann_id += 1

        man_rows.append([stem, src_img.as_posix(), src_xml.as_posix() if src_xml.exists() else "", "OK", len(boxes_out)])

    write_coco(OUT_ROOT / "coco" / "instances_train.json", coco_images, coco_annots)
    with open(man_path, "w", newline="") as f:
        csv.writer(f).writerows(man_rows)

    print(f"[train] images={len(coco_images):,}  anns={len(coco_annots):,}  pos(yolo files non-empty)={n_pos:,}  neg={n_neg:,}")
    return len(coco_images), len(coco_annots), n_pos


def build_seq_split(split: str, outname: str) -> Tuple[int, int, int]:
    img_root = SRC_ROOT / split / "Image"
    ann_root = SRC_ROOT / split / "Annotation"

    out_img_dir = OUT_ROOT / "images" / outname
    out_lbl_dir = OUT_ROOT / "labels" / outname
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    coco_images: List[dict] = []
    coco_annots: List[dict] = []
    ann_id = 1
    n_pos = 0
    n_neg = 0

    man_path = OUT_ROOT / "manifests" / f"{outname}_pairs.csv"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_rows = [["key", "src_image", "src_xml", "status", "num_boxes"]]

    seq_dirs = sorted([d for d in img_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    for seq_dir in seq_dirs:
        seq = seq_dir.name
        ann_seq = ann_root / seq
        imgs = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
                      key=lambda p: (natural_key(p.stem), p.suffix))
        for src_img in imgs:
            stem = src_img.stem
            src_xml = ann_seq / f"{stem}.xml"

            rel_img = Path(seq) / src_img.name
            dst_img = out_img_dir / rel_img
            dst_lbl = out_lbl_dir / Path(seq) / f"{stem}.txt"

            safe_link_or_copy(src_img, dst_img)
            with Image.open(src_img) as im:
                iw, ih = im.size

            boxes_out: List[Tuple[float, float, float, float]] = []
            if src_xml.exists():
                W, H, boxes = parse_xml(src_xml)
                sx = iw / float(W) if W else 1.0
                sy = ih / float(H) if H else 1.0
                seen = set()
                for x1, y1, x2, y2 in boxes:
                    X1, Y1, X2, Y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                    c = clip_box(X1, Y1, X2, Y2, iw, ih)
                    if not c:
                        continue
                    k = (round(c[0], 1), round(c[1], 1), round(c[2], 1), round(c[3], 1))
                    if k in seen:
                        continue
                    seen.add(k)
                    boxes_out.append(c)

            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if boxes_out:
                with open(dst_lbl, "w") as f:
                    for X1, Y1, X2, Y2 in boxes_out:
                        f.write(yolo_line_from_abs(X1, Y1, X2, Y2, iw, ih) + "\n")
                n_pos += 1
            else:
                dst_lbl.touch()
                n_neg += 1

            img_id = len(coco_images) + 1
            coco_images.append({
                "id": img_id, "file_name": rel_img.as_posix(), "width": iw, "height": ih
            })
            for X1, Y1, X2, Y2 in boxes_out:
                x, y, w, h = float(X1), float(Y1), float(X2 - X1), float(Y2 - Y1)
                coco_annots.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2), "iscrowd": 0
                })
                ann_id += 1

            man_rows.append([f"{seq}/{stem}", src_img.as_posix(), src_xml.as_posix() if src_xml.exists() else "",
                             "OK", len(boxes_out)])

    write_coco(OUT_ROOT / "coco" / f"instances_{outname}.json", coco_images, coco_annots)
    with open(man_path, "w", newline="") as f:
        csv.writer(f).writerows(man_rows)

    print(f"[{outname}] images={len(coco_images):,}  anns={len(coco_annots):,}  pos={n_pos:,}  neg={n_neg:,}")
    return len(coco_images), len(coco_annots), n_pos


def main():
    # Create top-level dirs
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "coco").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)

    # Build splits
    t_imgs, t_anns, t_pos = build_train_train2019()
    v_imgs, v_anns, v_pos = build_seq_split("val2019", "val")
    s_imgs, s_anns, s_pos = build_seq_split("test2019", "test")

    # data.yaml
    write_yaml(OUT_ROOT / "data.yaml")

    # Summary
    print("\n=== SUMMARY (kumc_clean) ===")
    print(f"train: images={t_imgs:,}  anns={t_anns:,}  yolo_pos_files={t_pos:,}")
    print(f"val  : images={v_imgs:,}  anns={v_anns:,}  yolo_pos_files={v_pos:,}")
    print(f"test : images={s_imgs:,}  anns={s_anns:,}  yolo_pos_files={s_pos:,}")
    print(f"\nWrote:\n- {OUT_ROOT/'data.yaml'}\n- {OUT_ROOT/'coco'/'instances_train.json'}\n- {OUT_ROOT/'coco'/'instances_val.json'}\n- {OUT_ROOT/'coco'/'instances_test.json'}\n- images -> {OUT_ROOT/'images'}\n- labels -> {OUT_ROOT/'labels'}\n- manifests -> {OUT_ROOT/'manifests'}")


if __name__ == "__main__":
    main()
