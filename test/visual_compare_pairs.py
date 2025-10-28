#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visual_compare_pairs.py

Side-by-side visual comparison for REAL-Colon variants.
Overlays YOLO boxes on images from BASE and VARIANT datasets and writes PNG pairs.

Use cases:
- Verify that labels/boxes match between base and subset (i1..i4, 1_3).
- Inspect added negatives (neg{R}x) or duplicates (pos* variants).
- Do quick manual spot-checks for thesis appendix.

Examples:
  python visual_compare_pairs.py \
    --base /data/local/aschwab/data/realColon_640x640 \
    --variant /data/local/aschwab/data/realColon_640x640_i1 \
    --split train --n 24 --mode common_pos --out /tmp/vis_i1

  python visual_compare_pairs.py \
    --base /data/local/aschwab/data/realColon_640x640 \
    --variant /data/local/aschwab/data/realColon_640x640_neg3x \
    --split train --n 24 --mode variant_only_neg --out /tmp/vis_neg3x

  python visual_compare_pairs.py \
    --base /data/local/aschwab/data/realColon_640x640 \
    --variant /data/local/aschwab/data/realColon_640x640_pos3x \
    --split train --n 24 --mode variant_duplicates --out /tmp/vis_pos3x

Modes:
- common_pos: sample frames present in BOTH datasets with non-empty labels in the BASE set.
- common_any: sample any frames present in BOTH datasets (pos or neg).
- variant_only_neg: sample frames that are NEGATIVE in VARIANT but either absent or positive in BASE (good for negR× audits).
- variant_duplicates: sample *_posdup/*_evtdup/*_bindup frames from VARIANT and pair with their base originals.

Outputs:
- PNG files "<idx>__<stem>.png" (side-by-side BASE | VARIANT)
- index.html gallery for quick viewing
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import os
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

IMG_EXT = ".jpg"
LBL_EXT = ".txt"

def load_yolo_boxes(lbl: Path) -> List[Tuple[float,float,float,float]]:
    out: List[Tuple[float,float,float,float]] = []
    if not lbl.exists():
        return out
    txt = lbl.read_text(encoding="utf-8").strip()
    if not txt:
        return out
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = map(float, (cx, cy, w, h))
        except Exception:
            continue
        x1 = cx - w/2.0; y1 = cy - h/2.0
        x2 = cx + w/2.0; y2 = cy + h/2.0
        out.append((x1,y1,x2,y2))
    return out

def draw_boxes(img_path: Path, lbl_path: Path, title: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    boxes = load_yolo_boxes(lbl_path)
    for (x1n,y1n,x2n,y2n) in boxes:
        x1 = int(x1n * W); y1 = int(y1n * H)
        x2 = int(x2n * W); y2 = int(y2n * H)
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    # header band with title + count
    band_h = 28
    band = Image.new("RGB", (W, band_h), (0,0,0))
    img2 = Image.new("RGB", (W, H+band_h), (0,0,0))
    img2.paste(band, (0,0))
    img2.paste(img, (0,band_h))
    # text
    draw2 = ImageDraw.Draw(img2)
    text = f"{title} | boxes: {len(boxes)}"
    draw2.text((8, 6), text, fill=(255,255,255))
    return img2

def find_common_stems(base_dir: Path, var_dir: Path, split: str) -> List[str]:
    bimgs = {p.stem for p in (base_dir/"images"/split).glob("*.jpg")}
    vimgs = {p.stem for p in (var_dir/"images"/split).glob("*.jpg")}
    return sorted(list(bimgs & vimgs))

def is_positive(lbl: Path) -> bool:
    return lbl.exists() and lbl.read_text(encoding="utf-8").strip() != ""

def pick_variant_only_neg(base_dir: Path, var_dir: Path, split: str) -> List[str]:
    # frames that are NEGATIVE in VARIANT but either (a) absent in BASE or (b) positive in BASE
    vimgs = {p.stem for p in (var_dir/"images"/split).glob("*.jpg")}
    out = []
    for stem in sorted(vimgs):
        v_lbl = var_dir/"labels"/split/(stem + LBL_EXT)
        if is_positive(v_lbl):
            continue
        b_img = base_dir/"images"/split/(stem + IMG_EXT)
        b_lbl = base_dir/"labels"/split/(stem + LBL_EXT)
        if not b_img.exists() or is_positive(b_lbl):
            out.append(stem)
    return out

def pick_variant_duplicates(var_dir: Path, split: str) -> List[Tuple[str,str]]:
    # Returns list of (dup_stem, base_stem)
    dup_pairs = []
    for img in (var_dir/"images"/split).glob("*.jpg"):
        stem = img.stem
        if "_posdup" in stem or "_evtdup" in stem or "_bindup" in stem:
            base_stem = stem.split("_posdup")[0].split("_evtdup")[0].split("_bindup")[0]
            dup_pairs.append((stem, base_stem))
    return sorted(dup_pairs)

def make_pair(base_dir: Path, var_dir: Path, split: str, stem_base: str, stem_var: Optional[str], idx: int, out_dir: Path):
    if stem_var is None:
        stem_var = stem_base
    b_img = base_dir/"images"/split/(stem_base + IMG_EXT)
    b_lbl = base_dir/"labels"/split/(stem_base + LBL_EXT)
    v_img = var_dir/"images"/split/(stem_var + IMG_EXT)
    v_lbl = var_dir/"labels"/split/(stem_var + LBL_EXT)
    if not (b_img.exists() and v_img.exists()):
        return False
    left = draw_boxes(b_img, b_lbl, f"BASE:{stem_base}")
    right = draw_boxes(v_img, v_lbl, f"VAR:{stem_var}")
    # side-by-side
    W = left.width + right.width
    H = max(left.height, right.height)
    canvas = Image.new("RGB", (W, H), (20,20,20))
    canvas.paste(left, (0,0))
    canvas.paste(right, (left.width, 0))
    out_path = out_dir / f"{idx:03d}__{stem_var}.png"
    canvas.save(out_path)
    return True

def build_html(out_dir: Path):
    items = sorted([p.name for p in out_dir.glob("*.png")])
    html = ["<html><head><meta charset='utf-8'><title>Visual Compare</title>",
            "<style>body{background:#111;color:#ddd;font-family:Arial, sans-serif} .g{display:flex;flex-wrap:wrap;gap:10px} img{max-width:48%;height:auto;border:1px solid #333}</style>",
            "</head><body><h1>Visual Compare</h1><div class='g'>"]
    for n in items:
        html.append(f"<div><img src='{n}'/><div>{n}</div></div>")
    html.append("</div></body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--n", type=int, default=24, help="Number of pairs to export")
    ap.add_argument("--mode", type=str, default="common_pos",
                    choices=["common_pos","common_any","variant_only_neg","variant_duplicates"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("common_pos","common_any"):
        stems = find_common_stems(args.base, args.variant, args.split)
        if args.mode == "common_pos":
            stems = [s for s in stems if is_positive(args.base/"labels"/args.split/(s + LBL_EXT))]
        random.shuffle(stems)
        stems = stems[:args.n]
        idx = 1
        for s in stems:
            if make_pair(args.base, args.variant, args.split, s, None, idx, out_dir):
                idx += 1

    elif args.mode == "variant_only_neg":
        stems = pick_variant_only_neg(args.base, args.variant, args.split)
        random.shuffle(stems)
        stems = stems[:args.n]
        idx = 1
        for s in stems:
            if make_pair(args.base, args.variant, args.split, s, s, idx, out_dir):
                idx += 1

    elif args.mode == "variant_duplicates":
        pairs = pick_variant_duplicates(args.variant, args.split)
        random.shuffle(pairs)
        pairs = pairs[:args.n]
        idx = 1
        for dup_stem, base_stem in pairs:
            if make_pair(args.base, args.variant, args.split, base_stem, dup_stem, idx, out_dir):
                idx += 1

    build_html(out_dir)
    print(f"Saved {len(list(out_dir.glob('*.png')))} pairs to {out_dir}")
    print(f"Open: {out_dir/'index.html'}")

if __name__ == "__main__":
    main()
