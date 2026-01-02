#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import xml.etree.ElementTree as ET


# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class ObjAnno:
    lesion_uid: str
    bbox_xyxy: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax


@dataclass
class FrameData:
    frame_idx: int
    img_path: Path
    boxes_by_uid: Dict[str, List[Tuple[int, int, int, int]]]  # lesion_uid -> list of xyxy boxes


# -----------------------------
# Helpers
# -----------------------------
_FRAMEIDX_RE = re.compile(r"_(\d+)(?:\.0)?$", re.IGNORECASE)


def safe_dirname(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_")


def discover_video_ids(realcolon_root: Path) -> List[str]:
    vids: List[str] = []
    for p in realcolon_root.iterdir():
        if p.is_dir() and p.name.endswith("_annotations"):
            vids.append(p.name[: -len("_annotations")])
    return sorted(vids)


def extract_frame_idx_from_name(name: str) -> int:
    """
    Robust to:
      001-001_18185.xml
      001-012_9999.0.xml
      001-012_9999.0.jpg
    """
    stem = Path(name).stem  # e.g. "001-012_9999.0"
    m = _FRAMEIDX_RE.search(stem)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def parse_voc_xml(xml_path: Path) -> Tuple[Optional[str], List[ObjAnno]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else None

    objs: List[ObjAnno] = []
    for obj in root.findall("object"):
        uid_node = obj.find("unique_id")
        bbox = obj.find("bndbox")
        if uid_node is None or not uid_node.text or not uid_node.text.strip():
            continue
        if bbox is None:
            continue

        def get_int(tag: str) -> Optional[int]:
            n = bbox.find(tag)
            if n is None or n.text is None:
                return None
            try:
                return int(float(n.text.strip()))
            except Exception:
                return None

        xmin = get_int("xmin")
        ymin = get_int("ymin")
        xmax = get_int("xmax")
        ymax = get_int("ymax")
        if None in (xmin, ymin, xmax, ymax):
            continue

        x1, x2 = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        y1, y2 = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
        objs.append(ObjAnno(lesion_uid=uid_node.text.strip(), bbox_xyxy=(x1, y1, x2, y2)))

    return filename, objs


def draw_boxes_for_frame_with_target_label(
    img,
    target_lesion_uid: str,
    boxes_by_uid: Dict[str, List[Tuple[int, int, int, int]]],
):
    """
    Draw boxes for ALL lesions in the frame.
    Only the target lesion gets a label "lesion_<uid>" (if present).
    """
    # 1) Draw all boxes (plain rectangles)
    for uid, boxes in boxes_by_uid.items():
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2) Label only target lesion, if it exists in this frame
    target_boxes = boxes_by_uid.get(target_lesion_uid, [])
    if target_boxes:
        x1, y1, _, _ = target_boxes[0]
        cv2.putText(
            img,
            f"lesion_{target_lesion_uid}",
            (max(0, x1), max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _toggle_dot0(name: str) -> str:
    """
    Toggle a trailing '.0' just before the extension:
      foo.jpg <-> foo.0.jpg
      foo.xml <-> foo.0.xml
    If it doesn't match the pattern, return unchanged.
    """
    p = Path(name)
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        # Used for images only in this script
        return name

    stem = p.stem  # could be "..._1234" or "..._1234.0"
    if stem.endswith(".0"):
        new_stem = stem[:-2]
    else:
        new_stem = stem + ".0"
    return new_stem + p.suffix


def resolve_image_path(frames_dir: Path, preferred_name: str) -> Optional[Path]:
    """
    Try to find the corresponding image even if REAL-Colon uses a '.0' variant.
    Order:
      1) preferred_name
      2) toggled .0 variant
    """
    c1 = frames_dir / preferred_name
    if c1.exists():
        return c1
    c2_name = _toggle_dot0(preferred_name)
    c2 = frames_dir / c2_name
    if c2.exists():
        return c2
    return None


def read_num_frames(video_info_csv: Path) -> Dict[str, int]:
    """
    Optional sanity check: expected num_frames per video.
    """
    out: Dict[str, int] = {}
    if not video_info_csv.exists():
        return out
    with video_info_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get("unique_video_name") or "").strip()
            nf = (row.get("num_frames") or "").strip()
            if not vid:
                continue
            try:
                out[vid] = int(float(nf))
            except Exception:
                pass
    return out


# -----------------------------
# Core logic
# -----------------------------
def build_frame_index_for_video(ann_dir: Path, frames_dir: Path) -> Dict[int, FrameData]:
    """
    Parse all XMLs for a video into:
      frame_idx -> FrameData(img_path, boxes_by_uid)
    Robust to '.0' in filenames and to image name mismatches.
    """
    frame_map: Dict[int, FrameData] = {}

    xml_files = sorted(ann_dir.glob("*.xml"), key=lambda p: extract_frame_idx_from_name(p.name))
    for xml_path in xml_files:
        frame_idx = extract_frame_idx_from_name(xml_path.name)
        if frame_idx < 0:
            continue

        filename, objs = parse_voc_xml(xml_path)

        # Determine the preferred image filename:
        # - If XML <filename> exists, use it.
        # - Else: use xml name but with .jpg (keeps .0 if present).
        preferred_jpg = Path(filename).name if filename else xml_path.with_suffix(".jpg").name

        img_path = resolve_image_path(frames_dir, preferred_jpg)
        if img_path is None:
            # Could not locate image in either variant; keep it missing by skipping
            continue

        boxes_by_uid: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for o in objs:
            boxes_by_uid.setdefault(o.lesion_uid, []).append(o.bbox_xyxy)

        frame_map[frame_idx] = FrameData(
            frame_idx=frame_idx,
            img_path=img_path,
            boxes_by_uid=boxes_by_uid,
        )

    return frame_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--realcolon_root", type=str, default="/data/local/aschwab/data/realColon")
    ap.add_argument("--out_root", type=str, default="/data/local/aschwab/data/lesions_realColon")
    ap.add_argument("--context", type=int, default=100)
    ap.add_argument("--limit_videos", type=int, default=0)
    ap.add_argument(
        "--prefix_with_video",
        action="store_true",
        help="Prefix copied filenames with <video>__ to avoid collisions across videos.",
    )
    ap.add_argument(
        "--warn_framecount",
        action="store_true",
        help="Warn if video_info.csv num_frames differs from number of annotated XML frames found.",
    )
    args = ap.parse_args()

    root = Path(args.realcolon_root).resolve()
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    vids = discover_video_ids(root)
    if args.limit_videos and args.limit_videos > 0:
        vids = vids[: args.limit_videos]

    if not vids:
        raise RuntimeError(f"No '*_annotations' folders found in: {root}")

    expected_num_frames = read_num_frames(root / "video_info.csv")

    # Avoid redoing the same work across videos/lesions
    done: Set[Tuple[str, str, int]] = set()  # (lesion_uid, video, frame_idx)

    total_originals = 0
    total_annotated = 0
    missing_or_unreadable = 0

    print(f"[INFO] Videos to process: {len(vids)}")
    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Context window: ±{args.context} frames")

    for vid in vids:
        ann_dir = root / f"{vid}_annotations"
        frames_dir = root / f"{vid}_frames"

        if not ann_dir.is_dir():
            print(f"[WARN] Missing annotations folder: {ann_dir}. Skipping {vid}.")
            continue
        if not frames_dir.is_dir():
            print(f"[WARN] Missing frames folder: {frames_dir}. Skipping {vid}.")
            continue

        # Optional framecount warning (does not block)
        if args.warn_framecount and vid in expected_num_frames:
            xml_count = len(list(ann_dir.glob("*.xml")))
            exp = expected_num_frames[vid]
            if xml_count != exp:
                print(f"[WARN] {vid}: video_info num_frames={exp} but XML files={xml_count}. Proceeding anyway.")

        frame_map = build_frame_index_for_video(ann_dir, frames_dir)
        if not frame_map:
            print(f"[WARN] {vid}: no usable frames (XML parsed but images not found). Skipping.")
            continue

        available = sorted(frame_map.keys())
        min_idx, max_idx = available[0], available[-1]

        # For each lesion, collect positive frames
        lesion_pos: Dict[str, List[int]] = {}
        for fi, fd in frame_map.items():
            for lesion_uid in fd.boxes_by_uid.keys():
                lesion_pos.setdefault(lesion_uid, []).append(fi)
        for uid in lesion_pos:
            lesion_pos[uid].sort()

        print(f"[INFO] {vid}: usable_frames={len(frame_map)} (idx {min_idx}..{max_idx}), lesions={len(lesion_pos)}")

        # Process each lesion
        for lesion_uid, pos_list in lesion_pos.items():
            # Build union of context windows around positive frames
            wanted: Set[int] = set()
            for p in pos_list:
                lo = max(min_idx, p - args.context)
                hi = min(max_idx, p + args.context)
                wanted.update(range(lo, hi + 1))

            lesion_dir = out_root / safe_dirname(lesion_uid)
            images_dir = lesion_dir / "images"
            ann_images_dir = lesion_dir / "images_annotated"
            ensure_dir(images_dir)
            ensure_dir(ann_images_dir)

            for fi in sorted(wanted):
                k = (lesion_uid, vid, fi)
                if k in done:
                    continue
                done.add(k)

                fd = frame_map.get(fi)
                if fd is None:
                    continue

                if not fd.img_path.exists():
                    missing_or_unreadable += 1
                    continue

                # Destination filename
                if args.prefix_with_video:
                    dst_name = f"{vid}__{fd.img_path.name}"
                else:
                    dst_name = fd.img_path.name

                # Copy original
                dst_img = images_dir / dst_name
                if not dst_img.exists():
                    shutil.copy2(fd.img_path, dst_img)
                    total_originals += 1

                # Write annotated
                dst_ann = ann_images_dir / dst_name
                if not dst_ann.exists():
                    img = cv2.imread(str(fd.img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        missing_or_unreadable += 1
                        continue
                    img_vis = draw_boxes_for_frame_with_target_label(
                        img,
                        target_lesion_uid=lesion_uid,
                        boxes_by_uid=fd.boxes_by_uid,
                    )
                    ok = cv2.imwrite(str(dst_ann), img_vis)
                    if ok:
                        total_annotated += 1
                    else:
                        missing_or_unreadable += 1

    print("[INFO] Done.")
    print(f"[INFO] Copied originals: {total_originals}")
    print(f"[INFO] Wrote annotated:  {total_annotated}")
    if missing_or_unreadable:
        print(f"[INFO] Missing/unreadable images: {missing_or_unreadable}")


if __name__ == "__main__":
    main()


''''
python3 move.py \
  --realcolon_root /data/local/aschwab/data/realColon \
  --out_root /data/local/aschwab/data/lesions_realColon \
  --context 100 \
  --prefix_with_video \
  --warn_framecount
'''