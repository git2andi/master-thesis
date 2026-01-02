"""
SUN split sanity checks (images/labels + per-case distribution).

Usage:
  python sun_sanity.py
  python sun_sanity.py --base /data/local/aschwab/data/sun_split
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ("train", "val", "test")


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (p for p in root.rglob(pattern) if p.is_file())


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in iter_files(root, pattern))


def case_prefix(case_id: int) -> str:
    # matches shell glob: case${i}_*.jpg
    return f"case{case_id}_"


@dataclass
class CaseInfo:
    case_id: int
    case_type: str  # "POS" or "NEG"
    split: str      # "train"|"val"|"test"|"NONE"|"MULTI"
    n_images: int


def count_images_per_case_by_split(images_base: Path) -> Dict[int, Dict[str, int]]:
    """
    One pass over each split directory. Returns:
      counts[case_id][split] = number of jpgs for that case in that split.
    Assumes SUN filenames start with 'case<id>_'
    """
    counts: Dict[int, Dict[str, int]] = {i: {s: 0 for s in SPLITS} for i in range(1, 114)}

    for s in SPLITS:
        d = images_base / s
        if not d.exists():
            continue
        for p in d.glob("*.jpg"):
            name = p.name.lower()
            if not name.startswith("case"):
                continue
            # expected: case12_XXXX.jpg
            j = 4
            k = j
            while k < len(name) and name[k].isdigit():
                k += 1
            if k == j or k >= len(name) or name[k] != "_":
                continue
            cid = int(name[j:k])
            if 1 <= cid <= 113:
                counts[cid][s] += 1

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="SUN split sanity checks")
    ap.add_argument("--base", type=Path, default=Path("/data/local/aschwab/data/sun_split"))
    args = ap.parse_args()

    base = args.base
    images = base / "images"
    labels = base / "labels"

    # 1) per-split counts + totals
    print("## SUN split counts")
    img_tot = 0
    lbl_tot = 0
    for s in SPLITS:
        n_img = count_files(images / s, "*.jpg")
        n_lbl = count_files(labels / s, "*.txt")
        img_tot += n_img
        lbl_tot += n_lbl
        warn = "" if n_img == n_lbl else "  [WARN images!=labels]"
        print(f"{s:5s}  images={n_img:6d}  labels={n_lbl:6d}{warn}")

    print(f"{'TOTAL':5s}  images={img_tot:6d}  labels={lbl_tot:6d}")
    print()

    # 2) POS/NEG totals
    print("## Images by case (cases 1–100 POS, 101–113 NEG)")
    counts = count_images_per_case_by_split(images)

    pos_total = 0
    neg_total = 0
    multi_cases: List[int] = []
    missing_cases: List[int] = []

    case_infos: List[CaseInfo] = []

    for cid in range(1, 114):
        ctype = "POS" if cid <= 100 else "NEG"
        per_split = counts[cid]
        splits_with = [s for s in SPLITS if per_split.get(s, 0) > 0]
        n_images = sum(per_split.values())

        if cid <= 100:
            pos_total += n_images
        else:
            neg_total += n_images

        if len(splits_with) == 0:
            split = "NONE"
            missing_cases.append(cid)
        elif len(splits_with) == 1:
            split = splits_with[0]
        else:
            split = "MULTI"
            multi_cases.append(cid)

        case_infos.append(CaseInfo(cid, ctype, split, n_images))

    print(f"Positive images (cases 1–100):   {pos_total}")
    print(f"Negative images (cases 101–113): {neg_total}")
    print(f"Total images (POS+NEG):          {pos_total + neg_total}")
    print()

    # 3) per-case listing
    print("## Per-case split assignment")
    for info in case_infos:
        print(f"case{info.case_id:3d} ({info.case_type}) -> split={info.split:<5s} | images={info.n_images:6d}")

    # 4) print warnings
    # TODO
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
