"""
REAL-Colon sanity checks

Usage:
  python get_dataset_info_realcolon.py --root /data/local/aschwab/data/real_colon_allPos_allNeg

  """



import os
import argparse
from collections import Counter
from typing import Tuple, List

from PIL import Image



SPLIT_CONFIGS = [
    ["train_images", "validation_images", "test_images"], # original                       
    ["train", "val", "test"],    # adapted original non padded
    ["train2", "val2", "test2"], # padded
]

def infer_splits(root: str) -> List[str]:
    entries = set(os.listdir(root))
    for config in SPLIT_CONFIGS:
        if all(split in entries and os.path.isdir(os.path.join(root, split))
               for split in config):
            print(f"[INFO] Using split configuration: {config}")
            return config
    raise RuntimeError(
        f"Could not infer splits under root '{root}'. "
        f"Expected one of: {SPLIT_CONFIGS}"
    )


def collect_image_sizes(root: str, splits: List[str]) -> Counter[Tuple[int, int]]:
    size_counter: Counter[Tuple[int, int]] = Counter()

    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            print(f"[WARN] Split directory not found, skipping: {split_path}")
            continue

        print(f"[INFO] Scanning {split_path} ...")
        for dirpath, dirnames, filenames in os.walk(split_path, followlinks=True):
            for fname in filenames:
                if not fname.lower().endswith((".jpg")):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with Image.open(fpath) as img:
                        width, height = img.size
                    size_counter[(width, height)] += 1
                except Exception as e:
                    print(f"[WARN] Could not read image {fpath}: {e}")

    return size_counter


def main():
    parser = argparse.ArgumentParser(
        description="Summarize image frame sizes and aspect ratios over dataset splits."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        default="/data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient",
    )
    args = parser.parse_args()

    splits = infer_splits(args.root)
    size_counter = collect_image_sizes(args.root, splits)

    total_images = sum(size_counter.values())
    print("\n=== Image size summary over all splits ===")
    print(f"Root:   {args.root}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Total images counted: {total_images}\n")

    print(f"{'count':>10}  {'size':>10}  {'ratio':>10}  {'ratio_float':>12}")
    print("-" * 50)

    for (w, h), count in sorted(size_counter.items(), key=lambda x: x[1], reverse=True):
        ratio_gcd = gcd(w, h)
        simple_w = w // ratio_gcd
        simple_h = h // ratio_gcd
        ratio_str = f"{simple_w}:{simple_h}"
        ratio_float = w / h if h != 0 else 0.0
        print(f"{count:10d}  {w}x{h:>5}  {ratio_str:>10}  {ratio_float:12.4f}")


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


if __name__ == "__main__":
    main()


#=== Image size summary over all splits ===
#Root:  /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient
#Splits: train_images, validation_images, test_images
#Total images counted: 2638023
# 1579743  1352x1080
#  296312  1350x1080
#  224536  1248x959
#  177462  1162x1007
#  124613  1244x1080
#   63316  1246x1080
#   53372  1164x1034
#   47315  1164x1010
#   28236  1158x1008
#   21634  1160x1052
#   21484  1158x1024


#=== Image size summary over all splits ===
#Root:   /data/local/aschwab/data/real_colon_allPos_allNeg
#Splits: train_images, validation_images, test_images
#Total images counted: 2757723
#     count        size       ratio   ratio_float
#--------------------------------------------------
#   1661738  1352x 1080     169:135        1.2519
#    304079  1350x 1080         5:4        1.2500
#    234073  1248x  959    1248:959        1.3014
#    183927  1162x 1007   1162:1007        1.1539
#    128751  1244x 1080     311:270        1.1519
#     66911  1246x 1080     623:540        1.1537
#     54674  1164x 1034     582:517        1.1257
#     48470  1164x 1010     582:505        1.1525
#     29108  1158x 1008     193:168        1.1488
#     23527  1160x 1052     290:263        1.1027
#     22465  1158x 1024     579:512        1.1309