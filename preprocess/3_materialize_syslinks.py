'''
Make syslinks to images > slightly faster training
'''

import os
import shutil

DATASET_ROOT = "/data/local/aschwab/data/real_colon_allPos_fraction1ofNeg_onlyPatient"
SOURCE_SPLITS = ["train", "val", "test"]
TARGET_SUFFIX = "2"

def materialize_split(root, split, target_suffix):
    src_dir = os.path.join(root, "images", split)
    dst_dir = os.path.join(root, "images", f"{split}{target_suffix}")
    if not os.path.isdir(src_dir):
        print(f"[{split}] dir does not exist {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    print(f"[{split}] copying from {src_dir} > {dst_dir}")

    for dirpath, _, filenames in os.walk(src_dir):
        rel = os.path.relpath(dirpath, src_dir)
        dst_subdir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(dst_subdir, exist_ok=True)
        for fname in filenames:
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_subdir, fname)
            real_src = os.path.realpath(src_path)
            shutil.copy2(real_src, dst_path)
    print(f"[{split}] finished")


def main():
    for split in SOURCE_SPLITS:
        materialize_split(DATASET_ROOT, split, TARGET_SUFFIX)

if __name__ == "__main__":
    main()

# change paths in data.yaml if not done in 2)