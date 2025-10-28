#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path

def check_dir(p: Path):
    if not p.exists():
        sys.exit(f"ERROR: missing path: {p}")

def make_yaml(prefix: str, held_out: int, base: Path, out: Path, insts=(1,2,3,4)):
    incl = [i for i in insts if i != held_out]
    def ip(i): return base / f"{prefix}_i{i}" / "images"

    # sanity checks
    for i in insts:
        for split in ["train", "val", "test"]:
            check_dir(ip(i) / split)

    train_list = [str(ip(i) / "train") for i in incl]
    val_list   = [str(ip(i) / "val")   for i in incl]
    test_path  = str(ip(held_out) / "test")

    yaml = f"""# REAL-Colon YOLO data — LOIO fold (hold out institution i{held_out})
# Train/val: institutions {', '.join(f'i{x}' for x in incl)}, Test: i{held_out}
train:
"""
    yaml += "".join([f"  - {p}\n" for p in train_list])
    yaml += "val:\n" + "".join([f"  - {p}\n" for p in val_list])
    yaml += f"""test: {test_path}

nc: 1
names: ["lesion"]
"""
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_LOIO_i{held_out}.yaml"
    (out / fname).write_text(yaml)
    return out / fname

def main():
    ap = argparse.ArgumentParser(description="Create LOIO YAMLs for REAL-Colon i1..i4")
    ap.add_argument("--base",   default="/data/local/aschwab/data", help="Root containing realColon_* directories")
    ap.add_argument("--prefix", default="realColon_640x640",        help="Dataset prefix (expects *_i1..i4)")
    ap.add_argument("--out",    default="/data/local/aschwab/data/realColon_640x640_loio",
                    help="Output folder for LOIO YAMLs")
    args = ap.parse_args()

    base = Path(args.base)
    out  = Path(args.out)
    prefix = args.prefix

    # write 4 folds
    written = []
    for k in [1,2,3,4]:
        p = make_yaml(prefix, k, base, out)
        written.append(p)

    print("Created LOIO YAMLs:")
    for p in written:
        print(" -", p)

if __name__ == "__main__":
    main()
