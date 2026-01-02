# Check original
BASE=/data/local/aschwab/data/piccolo

for SPLIT in train validation test; do
  echo "=== $SPLIT ==="
  echo -n "  masks (Corrected.tif): "
  find "$BASE/$SPLIT/masks"  -type f -iname '*corrected.tif' | wc -l

  echo -n "  void (Void.tif):      "
  find "$BASE/$SPLIT/void"   -type f -iname '*void.tif'      | wc -l

  echo -n "  polyps (.png):       "
  find "$BASE/$SPLIT/polyps" -type f -iname '*.png'          | wc -l
  echo
done


BASE=/data/local/aschwab/data/piccolo

echo "=== TOTAL (train + validation + test) ==="
echo -n "masks (Corrected.tif): "
find "$BASE" -path '*/masks/*'  -type f -iname '*corrected.tif' | wc -l

echo -n "void (Void.tif):      "
find "$BASE" -path '*/void/*'   -type f -iname '*void.tif'      | wc -l

echo -n "polyps (.png):       "
find "$BASE" -path '*/polyps/*' -type f -iname '*.png'          | wc -l


# Check new
BASE=/data/local/aschwab/data/piccolo_split

echo "=== Image / Label counts per split ==="
for SPLIT in train val test; do
  imgs=$(find "$BASE/images/$SPLIT" -type f -name '*.jpg' | wc -l)
  lbls=$(find "$BASE/labels/$SPLIT" -type f -name '*.txt' | wc -l)
  echo "$SPLIT: images=$imgs  labels=$lbls"
done

echo
echo "=== Check 1-1 mapping (no missing label/image) ==="
for SPLIT in train val test; do
  echo "--- $SPLIT ---"
  ( cd "$BASE" && \
    comm -3 \
      <(find "images/$SPLIT" -type f -name '*.jpg' \
          | sed 's/^images\/'"$SPLIT"'\///; s/\.jpg$//' | sort) \
      <(find "labels/$SPLIT" -type f -name '*.txt' \
          | sed 's/^labels\/'"$SPLIT"'\///; s/\.txt$//' | sort) \
  )
done



python - << 'PY'
import json
from pathlib import Path

BASE = Path("/data/local/aschwab/data/piccolo_split")

splits = [
    ("train", "coco_annotations_train.json"),
    ("val",   "coco_annotations_val.json"),
    ("test",  "coco_annotations_test.json"),
]

overall_frames = 0
overall_pos = 0
overall_neg = 0
overall_boxes = 0

for split_name, coco_file in splits:
    print(f"=== {split_name.upper()} ===")
    images_dir = BASE / "images" / split_name
    labels_dir = BASE / "labels" / split_name
    coco_path = BASE / coco_file

    if not coco_path.exists():
        print(f"COCO file missing: {coco_path}")
        continue

    with coco_path.open("r") as f:
        coco = json.load(f)

    coco_images = coco.get("images", [])
    coco_anns = coco.get("annotations", [])

    # index annotations by image_id
    ann_by_img = {}
    for ann in coco_anns:
        img_id = ann["image_id"]
        ann_by_img.setdefault(img_id, 0)
        ann_by_img[img_id] += 1

    num_images = len(coco_images)
    num_boxes = len(coco_anns)
    num_pos = 0
    num_neg = 0

    # physical files
    disk_images = sorted(images_dir.glob("*.jpg"))
    disk_labels = sorted(labels_dir.glob("*.txt"))

    print(f"COCO images:        {num_images}")
    print(f"COCO annotations:   {num_boxes}")
    print(f"Disk images:        {len(disk_images)}")
    print(f"Disk labels:        {len(disk_labels)}")

    if len(disk_images) != num_images:
        print("WARNING: disk image count != COCO images")

    if len(disk_labels) != num_images:
        print("WARNING: disk label count != COCO images")

    mismatched_yolo = 0
    missing_label_files = 0

    stem_to_label = {p.stem: p for p in disk_labels}

    for img in coco_images:
        img_id = img["id"]
        fname = img["file_name"]
        stem = Path(fname).stem

        ann_count = ann_by_img.get(img_id, 0)
        if ann_count > 0:
            num_pos += 1
        else:
            num_neg += 1

        lbl_path = stem_to_label.get(stem)
        if lbl_path is None:
            missing_label_files += 1
            continue

        with lbl_path.open("r") as lf:
            yolo_lines = [ln for ln in lf.read().strip().splitlines() if ln.strip()]

        if len(yolo_lines) != ann_count:
            mismatched_yolo += 1

    print(f"Frames (COCO images): {num_images}")
    print(f"  Positive frames:    {num_pos}")
    print(f"  Negative frames:    {num_neg}")
    print(f"  Total boxes:        {num_boxes}")
    print(f"Missing label files:  {missing_label_files}")
    print(f"YOLO/COCO mismatches: {mismatched_yolo}")
    print()

    overall_frames += num_images
    overall_pos += num_pos
    overall_neg += num_neg
    overall_boxes += num_boxes

print("=== OVERALL (train + val + test) ===")
print(f"Total frames:   {overall_frames}")
print(f"Total positive: {overall_pos}")
print(f"Total negative: {overall_neg}")
print(f"Total boxes:    {overall_boxes}")
PY
