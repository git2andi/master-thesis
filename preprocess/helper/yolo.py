# preprocess/helper/yolo.py

from typing import List, Tuple

# Write a YOLO .txt label file
# Writes one line per box in the format: <class> <cx> <cy> <w> <h>
def write_yolo_label_file(
    out_path: str,
    boxes: List[Tuple[int, float, float, float, float]],
) -> None:
    lines = []
    for cls, cx, cy, w, h in boxes:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
