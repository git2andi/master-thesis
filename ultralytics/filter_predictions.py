"""
Reduce the 22GB+ predictions detr file
  - drop detections with score <= score_thr
  - keep the top max_det detections per image_id, sorted. (COCO uses top 100 for eval only, detr produces 300 per image by its architecture)

python filter_predictions.py \
    --output path/dest/predictions.json \
    --input path/original/predictions.json \
    --max-det 100 \
    --score-thr 0.001

(max-det 100 > 7.8GB; max-det 100 + thr 0.005 > 7.5GB; max-det 100 + thr 0.1 > 70MB)
changing thr changes official mAP! (only verry small depending on the value set) but not doing it might kill the process by the system as its still too big...
"""


import argparse
import json
from collections import defaultdict
from pathlib import Path
import heapq
from decimal import Decimal

import ijson  # pip install ijson


def filter_predictions_topk(
    input_path: Path,
    output_path: Path,
    max_det: int = 100,
    score_thr: float = 0.00,
):
    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"max_det per image: {max_det}, score_thr: {score_thr}")

    # image_key > min-heap of (score, counter, det)
    per_image = defaultdict(list)
    counter = 0
    total_in = 0

    with input_path.open("rb") as f:
        for det in ijson.items(f, "item"):
            total_in += 1

            score = float(det.get("score", 0.0))
            if score <= score_thr:
                continue

            img_key = det.get("image_id")
            if img_key is None:
                img_key = det.get("file_name")
            if img_key is None:
                continue

            heap = per_image[img_key]
            counter += 1
            # python heap is min-heap by first elem
            heapq.heappush(heap, (score, counter, det))
            if len(heap) > max_det:
                heapq.heappop(heap)

            if total_in % 1_000_000 == 0:
                print(f"processed {total_in:_} detections...")

    print(f"total raw detections: {total_in:_}")
    print(f"total images kept: {len(per_image):_}")

    filtered = []
    for img_key, heap in per_image.items():
        heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
        filtered.extend([d for (_, _, d) in heap_sorted])

    print(f"total detections kept: {len(filtered):_}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        print(f"save to: {output_path}")
        json.dump(filtered, f, default=lambda o: float(o) if isinstance(o, Decimal) else o)


def main():
    ap = argparse.ArgumentParser(
    )
    ap.add_argument(
        "--input",
        default=Path("/home/stud/aschwab/master-thesis/missing/rtdetr_sun_realcolon/rtdetr_sun_realcolon.json"),
        type=str,
    )
    ap.add_argument(
        "--output",
        default=Path("/home/stud/aschwab/master-thesis/best_epochs/cross_dataset/rtdert_sun_realcolon.json"),
        type=str,
    )
    ap.add_argument(
        "--max-det",
        type=int,
        default=100
    )
    ap.add_argument(
        "--score-thr",
        type=float,
        default=0.01
    )
    args = ap.parse_args()

    filter_predictions_topk(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_det=args.max_det,
        score_thr=args.score_thr,
    )


if __name__ == "__main__":
    main()
