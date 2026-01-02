import sqlite3
import ijson
from pathlib import Path

DB_PATH = Path("pred_cache.db")

MODELS = [
    ("RTDETR", "ultralytics", Path("/home/stud/aschwab/master-thesis/best_epochs/results/filtered_rtdetr_realColon_640_s42.json")),
    ("YOLOv11", "ultralytics", Path("/home/stud/aschwab/master-thesis/best_epochs/results/predictions_y11m_realColon_640_s42_b208.json")),
    ("YOLOv8", "ultralytics", Path("/home/stud/aschwab/master-thesis/best_epochs/results/predictions_y8m_realColon_640_s42_b208.json")),
    ("FasterRCNN", "detectron2", Path("/home/stud/aschwab/master-thesis/best_epochs/results/fasterrcnn_realColon_640_s42_b96.json")),
]

TOPK_STORE = 20
COMMIT_EVERY = 50_000

def canonical_realcolon_stem_id(name: str):
    stem = Path(name).stem
    if "_" not in stem:
        return None
    p, s = stem.rsplit("_", 1)
    try:
        return f"{p}_{int(s)}"
    except ValueError:
        return None


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=OFF;

CREATE TABLE IF NOT EXISTS preds (
    model TEXT,
    frame_id TEXT,
    score REAL,
    x REAL,
    y REAL,
    w REAL,
    h REAL
);

CREATE INDEX IF NOT EXISTS idx_model_frame
ON preds(model, frame_id);
""")

def flush_frame(model, frame_id, buf):
    if not buf:
        return 0

    # sort by score descending
    buf.sort(key=lambda d: float(d["score"]), reverse=True)
    kept = buf[:TOPK_STORE]

    for d in kept:
        x, y, w, h = d["bbox"]

        cur.execute(
            "INSERT INTO preds VALUES (?,?,?,?,?,?,?)",
            (
                str(model),
                str(frame_id),
                float(d["score"]),   # force Python float
                float(x),
                float(y),
                float(w),
                float(h),
            ),
        )

    return len(kept)



for model, framework, pred_path in MODELS:
    print(f"[BUILD] {model}")
    last_frame = None
    buf = []
    n_rows = 0

    with pred_path.open("rb") as f:
        for det in ijson.items(f, "item"):
            if not isinstance(det, dict):
                continue

            score = det.get("score", det.get("conf"))
            bbox = det.get("bbox")
            if score is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            if framework == "ultralytics":
                img_id = det.get("image_id")
                if not isinstance(img_id, str):
                    continue
                frame_id = canonical_realcolon_stem_id(img_id)
            else:  # detectron2
                frame_id = str(det.get("image_id"))  # map later if needed

            if frame_id is None:
                continue

            if last_frame is None:
                last_frame = frame_id

            if frame_id != last_frame:
                n_rows += flush_frame(model, last_frame, buf)
                buf.clear()
                last_frame = frame_id

                if n_rows % COMMIT_EVERY == 0:
                    conn.commit()
                    print(f"  inserted {n_rows:,} rows")

            buf.append({
                "score": float(score),
                "bbox": [float(v) for v in bbox],
            })

    # flush last frame
    n_rows += flush_frame(model, last_frame, buf)
    conn.commit()
    print(f"[DONE] {model}: {n_rows:,} rows stored")

conn.close()
print("Cache build complete.")