#!/usr/bin/env bash
# Step 6 — Speed & FPS (detector-only via Ultralytics benchmark, optional end-to-end)
# Reqs: ultralytics>=8.x, pandas installed (benchmark returns a DataFrame)
#
# Usage (detector-only, FP16, imgsz=224 on GPU 0):
#   bash step6_speed.sh --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 --device 0 --fp16
#
# Add FP32 as well:
#   ... --fp16 --fp32
#
# Optional end-to-end timing on a video (decode+preproc+model+NMS[+draw]):
#   ... --e2e-video /path/to/video.mp4 [--draw]
#
# Outputs:
#   ~/master-thesis/masterThesis/<RUN>/pipeline/speed/
#     detector_benchmark.csv
#     detector_benchmark.json
#     detector_summary.json        # mean_ms, p50_ms, p90_ms, FPS for chosen row(s)
#     e2e_latency.json             # if --e2e-video is used
#
set -euo pipefail

RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
DO_FP16=0
DO_FP32=0
E2E_VIDEO=""
E2E_DRAW=0
RUN_ROOT="${HOME}/master-thesis/masterThesis"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)     RUN="$2"; shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    --imgsz)   IMGSZ="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --fp16)    DO_FP16=1; shift 1 ;;
    --fp32)    DO_FP32=1; shift 1 ;;
    --e2e-video) E2E_VIDEO="$2"; shift 2 ;;
    --draw)    E2E_DRAW=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" && -n "$DATA" ]] || { echo "Usage: --run <RUN> --data <DATA_YAML> [--imgsz N] [--device 0] [--fp16] [--fp32] [--e2e-video path] [--draw]"; exit 2; }

RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

MODEL="${RUN_DIR}/weights/best.pt"
[[ -f "$MODEL" ]] || { echo "Model weights not found: $MODEL"; exit 1; }
[[ -f "$DATA"  ]] || { echo "Dataset YAML not found: $DATA"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
SPEED_DIR="${PIPE_DIR}/speed"
mkdir -p "$SPEED_DIR"

echo "==> STEP 6 (Speed & FPS)"
echo "Run     : $RUN"
echo "Model   : $MODEL"
echo "Data    : $DATA"
echo "imgsz   : $IMGSZ"
echo "Device  : $DEVICE"
echo "FP16    : $DO_FP16"
echo "FP32    : $DO_FP32"
echo "E2E     : ${E2E_VIDEO:-<none>}  draw=${E2E_DRAW}"
echo "Out dir : $SPEED_DIR"
echo

# ---------- 6a) Detector-only benchmark via Ultralytics API ----------
DET_CSV="${SPEED_DIR}/detector_benchmark.csv"
DET_JSON="${SPEED_DIR}/detector_benchmark.json"
DET_SUM="${SPEED_DIR}/detector_summary.json"

python - <<PY
import json, pandas as pd, os, time
from pathlib import Path
from ultralytics.utils.benchmarks import benchmark  # docs: returns a pandas DataFrame
# ref: https://docs.ultralytics.com/modes/benchmark/ ; API attrs: imgsz, half, num_warmup_runs, num_timed_runs, min_time
#      https://docs.ultralytics.com/reference/utils/benchmarks/

model_path = "${MODEL}"
data_yaml  = "${DATA}"
imgsz     = int("${IMGSZ}")
device    = "${DEVICE}"

# We'll run up to two precise profiles: FP16 and/or FP32 (half=False)
jobs = []
if ${DO_FP16}:
    jobs.append(("fp16", True))
if ${DO_FP32} or not jobs:
    jobs.append(("fp32", False))  # default to FP32 if user didn't set any flag

rows = []
for tag, half in jobs:
    print(f"[bench] {tag}  imgsz={imgsz}  device={device}")
    # Use modest warmup/timed runs to finish quickly but stabilize clocks
    df = benchmark(
        model=model_path,
        data=data_yaml,
        imgsz=imgsz,
        half=half,
        device=device,
        verbose=False
    )
    # df typically includes columns like 'format','size(MB)','mAP50-95','inference(ms)','NMS(ms)','total(ms)', etc.
    df["tag"] = tag
    df["imgsz"] = imgsz
    rows.append(df)

# concat and save
out = pd.concat(rows, ignore_index=True)
csv_path = Path(r"${DET_CSV}")
json_path = Path(r"${DET_JSON}")
csv_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(csv_path, index=False)
out.to_json(json_path, orient="records", indent=2)

# Make a compact summary: pick the PyTorch row if present; else first row
def summarize(df: pd.DataFrame):
    # try to find pytorch/native row
    pref = None
    for k in df.columns:
        if k.lower() == "format":
            pref = df[df[k].str.lower().str.contains("pytorch|torchscript", na=False)]
            break
    pick = pref.iloc[0] if pref is not None and len(pref) else df.iloc[0]
    # infer timing columns (fall back gracefully)
    def get(colnames, default=None):
        for c in colnames:
            if c in pick:
                try:
                    return float(pick[c])
                except Exception:
                    pass
        return default
    mean_ms = get(["inference(ms)","total(ms)","latency(ms)"], None)
    # If total(ms) exists and includes NMS, prefer it
    total_ms = get(["total(ms)","latency(ms)","inference(ms)"], None)
    res = {
        "tag": pick.get("tag",""),
        "format": pick.get("format",""),
        "imgsz": int(pick.get("imgsz", ${IMGSZ})),
        "mean_ms": total_ms if total_ms is not None else mean_ms,
    }
    if res["mean_ms"] is not None and res["mean_ms"] > 0:
        res["fps"] = 1000.0 / res["mean_ms"]
    # grab median/p90 if present (newer versions may add quantiles)
    res["p50_ms"] = get(["p50(ms)","median(ms)"], None)
    res["p90_ms"] = get(["p90(ms)"], None)
    return res

summary = summarize(out.copy())
Path(r"${DET_SUM}").write_text(json.dumps(summary, indent=2))
print(f"[bench] wrote: {csv_path}  {json_path}  {Path(r"${DET_SUM}")}")
PY

# ---------- 6b) Optional end-to-end timing on a video ----------
if [[ -n "${E2E_VIDEO}" ]]; then
  [[ -f "$E2E_VIDEO" ]] || { echo "E2E video not found: $E2E_VIDEO"; exit 1; }
  E2E_JSON="${SPEED_DIR}/e2e_latency.json"
  python - <<PY
import time, json, statistics, cv2, torch
from ultralytics import YOLO

video = r"${E2E_VIDEO}"
imgsz = int("${IMGSZ}")
device = "${DEVICE}"
model = YOLO(r"${MODEL}")
model.to(f"cuda:{device}" if device not in ("cpu","-1") else "cpu")
model.model.eval()
torch.set_grad_enabled(False)

# warmup frames
N_WARM = 30
lat = []
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise SystemExit(f"cannot open video: {video}")

# warmup
i=0
while i < N_WARM:
    ok, frame = cap.read()
    if not ok: break
    _ = model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    i+=1

# measure up to 1000 frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
MAXF=1000
count=0
DRAW = bool(int("${E2E_DRAW}"))
while count < MAXF:
    ok, frame = cap.read()
    if not ok: break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    res = model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    if DRAW:
        r = res[0]
        if r and getattr(r, "boxes", None):
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lat.append( (time.perf_counter() - t0) * 1000.0 )
    count += 1

cap.release()
mean = statistics.fmean(lat) if lat else None
p50  = statistics.median(lat) if lat else None
p90  = (sorted(lat)[int(0.90*(len(lat)-1))] if lat else None)
out = {
  "video": video,
  "frames_measured": len(lat),
  "imgsz": imgsz,
  "draw": DRAW,
  "mean_ms": mean,
  "p50_ms": p50,
  "p90_ms": p90,
  "fps": (1000.0/mean if mean and mean>0 else None)
}
open(r"${E2E_JSON}", "w").write(json.dumps(out, indent=2))
print(f"[e2e] wrote: ${E2E_JSON}")
PY
fi

echo
echo "==> Step 6 done."
echo "Detector CSV : ${DET_CSV}"
echo "Detector JSON: ${DET_JSON}"
echo "Summary      : ${DET_SUM}"
[[ -n "${E2E_VIDEO}" ]] && echo "E2E JSON     : ${SPEED_DIR}/e2e_latency.json" || true
