#!/usr/bin/env bash
# Step 7 — Hard-negative mining at a locked operating point (default target: 2 FP/video)
# Reuses (DO NOT MODIFY):
#   - ~/master-thesis/tools/step5_mine_hard_negatives.py
#
# What this does:
#   1) Reads pipeline/locked_ops/locked_ops.json
#   2) Selects the OP whose 'target_fp_per_video' matches --target-fpv (default 2.0)
#   3) Ensures a TRAIN dump exists at that OP's (iou_dump, conf)
#   4) Runs hard-negative mining on TRAIN to create an overlay dataset
#
# Usage:
#   bash step7_hardneg_mine.sh \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 \
#     [--target-fpv 2.0] \
#     [--device 0] [--workers 16] \
#     [--per-video-cap 50] [--total-cap 20000] \
#     [--suffix hardneg_v1] \
#     [--force-train-dump]
#
# Output:
#   New dataset at: <base_ds>_<suffix> (images/labels with added empty-label negatives)
#   Manifest JSON printed and saved to <out_ds>/hardneg_manifest.json
#
set -euo pipefail

RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
WORKERS=16
TARGET_FPV="2.0"
PER_VIDEO_CAP=50
TOTAL_CAP=20000
SUFFIX="hardneg_v1"
FORCE_TRAIN_DUMP=0

RUN_ROOT="${HOME}/master-thesis/masterThesis"
TOOLS_ROOT="${HOME}/master-thesis/tools"

LOCK_JSON=""
MINE_PY="${TOOLS_ROOT}/step5_mine_hard_negatives.py"

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)            RUN="$2"; shift 2 ;;
    --data)           DATA="$2"; shift 2 ;;
    --imgsz)          IMGSZ="$2"; shift 2 ;;
    --device)         DEVICE="$2"; shift 2 ;;
    --workers)        WORKERS="$2"; shift 2 ;;
    --target-fpv)     TARGET_FPV="$2"; shift 2 ;;
    --per-video-cap)  PER_VIDEO_CAP="$2"; shift 2 ;;
    --total-cap)      TOTAL_CAP="$2"; shift 2 ;;
    --suffix)         SUFFIX="$2"; shift 2 ;;
    --force-train-dump) FORCE_TRAIN_DUMP=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" && -n "$DATA" ]] || { echo "Usage: --run <RUN> --data <DATA_YAML> [--imgsz N]"; exit 2; }

RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

MODEL="${RUN_DIR}/weights/best.pt"
[[ -f "$MODEL" ]] || { echo "Model weights not found: $MODEL"; exit 1; }
[[ -f "$DATA"  ]] || { echo "Dataset YAML not found: $DATA"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
DUMP_DIR="${PIPE_DIR}/dumps"
LOCK_DIR="${PIPE_DIR}/locked_ops"
LOCK_JSON="${LOCK_DIR}/locked_ops.json"
[[ -f "$LOCK_JSON" ]] || { echo "Locked OP spec not found: $LOCK_JSON (run Step 4 first)"; exit 1; }
[[ -f "$MINE_PY"   ]] || { echo "Mining script not found: $MINE_PY"; exit 1; }

# infer dataset base path
DS_BASE="$(awk '/^path:/ {print $2}' "$DATA" | tr -d "'\"")"
[[ -n "$DS_BASE" && -d "$DS_BASE" ]] || { echo "Could not resolve dataset base path from YAML 'path:'"; exit 1; }

echo "==> STEP 7 (Hard-negative mining)"
echo "Run         : $RUN"
echo "Model       : $MODEL"
echo "Data base   : $DS_BASE"
echo "imgsz       : $IMGSZ"
echo "Target FPv  : $TARGET_FPV"
echo "Suffix      : $SUFFIX"
echo "Per-video cap: $PER_VIDEO_CAP   Total cap: $TOTAL_CAP"
echo

# ---------- pick OP for target FP/video ----------
readarray -t OP_JSON < <(python - <<PY
import json,sys
j=json.load(open(sys.argv[1]))
t=float(sys.argv[2])
# exact match first; else closest by absolute diff
ops=j.get("locked_ops", [])
best=None
ops_sorted=sorted(ops, key=lambda o: (abs(float(o.get("target_fp_per_video", 1e9))-t), -float(o.get("conf",0))))
best=ops_sorted[0] if ops_sorted else None
print(json.dumps(best) if best else "")
PY
"$LOCK_JSON" "$TARGET_FPV")

if [[ -z "${OP_JSON[*]}" ]]; then
  echo "No operating point found in ${LOCK_JSON}"; exit 1
fi

# parse selected OP
IOU_DUMP="$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(f"{float(o['iou_dump']):.2f}")
PY <<< "${OP_JSON}")"
CONF_OP="$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(f"{float(o['conf']):.2f}")
PY <<< "${OP_JSON}")"

echo "Selected OP → iou_dump=${IOU_DUMP}  conf=${CONF_OP}"

# ---------- ensure TRAIN dump at this (iou, conf) ----------
TAG="iou${IOU_DUMP}_c${CONF_OP}_train"
OUT_TRAIN="${DUMP_DIR}/${TAG}"
if [[ $FORCE_TRAIN_DUMP -eq 1 ]]; then
  rm -rf "$OUT_TRAIN" || true
fi
if [[ ! -d "$OUT_TRAIN" || ! -s "${OUT_TRAIN}/predictions.json" ]]; then
  echo ">> Creating TRAIN dump at ${TAG}"
  mkdir -p "$DUMP_DIR"
  yolo detect val \
    model="$MODEL" data="$DATA" split=train imgsz="$IMGSZ" \
    device="$DEVICE" workers="$WORKERS" \
    conf="$CONF_OP" iou="$IOU_DUMP" max_det=300 \
    save_txt=True save_conf=True save_json=True \
    project="$DUMP_DIR" name="$TAG" verbose=True || true

  [[ -s "${OUT_TRAIN}/predictions.json" || -d "${OUT_TRAIN}/labels" ]] || {
    echo "!! Failed to create TRAIN dump at ${TAG}"; exit 1; }
else
  echo ">> TRAIN dump exists: ${OUT_TRAIN}"
fi

# ---------- run mining to build overlay dataset ----------
OUT_DS="${DS_BASE}_${SUFFIX}"
if [[ -e "$OUT_DS" ]]; then
  echo "!! Output dataset already exists: $OUT_DS"
  echo "   Choose a different --suffix or remove the directory."
  exit 1
fi

python "$MINE_PY" \
  --base_ds "$DS_BASE" \
  --train_dump "$OUT_TRAIN" \
  --out_ds "$OUT_DS" \
  --imgsz "$IMGSZ" \
  --conf "$CONF_OP" \
  --link_iou "$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(o['link_iou'])
PY <<< "${OP_JSON}")" \
  --max_gap "$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(o['max_gap'])
PY <<< "${OP_JSON}")" \
  --min_len "$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(o['min_len'])
PY <<< "${OP_JSON}")" \
  --match_iou "$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(o['match_iou'])
PY <<< "${OP_JSON}")" \
  --tiny_area_px "$(python - <<PY
import json,sys; o=json.loads(sys.stdin.read()); print(o['tiny_area_px'])
PY <<< "${OP_JSON}")" \
  --per_video_cap "$PER_VIDEO_CAP" \
  --total_cap "$TOTAL_CAP"

echo
echo "==> Step 7 done."
echo "New dataset with hard negatives: $OUT_DS"
echo "Manifest: $OUT_DS/hardneg_manifest.json"
echo "Retrain your next run on this dataset, then redo Steps 1–6."
