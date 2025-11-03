#!/usr/bin/env bash
# Step 2 — Dump broad prediction sweeps (VAL by default, TEST optional)
# Purpose:
#   Produce *superset* detections for later offline sweeps (Step 3).
#   We vary NMS-IoU at dump-time, fix a low conf (e.g., 0.05), and high max_det.
#
# Usage (batch is REQUIRED):
#   bash step2_dump_sweeps.sh \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 \
#     --device "0" \
#     --batch 64 \
#     [--workers 16] \
#     [--conf-dump 0.05] [--iou-list "0.20,0.30,0.40,0.50,0.60"] [--max-det 300] \
#     [--include-test]
#
# Outputs (per run):
#   ~/master-thesis/masterThesis/<RUN>/pipeline/dumps/
#     iou0.20_c0.05_val/{labels/,predictions.json,...}
#     iou0.30_c0.05_val/{...}
#     [optional TEST mirrors]
#
set -euo pipefail

# -------- defaults --------
RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
BATCH=""
WORKERS=16
CONF_DUMP=0.05
IOU_LIST="0.20,0.30,0.40,0.50,0.60"
MAXDET=300
INCLUDE_TEST=0

RUN_ROOT="${HOME}/master-thesis/masterThesis"

# -------- args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)         RUN="$2"; shift 2 ;;
    --data)        DATA="$2"; shift 2 ;;
    --imgsz)       IMGSZ="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --batch)       BATCH="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --conf-dump)   CONF_DUMP="$2"; shift 2 ;;
    --iou-list)    IOU_LIST="$2"; shift 2 ;;
    --max-det)     MAXDET="$2"; shift 2 ;;
    --include-test) INCLUDE_TEST=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# -------- required args check --------
if [[ -z "$RUN" || -z "$DATA" || -z "$BATCH" ]]; then
  echo "Usage: --run <RUN> --data <DATA_YAML> --imgsz N --device ID --batch N [--workers N] [--conf-dump F] [--iou-list list] [--max-det N] [--include-test]" >&2
  exit 2
fi
# numeric check for batch
if ! [[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -gt 0 ]]; then
  echo "Error: --batch must be a positive integer (got '$BATCH')." >&2
  exit 2
fi

RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

MODEL="${RUN_DIR}/weights/best.pt"
[[ -f "$MODEL" ]] || { echo "Model weights not found: $MODEL"; exit 1; }
[[ -f "$DATA"  ]] || { echo "Dataset YAML not found: $DATA"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
DUMP_DIR="${PIPE_DIR}/dumps"
mkdir -p "$DUMP_DIR"

echo "==> STEP 2 (Broad prediction dumps)"
echo "Run       : $RUN"
echo "Model     : $MODEL"
echo "Data YAML : $DATA"
echo "imgsz     : $IMGSZ"
echo "Device    : $DEVICE"
echo "Batch     : $BATCH"
echo "Workers   : $WORKERS"
echo "IoU list  : $IOU_LIST"
echo "Conf dump : $CONF_DUMP"
echo "Max det   : $MAXDET"
echo "Include T : $INCLUDE_TEST"
echo "Out dir   : $DUMP_DIR"
echo

# Split IoU list
IFS=',' read -r -a IARRAY <<< "$IOU_LIST"

# Function to dump one split (val or test) at one IoU
dump_split() {
  local IOU="$1"
  local SPLIT="$2"  # val | test
  local TAG="iou$(printf '%.2f' "$IOU")_c$(printf '%.2f' "$CONF_DUMP")_${SPLIT}"
  local OUT="${DUMP_DIR}/${TAG}"

  if [[ -d "$OUT" && -s "${OUT}/predictions.json" ]]; then
    echo ">> skip: ${TAG} (exists)"
    return 0
  fi

  echo ">> ${SPLIT}: IoU=${IOU}  conf=${CONF_DUMP}  max_det=${MAXDET}  -> ${TAG}"
  if ! yolo detect val \
    model="$MODEL" \
    data="$DATA" \
    split="$SPLIT" \
    imgsz="$IMGSZ" \
    device="$DEVICE" \
    workers="$WORKERS" \
    batch="$BATCH" \
    conf="$CONF_DUMP" \
    iou="$IOU" \
    max_det="$MAXDET" \
    save_txt=True \
    save_conf=True \
    save_json=True \
    project="$DUMP_DIR" \
    name="$TAG" \
    verbose=True
  then
    echo "!! YOLO val failed for ${TAG}" >&2
  fi
}

# Always VAL; TEST optional
status=0
for IOU in "${IARRAY[@]}"; do
  dump_split "$IOU" "val" || status=1
  if [[ $INCLUDE_TEST -eq 1 ]]; then
    dump_split "$IOU" "test" || status=1
  fi
done

if [[ $status -ne 0 ]]; then
  echo "==> Step 2 completed with warnings/failures (see above)."
  exit 1
fi

echo
echo "==> Step 2 done. Example VAL dump:"
echo "    ${DUMP_DIR}/iou0.50_c$(printf '%.2f' "$CONF_DUMP")_val/predictions.json"
echo "Next: Step 3 will compute VAL event-level FROC offline from these dumps."
