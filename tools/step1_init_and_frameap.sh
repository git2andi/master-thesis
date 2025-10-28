#!/usr/bin/env bash
# Step 1 — Init (meta & sanity) + Frame-level metrics (VAL/TEST)
# Usage:
#   bash step1_init_and_frameap.sh \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 \
#     --device 0 \
#     [--include-test]
#
# Outputs under:
#   ~/master-thesis/masterThesis/<RUN>/pipeline/{meta,frame_ap}/...

set -euo pipefail

# ---------- defaults ----------
RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
INCLUDE_TEST=0
RUN_ROOT="${HOME}/master-thesis/masterThesis"

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)           RUN="$2"; shift 2 ;;
    --data)          DATA="$2"; shift 2 ;;
    --imgsz)         IMGSZ="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --include-test)  INCLUDE_TEST=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" && -n "$DATA" ]] || { echo "Usage: --run <RUN> --data <DATA_YAML> [--imgsz N] [--device 0] [--include-test]"; exit 2; }

RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

MODEL="${RUN_DIR}/weights/best.pt"
[[ -f "$MODEL" ]] || { echo "Model weights not found: $MODEL"; exit 1; }

[[ -f "$DATA" ]] || { echo "Dataset YAML not found: $DATA"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
META_DIR="${PIPE_DIR}/meta"
FRAP_DIR="${PIPE_DIR}/frame_ap"
VAL_DIR="${FRAP_DIR}/val"
TST_DIR="${FRAP_DIR}/test"

mkdir -p "$META_DIR" "$VAL_DIR"
[[ $INCLUDE_TEST -eq 1 ]] && mkdir -p "$TST_DIR"

echo "==> STEP 1 (Init + Frame-AP)"
echo "Run         : $RUN"
echo "Model       : $MODEL"
echo "Data YAML   : $DATA"
echo "imgsz       : $IMGSZ"
echo "Device      : $DEVICE"
echo "Include TEST: $INCLUDE_TEST"
echo "Out dirs    : $PIPE_DIR"

# ---------- meta & sanity ----------
META_JSON="${META_DIR}/meta.json"
SANITY_TXT="${META_DIR}/sanity.txt"

if [[ ! -s "$META_JSON" || ! -s "$SANITY_TXT" ]]; then
  echo ">> Capturing meta + sanity"
  # GPU / system snapshot
  GPU_JSON_TMP="$(mktemp)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits \
      | awk -F, 'BEGIN{print "["}{printf("%s{\"index\":%s,\"name\":\"%s\",\"driver\":\"%s\",\"mem_total\":%s,\"mem_free\":%s}",NR>1?",":"", $1,$2,$3,$4,$5)}END{print "]"}' > "$GPU_JSON_TMP" || true
  else
    echo "[]" > "$GPU_JSON_TMP"
  fi

  # Python env snapshot (torch, ultralytics)
  PY_JSON_TMP="$(mktemp)"
  python - <<'PY' > "$PY_JSON_TMP" 2>/dev/null
import json, sys
info = {
  "python": sys.version.split()[0]
}
try:
  import torch
  info.update({
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_version": torch.version.cuda if hasattr(torch.version,'cuda') else None,
    "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends,'cudnn') else None
  })
except Exception as e:
  info["torch_error"] = str(e)
try:
  import ultralytics
  info["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
except Exception as e:
  info["ultralytics_error"] = str(e)
print(json.dumps(info, indent=2))
PY

  # Dataset sanity (counts)
  # We try to infer base path from YAML 'path:' if present; otherwise trust relative structure
  DS_BASE="$(awk '/^path:/ {print $2}' "$DATA" | tr -d "'\"")"
  {
    echo "# Sanity $(date -Iseconds)"
    echo "HOST: $(hostname)"
    echo "RUN : $RUN"
    echo "DATA: $DATA"
    echo "PATH: ${DS_BASE:-<unset>}"
    if [[ -n "${DS_BASE}" && -d "${DS_BASE}" ]]; then
      for SPL in train val test; do
        IMG="${DS_BASE}/images/${SPL}"
        LBL="${DS_BASE}/labels/${SPL}"
        echo
        echo "[$SPL]"
        [[ -d "$IMG" ]] && echo " images: $(find "$IMG" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | xargs)" || echo " images: MISSING"
        [[ -d "$LBL" ]] && echo " labels: $(find "$LBL" -type f -name '*.txt' | wc -l | xargs)" || echo " labels: MISSING"
      done
    else
      echo " WARNING: dataset base path not found (path: in YAML)."
    fi
  } > "$SANITY_TXT"

  # Compose meta.json
  python - <<PY > "$META_JSON"
import json, os, time, socket, sys
gpu = json.load(open("${GPU_JSON_TMP}"))
py  = json.load(open("${PY_JSON_TMP}"))
meta = {
  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "host": socket.gethostname(),
  "run_name": "${RUN}",
  "run_dir": "${RUN_DIR}",
  "weights": "${MODEL}",
  "data_yaml": "${DATA}",
  "imgsz": ${IMGSZ},
  "device": "${DEVICE}",
  "gpu": gpu,
  "python_env": py
}
print(json.dumps(meta, indent=2))
PY
  rm -f "$GPU_JSON_TMP" "$PY_JSON_TMP"
else
  echo ">> Meta already captured, skipping (${META_JSON})"
fi

# ---------- frame-level VAL ----------
VAL_SUM="${FRAP_DIR}/val_summary.json"
if [[ ! -s "$VAL_SUM" ]]; then
  echo ">> Running frame-level VAL"
  yolo val \
    model="$MODEL" data="$DATA" imgsz="$IMGSZ" device="$DEVICE" \
    project="$VAL_DIR" name="ultra_val" || true

  # Find results.csv produced by Ultralytics in $VAL_DIR/ultra_val/
  VAL_CSV="$(find "$VAL_DIR/ultra_val" -maxdepth 1 -type f -name 'results.csv' | head -n1 || true)"
  python - <<PY > "$VAL_SUM"
import csv, json, os, glob
csv_path = "${VAL_CSV}"
out = {"csv_found": bool(csv_path), "mAP50": None, "mAP50_95": None}
if csv_path and os.path.isfile(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline='')))
    if rows:
        # Ultralytics results.csv last row typically holds final metrics
        last = rows[-1]
        # Keys may vary slightly across versions
        out["mAP50_95"] = float(last.get("metrics/mAP50-95(B)", last.get("map50-95", 0)) or 0)
        out["mAP50"]     = float(last.get("metrics/mAP50(B)",     last.get("map50",     0)) or 0)
out["csv_path"] = csv_path
print(json.dumps(out, indent=2))
PY
else
  echo ">> Frame-level VAL already present (${VAL_SUM})"
fi

# ---------- frame-level TEST (optional) ----------
if [[ $INCLUDE_TEST -eq 1 ]]; then
  TST_SUM="${FRAP_DIR}/test_summary.json"
  if [[ ! -s "$TST_SUM" ]]; then
    echo ">> Running frame-level TEST"
    yolo val \
      model="$MODEL" data="$DATA" imgsz="$IMGSZ" device="$DEVICE" split=test \
      project="$TST_DIR" name="ultra_test" || true

    TST_CSV="$(find "$TST_DIR/ultra_test" -maxdepth 1 -type f -name 'results.csv' | head -n1 || true)"
    python - <<PY > "$TST_SUM"
import csv, json, os
csv_path = "${TST_CSV}"
out = {"csv_found": bool(csv_path), "mAP50": None, "mAP50_95": None}
if csv_path and os.path.isfile(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline='')))
    if rows:
        last = rows[-1]
        out["mAP50_95"] = float(last.get("metrics/mAP50-95(B)", last.get("map50-95", 0)) or 0)
        out["mAP50"]     = float(last.get("metrics/mAP50(B)",     last.get("map50",     0)) or 0)
out["csv_path"] = csv_path
print(json.dumps(out, indent=2))
PY
  else
    echo ">> Frame-level TEST already present (${TST_SUM})"
  fi
fi

echo "==> Step 1 done."
echo "Meta:        $META_JSON"
echo "Sanity:      $SANITY_TXT"
echo "VAL summary: $VAL_SUM"
[[ $INCLUDE_TEST -eq 1 ]] && echo "TEST summary: ${FRAP_DIR}/test_summary.json" || true
