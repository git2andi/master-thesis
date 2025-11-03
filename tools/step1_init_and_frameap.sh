#!/usr/bin/env bash
# Step 1 — Init (meta & sanity) + Frame-level metrics (VAL/TEST via Python API)
# Usage (batch is REQUIRED):
#   bash step1_init_and_frameap.sh \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 \
#     --device 0 \
#     --batch 64 \
#     [--include-test]

set -euo pipefail

# ---------- defaults ----------
RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
BATCH=""                  # required; validated below
INCLUDE_TEST=0
RUN_ROOT="${HOME}/master-thesis/masterThesis"

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)           RUN="$2"; shift 2 ;;
    --data)          DATA="$2"; shift 2 ;;
    --imgsz)         IMGSZ="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --batch)         BATCH="$2"; shift 2 ;;   # REQUIRED
    --include-test)  INCLUDE_TEST=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------- required args check ----------
if [[ -z "$RUN" || -z "$DATA" || -z "$BATCH" ]]; then
  echo "Usage: --run <RUN> --data <DATA_YAML> --imgsz N --device ID --batch N [--include-test]" >&2
  exit 2
fi
# numeric check for batch (positive integer)
if ! [[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -gt 0 ]]; then
  echo "Error: --batch must be a positive integer (got '$BATCH')." >&2
  exit 2
fi

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
echo "Batch       : $BATCH"     # required
echo "Include TEST: $INCLUDE_TEST"
echo "Out dirs    : $PIPE_DIR"

# ---------- meta & sanity ----------
META_JSON="${META_DIR}/meta.json"
SANITY_TXT="${META_DIR}/sanity.txt"

if [[ ! -s "$META_JSON" || ! -s "$SANITY_TXT" ]]; then
  echo ">> Capturing meta + sanity"
  GPU_JSON_TMP="$(mktemp)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits \
      | awk -F, 'BEGIN{print "["}{printf("%s{\"index\":%s,\"name\":\"%s\",\"driver\":\"%s\",\"mem_total\":%s,\"mem_free\":%s}",NR>1?",":"", $1,$2,$3,$4,$5)}END{print "]"}' > "$GPU_JSON_TMP" || true
  else
    echo "[]" > "$GPU_JSON_TMP"
  fi

  PY_JSON_TMP="$(mktemp)"
  python - <<'PY' > "$PY_JSON_TMP" 2>/dev/null
import json, sys
info = {"python": sys.version.split()[0]}
try:
  import torch
  info.update({
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_version": getattr(torch.version, "cuda", None),
    "cudnn_version": getattr(getattr(torch.backends,"cudnn",None), "version", lambda: None)()
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

  DS_BASE="$(awk '/^[[:space:]]*path:/ {print $2}' "$DATA" | tr -d "'\"")"
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

  python - <<PY > "$META_JSON"
import json, os, time, socket
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
  "batch": ${BATCH},           # required → write the integer
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
  echo ">> Running frame-level VAL (Python API)"
  python - <<PY > "$VAL_SUM"
from ultralytics import YOLO
import json
model = YOLO(r"${MODEL}")
res = model.val(
    data=r"${DATA}",
    imgsz=${IMGSZ},
    device=r"${DEVICE}",
    split="val",
    batch=${BATCH},             # required → pass as int
    project=r"${VAL_DIR}",
    name="ultra_val",
    save_json=False,
    verbose=True
)
out = {
  "results": getattr(res, "results_dict", {}),
  "speed": getattr(res, "speed", None),
  "save_dir": str(getattr(res, "save_dir", "")),
}
print(json.dumps(out, indent=2))
PY
else
  echo ">> Frame-level VAL already present (${VAL_SUM})"
fi

# ---------- frame-level TEST (optional) ----------
if [[ $INCLUDE_TEST -eq 1 ]]; then
  TST_SUM="${FRAP_DIR}/test_summary.json"
  mkdir -p "$TST_DIR"
  if [[ ! -s "$TST_SUM" ]]; then
    echo ">> Running frame-level TEST (Python API)"
    python - <<PY > "$TST_SUM"
from ultralytics import YOLO
import json
model = YOLO(r"${MODEL}")
res = model.val(
    data=r"${DATA}",
    imgsz=${IMGSZ},
    device=r"${DEVICE}",
    split="test",
    batch=${BATCH},             # required → pass as int
    project=r"${TST_DIR}",
    name="ultra_test",
    save_json=False,
    verbose=False
)
out = {
  "results": getattr(res, "results_dict", {}),
  "speed": getattr(res, "speed", None),
  "save_dir": str(getattr(res, "save_dir", "")),
}
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
