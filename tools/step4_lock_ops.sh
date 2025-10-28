#!/usr/bin/env bash
# Step 4 — Lock operating points (VAL-selected) at target FP/video
# Reuses your existing: ~/master-thesis/tools/step3_pick_ops.py  (DO NOT MODIFY)
#
# It scans the VAL event-FROC CSVs from Step 3 and picks, for each target FP/video,
# the row closest to the target (tie-break: higher sensitivity, then higher conf).
# The result is a single spec: pipeline/locked_ops/locked_ops.json
#
# Usage:
#   bash step4_lock_ops.sh \
#     --run y11m_224_s42_b512_baseM \
#     [--targets "2,4"] \
#     [--filter-liou 0.30] [--filter-gap 3] [--filter-minlen 2] \
#     [--filter-miou 0.30] [--filter-tapx 64] \
#     [--force]
#
# Outputs:
#   ~/master-thesis/masterThesis/<RUN>/pipeline/locked_ops/locked_ops.json
#
set -euo pipefail

RUN=""
TARGETS="2,4"
FILTER_LIOU=""
FILTER_GAP=""
FILTER_MINLEN=""
FILTER_MIOU=""
FILTER_TAPX=""
FORCE=0

RUN_ROOT="${HOME}/master-thesis/masterThesis"
TOOL_ROOT="${HOME}/master-thesis/tools"
PICK_PY="${TOOL_ROOT}/step3_pick_ops.py"   # reuse existing python

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)            RUN="$2"; shift 2 ;;
    --targets)        TARGETS="$2"; shift 2 ;;
    --filter-liou)    FILTER_LIOU="$2"; shift 2 ;;
    --filter-gap)     FILTER_GAP="$2"; shift 2 ;;
    --filter-minlen)  FILTER_MINLEN="$2"; shift 2 ;;
    --filter-miou)    FILTER_MIOU="$2"; shift 2 ;;
    --filter-tapx)    FILTER_TAPX="$2"; shift 2 ;;
    --force)          FORCE=1; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  endcase
done

[[ -n "$RUN" ]] || { echo "Usage: --run <RUN> [--targets \"2,4\"] [...]"; exit 2; }

RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
CSV_DIR="${PIPE_DIR}/event_val"
OUT_DIR="${PIPE_DIR}/locked_ops"
OUT_JSON="${OUT_DIR}/locked_ops.json"

[[ -d "$CSV_DIR" ]] || { echo "Event-FROC CSV dir not found: $CSV_DIR (run Step 3 first)"; exit 1; }
[[ -f "$PICK_PY"  ]] || { echo "Missing picker: $PICK_PY"; exit 1; }

mkdir -p "$OUT_DIR"

if [[ -s "$OUT_JSON" && $FORCE -eq 0 ]]; then
  echo ">> locked_ops.json already exists: $OUT_JSON"
  echo "   Use --force to overwrite."
  exit 0
fi

echo "==> STEP 4 (Lock operating points)"
echo "Run         : $RUN"
echo "CSV dir     : $CSV_DIR"
echo "Targets     : $TARGETS  (FP/video)"
echo "Filters     : liou=${FILTER_LIOU:-<none>} gap=${FILTER_GAP:-<none>} minlen=${FILTER_MINLEN:-<none>} miou=${FILTER_MIOU:-<none>} tapx=${FILTER_TAPX:-<none>}"
echo "Out         : $OUT_JSON"
echo

# Build python args
PY_ARGS=( --csv_dir "$CSV_DIR" --targets "$TARGETS" --out_json "$OUT_JSON" )
[[ -n "$FILTER_LIOU"   ]] && PY_ARGS+=( --filter_liou "$FILTER_LIOU" )
[[ -n "$FILTER_GAP"    ]] && PY_ARGS+=( --filter_gap "$FILTER_GAP" )
[[ -n "$FILTER_MINLEN" ]] && PY_ARGS+=( --filter_minlen "$FILTER_MINLEN" )
[[ -n "$FILTER_MIOU"   ]] && PY_ARGS+=( --filter_miou "$FILTER_MIOU" )
[[ -n "$FILTER_TAPX"   ]] && PY_ARGS+=( --filter_tapx "$FILTER_TAPX" )

python "$PICK_PY" "${PY_ARGS[@]}"

echo
echo "==> Step 4 done."
echo "Locked OP spec: $OUT_JSON"
