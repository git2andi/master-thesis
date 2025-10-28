#!/usr/bin/env bash
# Step 5 — Final TEST evaluation at locked operating points
# Reuses: ~/master-thesis/tools/step4_eval_test_at_ops.py  (DO NOT MODIFY)
#
# It reads pipeline/locked_ops/locked_ops.json (from Step 4),
# ensures the corresponding TEST dumps exist (optionally),
# and writes your final TEST event-level metrics table.
#
# Usage:
#   bash step5_eval_test_locked_ops.sh \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 \
#     [--ensure-dumps] [--device 0] [--workers 16] \
#     [--px-bins "0,16,32,1e9"] \
#     [--force]
#
# Outputs:
#   ~/master-thesis/masterThesis/<RUN>/pipeline/event_test/test_report.csv
#   ~/master-thesis/masterThesis/<RUN>/pipeline/event_test/test_summary.json
#
set -euo pipefail

# -------- defaults --------
RUN=""
DATA=""
IMGSZ=640
DEVICE="0"
WORKERS=16
ENSURE_DUMPS=0
FORCE=0
PX_BINS="0,16,32,1e9"

RUN_ROOT="${HOME}/master-thesis/masterThesis"
TOOL_ROOT="${HOME}/master-thesis/tools"

EVAL_PY="${TOOL_ROOT}/step4_eval_test_at_ops.py"     # existing python (do not modify)
STEP2_SH="${TOOL_ROOT}/step2_dump_sweeps.sh"         # to create missing TEST dumps if asked

# -------- args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)           RUN="$2"; shift 2 ;;
    --data)          DATA="$2"; shift 2 ;;
    --imgsz)         IMGSZ="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --ensure-dumps)  ENSURE_DUMPS=1; shift 1 ;;
    --force)         FORCE=1; shift 1 ;;
    --px-bins)       PX_BINS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" && -n "$DATA" ]] || { echo "Usage: --run <RUN> --data <DATA_YAML> [--imgsz N] [--ensure-dumps]"; exit 2; }

# -------- resolve paths --------
RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }

PIPE_DIR="${RUN_DIR}/pipeline"
LOCK_DIR="${PIPE_DIR}/locked_ops"
DUMP_DIR="${PIPE_DIR}/dumps"
OUT_DIR="${PIPE_DIR}/event_test"
mkdir -p "$OUT_DIR"

LOCK_JSON="${LOCK_DIR}/locked_ops.json"
[[ -f "$LOCK_JSON" ]] || { echo "Locked OP spec not found: $LOCK_JSON (run Step 4 first)"; exit 1; }

[[ -f "$EVAL_PY"  ]] || { echo "Missing evaluator: $EVAL_PY"; exit 1; }

# infer labels/test from dataset YAML
if [[ -f "$DATA" ]]; then
  DS_BASE="$(awk '/^path:/ {print $2}' "$DATA" | tr -d "'\"")"
else
  DS_BASE=""
fi
LABELS_TEST=""
if [[ -n "$DS_BASE" && -d "$DS_BASE/labels/test" ]]; then
  LABELS_TEST="$DS_BASE/labels/test"
else
  echo "Could not infer labels/test from $DATA (missing 'path:' or directory)."; exit 1
fi

REPORT_CSV="${OUT_DIR}/test_report.csv"
SUMMARY_JSON="${OUT_DIR}/test_summary.json"

if [[ -s "$REPORT_CSV" && $FORCE -eq 0 ]]; then
  echo ">> TEST report already exists: $REPORT_CSV"
  echo "   Use --force to overwrite."
  exit 0
fi

echo "==> STEP 5 (Final TEST at locked OPs)"
echo "Run         : $RUN"
echo "Labels TEST : $LABELS_TEST"
echo "Dumps dir   : $DUMP_DIR"
echo "Locked OPs  : $LOCK_JSON"
echo "imgsz       : $IMGSZ"
echo "Ensure dumps: $ENSURE_DUMPS"
echo "Out         : $REPORT_CSV"
echo

# -------- ensure required TEST dumps (optional) --------
if [[ $ENSURE_DUMPS -eq 1 ]]; then
  [[ -x "$STEP2_SH" ]] || { echo "Missing step2_dump_sweeps.sh at $STEP2_SH"; exit 1; }
  # parse needed IoUs from locked_ops.json
  IOU_LIST="$(python - <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
ious=sorted({float(f"{op['iou_dump']:.2f}") for op in j['locked_ops']})
print(",".join(f"{x:.2f}" for x in ious))
PY
"$LOCK_JSON")"
  echo ">> Ensuring TEST dumps for IoUs: ${IOU_LIST:-<none>}"
  if [[ -n "$IOU_LIST" ]]; then
    bash "$STEP2_SH" \
      --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" \
      --device "$DEVICE" --workers "$WORKERS" \
      --iou-list "$IOU_LIST" --include-test
  fi
fi

# -------- run evaluator on TEST at locked OPs --------
TMP_CSV="${REPORT_CSV}.tmp"
python "$EVAL_PY" \
  --labels_test "$LABELS_TEST" \
  --pred_root   "$DUMP_DIR" \
  --locked_ops  "$LOCK_JSON" \
  --out_csv     "$TMP_CSV" \
  --imgsz       "$IMGSZ" \
  --px_bins     "$PX_BINS"

mv -f "$TMP_CSV" "$REPORT_CSV"

# -------- optional compact summary --------
python - <<'PY' > "$SUMMARY_JSON" 2>/dev/null || true
import csv, json, os, sys
csv_path = sys.argv[1]
out = {"rows": 0, "ops": []}
if os.path.isfile(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline='', encoding='utf-8')))
    out["rows"] = len(rows)
    # extract the two standard OPs if present (2,4 FP/video); otherwise just echo all
    for r in rows:
        try:
            out["ops"].append({
                "tag": r.get("tag"),
                "iou_dump": float(r.get("iou_dump", "nan")),
                "conf": float(r.get("conf", "nan")),
                "sensitivity": float(r.get("sensitivity", "nan")),
                "fp_per_video": float(r.get("fp_per_video", "nan")),
                "rt_med_frames": int(float(r.get("rt_median_frames","-1"))),
                "rt_p90_frames": int(float(r.get("rt_p90_frames","-1"))),
                "sens_bin0": float(r.get("sens_bin0","nan")) if "sens_bin0" in r else None
            })
        except Exception:
            pass
print(json.dumps(out, indent=2))
PY
"$REPORT_CSV"

echo
echo "==> Step 5 done."
echo "TEST table : $REPORT_CSV"
[[ -s "$SUMMARY_JSON" ]] && echo "Summary    : $SUMMARY_JSON" || true
