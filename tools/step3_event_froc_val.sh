#!/usr/bin/env bash
# Step 3 — Event-level FROC on VAL from Step-2 dumps (no model re-run)
# Reuses: ~/master-thesis/tools/event_eval/step2_event_froc.py
set -euo pipefail

# --- defaults ---
RUN=""
DATA=""
IMGSZ=640
CONF_SWEEP="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
LINK_IOU=0.30
MAX_GAP=3
MIN_LEN=2
MATCH_IOU=0.30
TINY_AREA_PX=64
PX_BINS="0,16,32,1e9"

RUN_ROOT="${HOME}/master-thesis/masterThesis"
TOOL_ROOT="${HOME}/master-thesis/tools"
EVAL_PY="${TOOL_ROOT}/event_eval/step2_event_froc.py"
LABELS_VAL=""

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --conf-sweep) CONF_SWEEP="$2"; shift 2;;
    --link-iou) LINK_IOU="$2"; shift 2;;
    --max-gap) MAX_GAP="$2"; shift 2;;
    --min-len) MIN_LEN="$2"; shift 2;;
    --match-iou) MATCH_IOU="$2"; shift 2;;
    --tiny-area-px) TINY_AREA_PX="$2"; shift 2;;
    --px-bins) PX_BINS="$2"; shift 2;;
    -l|--labels-val) LABELS_VAL="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n "$RUN" && -n "$DATA" ]] || { echo "Usage: --run <RUN> --data <DATA_YAML> [...]"; exit 2; }

# --- paths ---
RUN_DIR="${RUN_ROOT}/${RUN}"
[[ -d "$RUN_DIR" ]] || { echo "Run dir not found: $RUN_DIR"; exit 1; }
PIPE_DIR="${RUN_DIR}/pipeline"
DUMP_DIR="${PIPE_DIR}/dumps"
OUT_DIR="${PIPE_DIR}/event_val"
mkdir -p "$OUT_DIR"
[[ -d "$DUMP_DIR" ]] || { echo "Dump dir not found: $DUMP_DIR (run Step 2 first)"; exit 1; }
[[ -f "$EVAL_PY"  ]] || { echo "Missing evaluator: $EVAL_PY"; exit 1; }

# --- resolve GT labels ---
if [[ -z "$LABELS_VAL" ]]; then
  DS_BASE="$(awk '/^[[:space:]]*path:/ {print $2}' "$DATA" 2>/dev/null | tr -d "'\"")"
  if [[ -n "${DS_BASE:-}" && -d "$DS_BASE/labels/val" ]]; then
    LABELS_VAL="$DS_BASE/labels/val"
  else
    VAL_IMG="$(awk '/^[[:space:]]*val:/ {print $2}' "$DATA" 2>/dev/null | tr -d "'\"")"
    if [[ -n "${VAL_IMG:-}" && -d "$VAL_IMG" ]]; then
      LABELS_VAL="${VAL_IMG/\/images\//\/labels\/}"
      case "$LABELS_VAL" in */val|*/val/) :;; *) LABELS_VAL="${LABELS_VAL%/}/val";; esac
    fi
  fi
fi
[[ -n "$LABELS_VAL" && -d "$LABELS_VAL" ]] || { echo "Could not resolve GT labels. Provide -l /path/to/labels/val or add 'path:' to $DATA."; exit 1; }

# --- environment ---
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${TOOL_ROOT}/event_eval:${PYTHONPATH:-}"

# --- header ---
echo "==> STEP 3 (Event-FROC on VAL)"
echo "Run        : $RUN"
echo "Labels VAL : $LABELS_VAL"
echo "Dumps dir  : $DUMP_DIR"
echo "Out dir    : $OUT_DIR"
echo "Params     : imgsz=$IMGSZ  link_iou=$LINK_IOU  max_gap=$MAX_GAP  min_len=$MIN_LEN  match_iou=$MATCH_IOU  tiny_area_px=$TINY_AREA_PX"
echo "             conf_sweep=$CONF_SWEEP  px_bins=$PX_BINS"
python - <<'__PY__' || true
import sys, platform
try:
    import ultralytics as u; v=getattr(u,"__version__","?")
except Exception:
    v="n/a"
print(f"[py] {sys.version.split()[0]}  ultralytics={v}  platform={platform.system()}-{platform.machine()}")
__PY__
echo

# --- enumerate dumps ---
mapfile -t DUMPS < <(find "$DUMP_DIR" -maxdepth 1 -mindepth 1 -type d -name 'iou*_c*_val' | sort)
TOTAL=${#DUMPS[@]}
echo "Found dumps : $TOTAL"
(( TOTAL > 0 )) || { echo "No VAL dumps found in: $DUMP_DIR"; exit 1; }

# timing wrapper
if [[ -x /usr/bin/time ]]; then
  TIME_CMD=(/usr/bin/time -f "    done in %E, maxRSS=%MkB")
else
  TIME_CMD=(bash -c 's=$SECONDS; shift; "$@"; rc=$?; d=$((SECONDS-s)); printf "    done in %02d:%02d, maxRSS=?\n" $((d/60)) $((d%60))); exit $rc' _)
fi

# --- iterate ---
status=0
i=0
for D in "${DUMPS[@]}"; do
  i=$((i+1))
  D="${D%$'\r'}"
  tag="$(basename "$D")"

  # prefer dump labels/ (faster); fallback to predictions.json
  PJ_DIR="$D"; PJ_JSON="$D/predictions.json"; SRC="labels"
  if [[ ! -d "$D/labels" && -f "$PJ_JSON" ]]; then PJ_DIR="$PJ_JSON"; SRC="json"; fi

  sz="-"; if [[ -f "$PJ_JSON" ]]; then sz="$(du -h "$PJ_JSON" 2>/dev/null | awk '{print $1}')"; fi
  mt="?"; if [[ -f "$PJ_JSON" ]]; then ts="$(date -r "$PJ_JSON" '+%F %T' 2>/dev/null || true)"; [[ -n "${ts:-}" ]] && mt="$ts"; fi

  out_csv="${OUT_DIR}/${tag}_liou$(printf '%.2f' "$LINK_IOU")_gap${MAX_GAP}_ml${MIN_LEN}_miou$(printf '%.2f' "$MATCH_IOU")_tapx${TINY_AREA_PX}_event_froc_val.csv"
  if [[ -s "$out_csv" ]]; then
    printf "[%d/%d] %-22s  %6s  %s  (%s) -> skip (exists)\n" "$i" "$TOTAL" "$tag" "$sz" "$mt" "$SRC"
    continue
  fi
  if [[ ! -d "$D/labels" && ! -f "$PJ_JSON" ]]; then
    printf "[%d/%d] %-22s  %6s  %s  -> skip (no predictions)\n" "$i" "$TOTAL" "$tag" "$sz" "$mt"
    continue
  fi

  printf "[%d/%d] %-22s  %6s  %s  (%s) -> running...\n" "$i" "$TOTAL" "$tag" "$sz" "$mt" "$SRC"

  if "${TIME_CMD[@]}" python -u "$EVAL_PY" \
      --labels "$LABELS_VAL" \
      --pred_json "$PJ_DIR" \
      --out_csv "$out_csv" \
      --imgsz "$IMGSZ" \
      --conf_sweep "$CONF_SWEEP" \
      --link_iou "$LINK_IOU" \
      --max_gap "$MAX_GAP" \
      --min_len "$MIN_LEN" \
      --match_iou "$MATCH_IOU" \
      --tiny_area_px "$TINY_AREA_PX" \
      --px_bins "$PX_BINS" \
      --progress
  then
    :
  else
    echo "!! Failed on $tag"
    status=1
  fi


done

# --- summary ---
OUT_DIR_EXPORT="$OUT_DIR" python - <<'__PY__' > "$OUT_DIR/summary.json" 2>/dev/null || true
import csv, json, glob, os, re
out_dir = os.environ["OUT_DIR_EXPORT"]
out=[]
for f in sorted(glob.glob(os.path.join(out_dir, "*_event_froc_val.csv"))):
    try:
        with open(f, newline='', encoding='utf-8') as fh:
            meta = fh.readline().strip()
            auc = None
            m = re.search(r'auc0_4=([0-9.]+)', meta)
            if m: auc = float(m.group(1))
            rows = list(csv.DictReader(fh))
            out.append({"file": os.path.basename(f), "auc0_4": auc, "n_rows": len(rows)})
    except Exception:
        pass
out.sort(key=lambda x: (-(x["auc0_4"] or -1), -x["n_rows"]))
print(json.dumps({"event_val_files": out}, indent=2))
__PY__

echo
echo "==> Step 3 done."
echo "CSVs in: $OUT_DIR"
[[ -s "$OUT_DIR/summary.json" ]] && echo "Summary: $OUT_DIR/summary.json" || true
exit $status
