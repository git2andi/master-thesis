#!/usr/bin/env bash
# Orchestrator for the unified REAL-Colon pipeline (Steps 1–8)
# This script ONLY calls the step scripts you already saved; it does not modify them.
#
# Location assumptions (do not change existing stuff):
#   ~/master-thesis/tools/
#     ├─ step1_init_and_frameap.sh
#     ├─ step2_dump_sweeps.sh
#     ├─ step3_event_froc_val.sh
#     ├─ step4_lock_ops.sh
#     ├─ step5_eval_test_locked_ops.sh
#     ├─ step6_speed.sh
#     ├─ step7_hardneg_mine.sh
#     └─ step8_reports.sh
#
# Quick starts:
#   # full pipeline VAL→TEST with dumps+speed, then reports
#   bash pipeline.sh run all \
#     --run y11m_224_s42_b512_baseM \
#     --data /data/local/aschwab/data/realColon_224x224/data.yaml \
#     --imgsz 224 --device 0
#
#   # do steps 1–4 only (no TEST, no speed):
#   bash pipeline.sh run 1-4 --run <RUN> --data <YAML> --imgsz 224 --device 0
#
#   # final TEST + reports (assumes 1–4 done already):
#   bash pipeline.sh run 5,8 --run <RUN> --data <YAML> --imgsz 224 --device 0 --ensure-dumps
#
set -euo pipefail

TOOLS_DIR="${HOME}/master-thesis/tools"

# ---- helpers ----
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
cyan()  { printf "\033[36m%s\033[0m\n" "$*"; }

usage() {
  cat <<EOF
Usage:
  pipeline.sh run <stages> --run <RUN_NAME> --data <DATA_YAML> --imgsz <N> [common flags]

<stages>:
  all            = 1-8
  N              = single step number (1..8)
  A-B            = inclusive range (e.g., 1-5)
  list           = comma list (e.g., 1,2,5,8)

Common flags (apply to relevant steps; safe to pass extra):
  --run <RUN>                     # required
  --data <DATA_YAML>              # required
  --imgsz <N>                     # required (e.g., 224, 640)
  --device <str>                  # default 0
  --workers <int>                 # default 16
  --include-test                  # Step1/2: enable TEST (use sparingly; avoid leakage)
  --conf-dump <0.x>               # Step2: default 0.05
  --iou-list "<list>"             # Step2: default "0.20,0.30,0.40,0.50,0.60"
  --max-det <int>                 # Step2: default 300
  --conf-sweep "<list>"           # Step3: default "0.05,0.10,...,0.90"
  --link-iou <f>                  # Step3/4/5: default 0.30
  --max-gap <int>                 # Step3/4/5: default 3
  --min-len <int>                 # Step3/4/5: default 2
  --match-iou <f>                 # Step3/4/5: default 0.30
  --tiny-area-px <f>              # Step3/4/5: default 64
  --px-bins "<list>"              # Step3/5: default "0,16,32,1e9"
  --targets "<list>"              # Step4: default "2,4"
  --filter-liou <f>               # Step4 filters (optional)
  --filter-gap <int>
  --filter-minlen <int>
  --filter-miou <f>
  --filter-tapx <f>
  --ensure-dumps                  # Step5: auto-create required TEST dumps
  --fp16                          # Step6: detector benchmark FP16
  --fp32                          # Step6: detector benchmark FP32 (default if neither given)
  --e2e-video <path>              # Step6: optional end-to-end timing video
  --draw                          # Step6: draw boxes during e2e timing
  --target-fpv <f>                # Step7: default 2.0 (pick OP near this)
  --per-video-cap <int>           # Step7: default 50
  --total-cap <int>               # Step7: default 20000
  --suffix <str>                  # Step7: default hardneg_v1
  --force                         # steps 4/5 may overwrite their outputs
  --force-train-dump              # Step7: force re-dump of TRAIN at OP

Examples:
  pipeline.sh run all --run R --data D --imgsz 224 --device 0
  pipeline.sh run 1-4 --run R --data D --imgsz 224
  pipeline.sh run 5,6,8 --run R --data D --imgsz 224 --ensure-dumps --fp16 --e2e-video /path/x.mp4
EOF
}

ensure_steps_exist() {
  local missing=0
  for s in 1 2 3 4 5 6 7 8; do
    case $s in
      1) file="${TOOLS_DIR}/step1_init_and_frameap.sh" ;;
      2) file="${TOOLS_DIR}/step2_dump_sweeps.sh" ;;
      3) file="${TOOLS_DIR}/step3_event_froc_val.sh" ;;
      4) file="${TOOLS_DIR}/step4_lock_ops.sh" ;;
      5) file="${TOOLS_DIR}/step5_eval_test_locked_ops.sh" ;;
      6) file="${TOOLS_DIR}/step6_speed.sh" ;;
      7) file="${TOOLS_DIR}/step7_hardneg_mine.sh" ;;
      8) file="${TOOLS_DIR}/step8_reports.sh" ;;
    esac
    [[ -x "$file" ]] || { red "Missing or non-executable: $file"; missing=1; }
  done
  [[ $missing -eq 0 ]] || exit 1
}

parse_stage_spec() {
  local spec="$1"
  if [[ "$spec" == "all" ]]; then
    echo "1 2 3 4 5 6 7 8"; return
  fi
  # commas or single
  if [[ "$spec" =~ , ]]; then
    echo "$spec" | tr ',' ' '; return
  fi
  # range A-B
  if [[ "$spec" =~ ^([1-8])\-([1-8])$ ]]; then
    local a=${BASH_REMATCH[1]} b=${BASH_REMATCH[2]}
    if (( a <= b )); then
      local out=()
      for ((i=a;i<=b;i++)); do out+=("$i"); done
      echo "${out[*]}"; return
    fi
  fi
  # single number
  if [[ "$spec" =~ ^[1-8]$ ]]; then
    echo "$spec"; return
  fi
  red "Invalid stage spec: $spec"; exit 2
}

# ---- subcommand dispatch ----
[[ $# -lt 1 ]] && { usage; exit 2; }
cmd="$1"; shift

case "$cmd" in
  run)
    [[ $# -lt 1 ]] && { usage; exit 2; }
    stages_spec="$1"; shift
    stages=( $(parse_stage_spec "$stages_spec") )
    ensure_steps_exist

    # collect common args
    RUN=""; DATA=""; IMGSZ=""
    DEVICE="0"; WORKERS="16"
    INCLUDE_TEST=0
    CONF_DUMP="0.05"; IOU_LIST="0.20,0.30,0.40,0.50,0.60"; MAXDET="300"
    CONF_SWEEP="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
    LINK_IOU="0.30"; MAX_GAP="3"; MIN_LEN="2"; MATCH_IOU="0.30"; TINY_AREA_PX="64"
    PX_BINS="0,16,32,1e9"
    TARGETS="2,4"
    FILTER_LIOU=""; FILTER_GAP=""; FILTER_MINLEN=""; FILTER_MIOU=""; FILTER_TAPX=""
    ENSURE_DUMPS=0; FORCE=0
    FP16=0; FP32=0; E2E_VIDEO=""; DRAW=0
    TARGET_FPV="2.0"; PER_VIDEO_CAP="50"; TOTAL_CAP="20000"; SUFFIX="hardneg_v1"; FORCE_TRAIN_DUMP=0

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --run) RUN="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --include-test) INCLUDE_TEST=1; shift 1 ;;
        --conf-dump) CONF_DUMP="$2"; shift 2 ;;
        --iou-list) IOU_LIST="$2"; shift 2 ;;
        --max-det) MAXDET="$2"; shift 2 ;;
        --conf-sweep) CONF_SWEEP="$2"; shift 2 ;;
        --link-iou) LINK_IOU="$2"; shift 2 ;;
        --max-gap) MAX_GAP="$2"; shift 2 ;;
        --min-len) MIN_LEN="$2"; shift 2 ;;
        --match-iou) MATCH_IOU="$2"; shift 2 ;;
        --tiny-area-px) TINY_AREA_PX="$2"; shift 2 ;;
        --px-bins) PX_BINS="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --filter-liou) FILTER_LIOU="$2"; shift 2 ;;
        --filter-gap) FILTER_GAP="$2"; shift 2 ;;
        --filter-minlen) FILTER_MINLEN="$2"; shift 2 ;;
        --filter-miou) FILTER_MIOU="$2"; shift 2 ;;
        --filter-tapx) FILTER_TAPX="$2"; shift 2 ;;
        --ensure-dumps) ENSURE_DUMPS=1; shift 1 ;;
        --force) FORCE=1; shift 1 ;;
        --fp16) FP16=1; shift 1 ;;
        --fp32) FP32=1; shift 1 ;;
        --e2e-video) E2E_VIDEO="$2"; shift 2 ;;
        --draw) DRAW=1; shift 1 ;;
        --target-fpv) TARGET_FPV="$2"; shift 2 ;;
        --per-video-cap) PER_VIDEO_CAP="$2"; shift 2 ;;
        --total-cap) TOTAL_CAP="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --force-train-dump) FORCE_TRAIN_DUMP=1; shift 1 ;;
        *) red "Unknown flag: $1"; usage; exit 2 ;;
      esac
    done

    # requireds
    [[ -n "$RUN" && -n "$DATA" && -n "$IMGSZ" ]] || { red "Missing --run / --data / --imgsz"; usage; exit 2; }

    cyan "Running stages: ${stages[*]}  (run=${RUN}, imgsz=${IMGSZ}, device=${DEVICE})"

    for s in "${stages[@]}"; do
      case "$s" in
        1)
          green "[1] Init + Frame-AP (VAL [TEST optional])"
          bash "${TOOLS_DIR}/step1_init_and_frameap.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" \
            --device "$DEVICE" $( (( INCLUDE_TEST )) && echo --include-test )
          ;;
        2)
          green "[2] Broad prediction dumps (VAL; TEST optional)"
          bash "${TOOLS_DIR}/step2_dump_sweeps.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" \
            --device "$DEVICE" --workers "$WORKERS" \
            --conf-dump "$CONF_DUMP" --iou-list "$IOU_LIST" --max-det "$MAXDET" \
            $( (( INCLUDE_TEST )) && echo --include-test )
          ;;
        3)
          green "[3] Event-level FROC on VAL (offline from dumps)"
          bash "${TOOLS_DIR}/step3_event_froc_val.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" \
            --conf-sweep "$CONF_SWEEP" \
            --link-iou "$LINK_IOU" --max-gap "$MAX_GAP" --min-len "$MIN_LEN" \
            --match-iou "$MATCH_IOU" --tiny-area-px "$TINY_AREA_PX" \
            --px-bins "$PX_BINS"
          ;;
        4)
          green "[4] Lock operating points (targets: $TARGETS)"
          bash "${TOOLS_DIR}/step4_lock_ops.sh" \
            --run "$RUN" --targets "$TARGETS" \
            $( [[ -n "$FILTER_LIOU"   ]] && echo --filter-liou "$FILTER_LIOU" ) \
            $( [[ -n "$FILTER_GAP"    ]] && echo --filter-gap "$FILTER_GAP" ) \
            $( [[ -n "$FILTER_MINLEN" ]] && echo --filter-minlen "$FILTER_MINLEN" ) \
            $( [[ -n "$FILTER_MIOU"   ]] && echo --filter-miou "$FILTER_MIOU" ) \
            $( [[ -n "$FILTER_TAPX"   ]] && echo --filter-tapx "$FILTER_TAPX" ) \
            $( (( FORCE )) && echo --force )
          ;;
        5)
          green "[5] Final TEST at locked OPs"
          bash "${TOOLS_DIR}/step5_eval_test_locked_ops.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" \
            $( (( ENSURE_DUMPS )) && echo --ensure-dumps ) \
            --device "$DEVICE" --workers "$WORKERS" \
            --px-bins "$PX_BINS" \
            $( (( FORCE )) && echo --force )
          ;;
        6)
          green "[6] Speed & FPS"
          bash "${TOOLS_DIR}/step6_speed.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" --device "$DEVICE" \
            $( (( FP16 )) && echo --fp16 ) \
            $( (( FP32 )) && echo --fp32 ) \
            $( [[ -n "$E2E_VIDEO" ]] && echo --e2e-video "$E2E_VIDEO" ) \
            $( (( DRAW )) && echo --draw )
          ;;
        7)
          green "[7] Hard-negative mining (optional)"
          bash "${TOOLS_DIR}/step7_hardneg_mine.sh" \
            --run "$RUN" --data "$DATA" --imgsz "$IMGSZ" --device "$DEVICE" --workers "$WORKERS" \
            --target-fpv "$TARGET_FPV" --per-video-cap "$PER_VIDEO_CAP" --total-cap "$TOTAL_CAP" \
            --suffix "$SUFFIX" $( (( FORCE_TRAIN_DUMP )) && echo --force-train-dump )
          ;;
        8)
          green "[8] Reports (CSV + LaTeX)"
          bash "${TOOLS_DIR}/step8_reports.sh" --run "$RUN"
          ;;
      esac
    done
    green "Pipeline finished for run: $RUN"
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    red "Unknown subcommand: $cmd"
    usage
    exit 2
    ;;
esac
