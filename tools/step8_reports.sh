#!/usr/bin/env bash
# Step 8 — Aggregate reports (CSV + LaTeX) for a run
#
# Inputs (created by prior steps):
#   pipeline/meta/meta.json
#   pipeline/frame_ap/val_summary.json
#   pipeline/frame_ap/test_summary.json       (optional)
#   pipeline/event_test/test_report.csv       (Step 5)
#   pipeline/speed/detector_summary.json      (Step 6a)
#   pipeline/speed/e2e_latency.json           (Step 6b, optional)
#
# Outputs:
#   pipeline/reports/
#     tables_frame_ap.{csv,tex}
#     tables_event_test.{csv,tex}
#     tables_latency.{csv,tex}
#     meta_hardware.tex
#     README.md
#
# Usage:
#   bash step8_reports.sh --run y11m_224_s42_b512_baseM
#
set -euo pipefail

RUN=""
RUN_ROOT="${HOME}/master-thesis/masterThesis"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN" ]] || { echo "Usage: --run <RUN_NAME>"; exit 2; }

RUN_DIR="${RUN_ROOT}/${RUN}"
PIPE_DIR="${RUN_DIR}/pipeline"
META_JSON="${PIPE_DIR}/meta/meta.json"
VAL_SUM="${PIPE_DIR}/frame_ap/val_summary.json"
TST_SUM="${PIPE_DIR}/frame_ap/test_summary.json"
TEST_EVT_CSV="${PIPE_DIR}/event_test/test_report.csv"
DET_SUM="${PIPE_DIR}/speed/detector_summary.json"
E2E_JSON="${PIPE_DIR}/speed/e2e_latency.json"
OUT_DIR="${PIPE_DIR}/reports"
mkdir -p "$OUT_DIR"

# -------- helpers: soft checks --------
missing=0
for f in "$META_JSON" "$VAL_SUM" "$TEST_EVT_CSV" "$DET_SUM"; do
  if [[ ! -s "$f" ]]; then echo "!! missing required: $f"; missing=1; fi
done
if [[ $missing -eq 1 ]]; then
  echo "Some required inputs are missing. Ensure Steps 1,5,6 ran."
  exit 1
fi

# -------- Frame-AP table --------
python - <<'PY'
import json, csv, os, sys, pathlib
pipe = pathlib.Path(os.environ["PIPE_DIR"])
val = json.load(open(pipe/"frame_ap/val_summary.json"))
tstp = pipe/"frame_ap/test_summary.json"
test = json.load(open(tstp)) if tstp.exists() and os.path.getsize(tstp)>0 else None
out_csv = pipe/"reports/tables_frame_ap.csv"
rows = []
rows.append({"Split":"VAL","mAP50":val.get("mAP50"),"mAP50-95":val.get("mAP50_95")})
if test is not None:
    rows.append({"Split":"TEST","mAP50":test.get("mAP50"),"mAP50-95":test.get("mAP50_95")})
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Split","mAP50","mAP50-95"])
    w.writeheader(); w.writerows(rows)

# LaTeX
out_tex = pipe/"reports/tables_frame_ap.tex"
with open(out_tex, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[t]\\centering\\small\n")
    f.write("\\caption{Frame-level detection on REAL-Colon.}\n")
    f.write("\\begin{tabular}{lcc}\\toprule\nSplit & mAP@0.5 & mAP@0.5:0.95\\\\\\midrule\n")
    for r in rows:
        f.write(f"{r['Split']} & {r['mAP50']:.3f} & {r['mAP50-95']:.3f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\\label{tab:frame_ap}\\end{table}\n")
print(f"[frame_ap] wrote: {out_csv} and {out_tex}")
PY
PIPE_DIR="$PIPE_DIR"

# -------- Event-level TEST @ locked OPs (2 & 4 FP/video if present) --------
python - <<'PY'
import csv, json, os, pathlib, math
pipe = pathlib.Path(os.environ["PIPE_DIR"])
csv_path = pipe/"event_test/test_report.csv"
out_csv = pipe/"reports/tables_event_test.csv"
rows=list(csv.DictReader(open(csv_path, newline='', encoding='utf-8')))
# keep all OPs; later you can select 2 & 4 in the text
fields = ["tag","iou_dump","conf","sensitivity","fp_per_video","rt_median_frames","rt_p90_frames"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k:r.get(k) for k in fields})

# LaTeX
def fmt(x, nd=3):
    try:
        v=float(x)
        if math.isnan(v): return "--"
        return f"{v:.{nd}f}"
    except: return str(x)
out_tex = pipe/"reports/tables_event_test.tex"
with open(out_tex, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[t]\\centering\\small\n")
    f.write("\\caption{Event-level results on TEST at VAL-selected operating points. Sensitivity and false positives per video (FP/v), plus reaction time (RT).}\n")
    f.write("\\begin{tabular}{lcccccc}\\toprule\n")
    f.write("OP tag & IoU$_{NMS}$ & Conf & Sens. & FP/v & RT$_{med}$ (f) & RT$_{p90}$ (f)\\\\\\midrule\n")
    for r in rows:
        f.write(f"{r.get('tag','')} & {fmt(r.get('iou_dump'),2)} & {fmt(r.get('conf'),2)} & {fmt(r.get('sensitivity'))} & {fmt(r.get('fp_per_video'))} & {fmt(r.get('rt_median_frames'),0)} & {fmt(r.get('rt_p90_frames'),0)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\\label{tab:event_test}\\end{table}\n")
print(f"[event_test] wrote: {out_csv} and {out_tex}")
PY

# -------- Latency/FPS table --------
python - <<'PY'
import json, csv, os, pathlib, math
pipe = pathlib.Path(os.environ["PIPE_DIR"])
det = json.load(open(pipe/"speed/detector_summary.json"))
det_row = {
  "Mode":"Detector (batch=1)",
  "Format": det.get("format",""),
  "imgsz": det.get("imgsz",""),
  "mean_ms": det.get("mean_ms"),
  "p50_ms": det.get("p50_ms"),
  "p90_ms": det.get("p90_ms"),
  "FPS": det.get("fps")
}
rows=[det_row]
e2e = pipe/"speed/e2e_latency.json"
if e2e.exists() and os.path.getsize(e2e)>0:
    j=json.load(open(e2e))
    rows.append({
      "Mode":"End-to-end (video)",
      "Format":"decode+det[+draw]",
      "imgsz": j.get("imgsz"),
      "mean_ms": j.get("mean_ms"),
      "p50_ms": j.get("p50_ms"),
      "p90_ms": j.get("p90_ms"),
      "FPS": j.get("fps")
    })

# CSV
out_csv = pipe/"reports/tables_latency.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Mode","Format","imgsz","mean_ms","p50_ms","p90_ms","FPS"])
    w.writeheader(); w.writerows(rows)

# LaTeX
def fmt(v, nd=1):
    try:
        if v is None or (isinstance(v,float) and math.isnan(v)): return "--"
        return f"{float(v):.{nd}f}"
    except: return str(v)
out_tex = pipe/"reports/tables_latency.tex"
with open(out_tex, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[t]\\centering\\small\n")
    f.write("\\caption{Latency and throughput (batch=1) at deployment image size.}\n")
    f.write("\\begin{tabular}{lcccccc}\\toprule\n")
    f.write("Mode & Format & Size & Mean (ms) & p50 (ms) & p90 (ms) & FPS\\\\\\midrule\n")
    for r in rows:
        f.write(f"{r['Mode']} & {r['Format']} & {r['imgsz']} & {fmt(r['mean_ms'])} & {fmt(r['p50_ms'])} & {fmt(r['p90_ms'])} & {fmt(r['FPS'])}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\\label{tab:latency}\\end{table}\n")
print(f"[latency] wrote: {out_csv} and {out_tex}")
PY

# -------- Meta hardware snippet (LaTeX) --------
python - <<'PY'
import json, os, pathlib
pipe = pathlib.Path(os.environ["PIPE_DIR"])
j=json.load(open(pipe/"meta/meta.json"))
gpu = j.get("gpu", [])
torch = j.get("python_env", {}).get("torch","?")
cuda = j.get("python_env", {}).get("cuda_version","?")
ultra = j.get("python_env", {}).get("ultralytics","?")
host = j.get("host","?")
rows=[]
for g in gpu:
    rows.append(f"{g.get('name','GPU')} ({g.get('mem_total','?')} MB)")
gpus = ", ".join(rows) if rows else "N/A"
out = pipe/"reports/meta_hardware.tex"
with open(out,"w",encoding="utf-8") as f:
    f.write("% Auto-generated hardware/software snippet\n")
    f.write("\\noindent\\textbf{Hardware/Software.} ")
    f.write(f"Experiments ran on {gpus}; CUDA {cuda}; PyTorch {torch}; Ultralytics {ultra}. Host: {host}.\n")
print(f"[meta] wrote: {out}")
PY

# -------- README --------
cat > "${OUT_DIR}/README.md" <<EOF
# Reports for ${RUN}

Artifacts:
- \`tables_frame_ap.csv|tex\` — frame-level AP (VAL/TEST)
- \`tables_event_test.csv|tex\` — TEST event-level metrics at locked OPs
- \`tables_latency.csv|tex\` — detector and end-to-end latency/FPS
- \`meta_hardware.tex\` — LaTeX snippet for hardware/software
- Inputs: see \`../meta\`, \`../frame_ap\`, \`../event_test\`, \`../speed\`
EOF

echo "==> Step 8 done."
echo "Reports in: ${OUT_DIR}"
