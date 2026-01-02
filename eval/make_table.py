#!/usr/bin/env python3
"""
Read a TSV with columns like:
model, metric, type, mean, std, latex, n, conf_mean, conf_std, iou_mean, iou_std

and write a plain text file containing LaTeX table rows that you can copy-paste directly.

Default output format (one metric per row):
<pretty metric name> & $<latex>$ & $<latex>$ & $<latex>$ & $<latex>$ \\

Where the four columns correspond to (in order):
Faster R-CNN, YOLOv8, YOLOv11, RT-DETR

You can filter by metric prefix (e.g., "frame_metrics.") and/or an explicit metric list.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_MODEL_ORDER = ["Faster R-CNN", "YOLOv8", "YOLOv11", "RT-DETR"]


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"model", "metric", "latex"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    return rows


def normalize_latex_cell(s: str) -> str:
    # Allow either "0.532 \\pm 0.047" or "$0.532 \\pm 0.047$" in TSV.
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def build_index(rows: List[dict]) -> Dict[Tuple[str, str], str]:
    """
    Returns mapping (metric, model) -> latex cell content (without $...$).
    If duplicates exist, last one wins.
    """
    idx: Dict[Tuple[str, str], str] = {}
    for r in rows:
        metric = (r.get("metric") or "").strip()
        model = (r.get("model") or "").strip()
        cell = normalize_latex_cell(r.get("latex", ""))
        if metric and model and cell:
            idx[(metric, model)] = cell
    return idx


def pretty_metric_name(metric: str, mapping: Dict[str, str]) -> str:
    # Use explicit mapping first, else a minimal fallback.
    if metric in mapping:
        return mapping[metric]
    # Fallback: keep suffix after last dot and escape underscores.
    name = metric.split(".")[-1]
    name = name.replace("_", r"\_")
    return name


def write_latex_rows(
    out_path: Path,
    idx: Dict[Tuple[str, str], str],
    metrics: List[str],
    model_order: List[str],
    name_map: Dict[str, str],
) -> None:
    lines: List[str] = []
    for m in metrics:
        row_name = pretty_metric_name(m, name_map)
        cells = []
        for model in model_order:
            cell = idx.get((m, model), "")
            cells.append(f"${cell}$" if cell else "--")
        line = f"{row_name}\n    & " + "\n    & ".join(cells) + r" \\"
        lines.append(line)

    out_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def parse_name_map(path: str | None) -> Dict[str, str]:
    """
    Optional TSV mapping file with columns: metric <tab> name
    """
    if not path:
        return {}
    p = Path(path)
    mp: Dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            metric = row[0].strip()
            name = row[1].strip()
            if metric and name:
                mp[metric] = name
    return mp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path, help="Input .tsv file")
    ap.add_argument("-o", "--out", type=Path, default=Path("latex_rows.txt"), help="Output text file")
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_ORDER),
        help=f"Comma-separated model order (default: {', '.join(DEFAULT_MODEL_ORDER)})",
    )
    ap.add_argument(
        "--metric-prefix",
        type=str,
        default="",
        help="Only include metrics starting with this prefix (e.g., frame_metrics.)",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated explicit metric list (overrides --metric-prefix ordering)",
    )
    ap.add_argument(
        "--name-map",
        type=str,
        default="",
        help="Optional TSV file mapping metric -> pretty row name (columns: metric<TAB>name)",
    )

    args = ap.parse_args()

    rows = read_tsv(args.tsv)
    idx = build_index(rows)
    model_order = [m.strip() for m in args.models.split(",") if m.strip()]
    name_map = parse_name_map(args.name_map or None)

    # Determine metric list
    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        seen = set()
        metrics = []
        for r in rows:
            metric = (r.get("metric") or "").strip()
            if not metric:
                continue
            if args.metric_prefix and not metric.startswith(args.metric_prefix):
                continue
            if metric not in seen:
                seen.add(metric)
                metrics.append(metric)

    if not metrics:
        raise ValueError("No metrics selected. Check --metric-prefix/--metrics.")

    write_latex_rows(args.out, idx, metrics, model_order, name_map)
    print(f"Wrote {len(metrics)} LaTeX rows to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
