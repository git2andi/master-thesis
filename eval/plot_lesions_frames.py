#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class LesionRecord:
    unique_id: str
    total_frames: int
    detected_frames: int
    latency_frames: Optional[int]
    histology_class: str


# ----------------------------
# Helpers
# ----------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if not _is_number(x):
        return default
    try:
        return int(round(float(x)))
    except Exception:
        return default


def _as_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        s = str(x).strip()
        return s if s != "" else default
    except Exception:
        return default


def _normalize_histology_label(hist: str) -> str:
    h = (hist or "").strip()
    if h.lower() == "no polyp":
        return "NP"
    return h


def _extract_rows_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    lm = data.get("lesion_metrics")
    if not isinstance(lm, dict):
        return []
    rows = lm.get("rows")
    if isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
        return rows
    return []


def parse_results_json(path: Path) -> List[LesionRecord]:
    """
    Expects:
      data["lesion_metrics"]["rows"] entries with keys:
        unique_id, n_frames_lesion, n_frames_detected, latency_frames, histology_class
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _extract_rows_list(data)
    if not rows:
        lm = data.get("lesion_metrics", {})
        lm_keys = sorted(list(lm.keys())) if isinstance(lm, dict) else []
        raise ValueError(
            f"No lesion rows found in {path}. Expected data['lesion_metrics']['rows'].\n"
            f"lesion_metrics keys: {lm_keys}"
        )

    out: List[LesionRecord] = []
    skipped = 0

    for rec in rows:
        uid = rec.get("unique_id", None)
        total = _as_int(rec.get("n_frames_lesion", None), default=None)
        detected = _as_int(rec.get("n_frames_detected", None), default=None)

        # NEW: latency_frames
        lat = _as_int(rec.get("latency_frames", None), default=None)

        # histology label (normalized later for display)
        hist = _as_str(rec.get("histology_class", None), default="")

        if uid is None or total is None or detected is None:
            skipped += 1
            continue

        total_i = max(0, int(total))
        det_i = max(0, min(int(detected), total_i))  # clamp

        # Normalize: treat negative latencies as missing
        if lat is not None and lat < 0:
            lat = None

        out.append(
            LesionRecord(
                unique_id=str(uid),
                total_frames=total_i,
                detected_frames=det_i,
                latency_frames=lat,
                histology_class=hist,
            )
        )

    if not out:
        ex = rows[0] if rows else {}
        raise ValueError(
            f"Found lesion_metrics.rows in {path}, but could not parse any entries.\n"
            f"Example entry keys: {sorted(list(ex.keys()))}\n"
            f"Skipped rows: {skipped}/{len(rows)}"
        )

    out.sort(key=lambda r: r.unique_id)
    return out


# ----------------------------
# Aggregation across seeds
# ----------------------------

@dataclass(frozen=True)
class LesionAgg:
    unique_id: str
    total_frames: int
    det_values: List[int]
    latency_values: List[Optional[int]]
    histology_class: str

    @property
    def det_mean(self) -> float:
        return sum(self.det_values) / len(self.det_values)

    @property
    def det_min(self) -> int:
        return min(self.det_values)

    @property
    def det_max(self) -> int:
        return max(self.det_values)

    @property
    def missed_mean(self) -> float:
        return self.total_frames - self.det_mean

    @property
    def latency_min(self) -> Optional[int]:
        vals = [v for v in self.latency_values if v is not None]
        return min(vals) if vals else None


def aggregate_seeds(paths: List[Path]) -> List[LesionAgg]:
    per_seed: List[List[LesionRecord]] = [parse_results_json(p) for p in paths]
    all_ids = sorted({lr.unique_id for seed in per_seed for lr in seed})

    seed_maps: List[Dict[str, LesionRecord]] = [{lr.unique_id: lr for lr in seed} for seed in per_seed]

    out: List[LesionAgg] = []
    for uid in all_ids:
        totals: List[int] = []
        dets: List[int] = []
        lats: List[Optional[int]] = []
        hist_values: List[str] = []

        for sm in seed_maps:
            if uid not in sm:
                dets.append(0)
                lats.append(None)
                continue
            lr = sm[uid]
            totals.append(lr.total_frames)
            dets.append(lr.detected_frames)
            lats.append(lr.latency_frames)
            if lr.histology_class:
                hist_values.append(lr.histology_class)

        total_frames = totals[0] if totals else 0
        hist = hist_values[0] if hist_values else ""
        out.append(
            LesionAgg(
                unique_id=uid,
                total_frames=total_frames,
                det_values=dets,
                latency_values=lats,
                histology_class=hist,
            )
        )
    return out


# ----------------------------
# Plotting
# ----------------------------

def plot_lesion_stacked_bars_with_seed_markers(
    *,
    model_name: str,
    lesions: List[LesionAgg],
    out_path: Path,
    title: Optional[str] = None,
    missed_alpha: float = 0.45,
    bar_width: float = 0.8,
    marker_width_frac: float = 0.75,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(lesions)
    x = list(range(1, n + 1))

    det_mean = [la.det_mean for la in lesions]
    missed_mean = [la.missed_mean for la in lesions]

    det_min = [la.det_min for la in lesions]
    det_max = [la.det_max for la in lesions]

    fig_w = 10
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.bar(x, det_mean, width=bar_width, label="Detected (mean)", zorder=1)
    ax.bar(
        x,
        missed_mean,
        bottom=det_mean,
        width=bar_width,
        alpha=missed_alpha,
        label="Missed (mean)",
        zorder=1,
    )

    half_w = (bar_width * marker_width_frac) / 2.0
    for xi, dmin, dmax in zip(x, det_min, det_max):
        ax.hlines(y=dmin, xmin=xi - half_w, xmax=xi + half_w,
                  colors="black", linewidth=1.4, linestyles="dotted", zorder=5)
        ax.hlines(y=dmax, xmin=xi - half_w, xmax=xi + half_w,
                  colors="black", linewidth=1.4, linestyles="dotted", zorder=5)

    # NEW: top labels = latency_frames (min across seeds)
    max_total = max((la.total_frames for la in lesions), default=0)
    y_off = max(0.5, 0.01 * max_total)

    for xi, la, dm, mm in zip(x, lesions, det_mean, missed_mean):
        y_top = dm + mm
        label = str(la.latency_min) if la.latency_min is not None else "—"
        ax.text(xi, y_top + y_off, label, ha="center", va="bottom", fontsize=8, zorder=10)

    ax.set_xlabel("Lesion index")
    ax.set_ylabel("Number of frames")
    ax.set_title(title or f"{model_name}")

    ax.set_xticks(x)

    # NEW: histology line (NO POLYP -> NP)
    ax.set_xticklabels([f"{i}\n{_normalize_histology_label(lesions[i-1].histology_class)}" for i in x])

    ax.plot([], [], color="black", linestyle="dotted", linewidth=1.4,
            label="Detected min/max (across seeds)")
    ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    MODEL_RESULTS: Dict[str, List[Path]] = {
        "Faster R-CNN": [
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s0_b96/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s42_b96/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s123_b96/i00_c00020/results.json"),
        ],
        "YOLOv8": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s0_b208/i00_c00006/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s42_b208/i00_c00006/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s123_b208/i00_c00006/results.json"),
        ],
        "YOLOv11": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s0_b208/i00_c00005/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s42_b208/i00_c00005/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s123_b208/i00_c00005/results.json"),
        ],
        "RT-DETR": [
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s0/i00_c00032/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s42/i00_c00032/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s123/i00_c00032/results.json"),
        ],
    }

    OUT_DIR = Path("./lesion_barplots_seeded")

    for model, paths in MODEL_RESULTS.items():
        if not paths:
            continue
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"{model}: file not found: {p}")

        lesions = aggregate_seeds(paths)

        out_path = OUT_DIR / f"{model.replace(' ', '_').replace('/', '_')}_lesions_seeded.png"
        plot_lesion_stacked_bars_with_seed_markers(
            model_name=model,
            lesions=lesions,
            out_path=out_path,
            missed_alpha=0.45,
        )
        print(f"Wrote: {out_path}  (n_lesions={len(lesions)}; n_seeds={len(paths)})")


if __name__ == "__main__":
    main()
