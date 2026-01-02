from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import matplotlib.pyplot as plt

def _format_thr(t: float) -> str:
    # compact, readable tick labels
    if t < 0.01:
        return f"{t:.3f}"
    if t < 0.1:
        return f"{t:.2f}"
    return f"{t:.2f}"

def plot_froc_curve(
    *,
    fp_per_image: List[float],
    tpr_loc: List[float],
    out_path: Path,
    title: Optional[str] = "Free-Response ROC (FROC)",
    log_x: bool = False,
    conf_thresholds: Optional[List[float]] = None,
    highlight_conf: Optional[float] = None,   # NEW
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(fp_per_image, tpr_loc)

    ax.set_xlabel("False positives per image (FPPI)")
    ax.set_ylabel("Fraction of GT boxes detected (≥1 match)")

    if title:
        ax.set_title(title)

    if log_x:
        ax.set_xscale("log")

    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Annotate confidence grid info (optional)
    if conf_thresholds:
        note = f"Evaluated at fixed confidence thresholds (n={len(conf_thresholds)})"
        ax.text(0.99, 0.01, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    # NEW: highlight selected operating point
    if highlight_conf is not None and conf_thresholds:
        # find closest threshold (handles float representation issues)
        idx = min(range(len(conf_thresholds)), key=lambda i: abs(conf_thresholds[i] - float(highlight_conf)))
        xh = fp_per_image[idx]
        yh = tpr_loc[idx]
        ax.scatter([xh], [yh], marker="o", s=50, zorder=5, label=f"conf={highlight_conf:g}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def _moving_average(y: List[float], window: int) -> List[float]:
    """
    Simple centered moving average for plotting only.
    Uses edge-clamped windows near boundaries.
    """
    if window <= 1:
        return list(y)
    if window % 2 == 0:
        window += 1  # force odd window
    r = window // 2
    out: List[float] = []
    n = len(y)
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def plot_frame_tp_fp_bars_from_sweep_json(
    *,
    sweep_json: Path,
    out_path: Path,
    title: Optional[str] = "Frame-level TP/FP vs confidence threshold",
) -> Path:
    """
    Grouped bar chart for the curated thresholds:
      x-axis: selected confidence thresholds
      bars: TP_frame and FP_frame (counts)

    Y axis is fixed to [0, 100000] with 20000-step ticks.
    If any value exceeds 100000, the displayed bar is capped at 100000.
    Additionally, for FP at confidence 0.001, the original FP count is annotated next to the bar.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sweep_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    selected: List[Dict[str, Any]] = data.get("selected", [])
    if not selected:
        raise ValueError(
            f"No 'selected' block found in {sweep_json}. "
            "Make sure you called compute_frame_tpr_fpr_sweep(..., selected_thresholds=...)."
        )

    # Preserve the order given by selected_thresholds (already curated)
    thresholds = [float(r["threshold"]) for r in selected]
    tp_raw = [int(r["tp"]) for r in selected]
    fp_raw = [int(r["fp"]) for r in selected]

    # Fixed y-axis setup
    y_max = 100_000
    y_step = 20_000

    # Cap bars for display (counts beyond y_max are visually clipped)
    tp = [min(v, y_max) for v in tp_raw]
    fp = [min(v, y_max) for v in fp_raw]

    x = list(range(len(thresholds)))
    width = 0.42

    fig, ax = plt.subplots()
    ax.bar([i - width / 2 for i in x], tp, width=width, label="TP$_{frame}$")
    ax.bar([i + width / 2 for i in x], fp, width=width, label="FP$_{frame}$")

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Count (frames)")

    if title:
        ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([_format_thr(t) for t in thresholds], rotation=45, ha="right")

    # Fixed y-axis limits and ticks
    ax.set_ylim(0, y_max)
    ax.set_yticks(list(range(0, y_max + y_step, y_step)))

        # Annotate FP total at confidence 0.001 (only this case)
    target_thr = 0.001
    target_idx: Optional[int] = None
    for i, t in enumerate(thresholds):
        if abs(t - target_thr) <= 1e-12:
            target_idx = i
            break

    if target_idx is not None:
        x_fp = target_idx + width / 2
        y_disp = fp[target_idx]          # capped display height
        y_raw = fp_raw[target_idx]       # true FP count

        # If bar is capped, put label clearly *below* the top border (inside axes).
        # Otherwise, put it just above the bar.
        if y_raw > y_max:
            y_text = y_max - 0.06 * y_max   # 6% below top
            va = "top"
        else:
            y_text = y_disp + 0.015 * y_max # slightly above bar
            va = "bottom"

        ax.text(
            x_fp,
            y_text,
            f"{y_raw:,}",                  # thousands separators
            ha="center",
            va=va,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
            clip_on=True,
        )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def plot_frame_two_metric_smooth_from_sweep_json(
    *,
    sweep_json: Path,
    out_path: Path,
    y1_key: str,
    y2_key: str,
    y1_label: str,
    y2_label: str,
    title: str,
    x_min: float | None = None,
    x_max: float = 1.0,
    smooth_window: int = 1,
) -> Path:
    """
    Generic smooth-curve plot for two metrics vs confidence threshold.
    Uses full sweep arrays: data["thresholds"], data[y_key].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sweep_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    thr = [float(t) for t in data.get("thresholds", [])]
    y1 = [float(v) for v in data.get(y1_key, [])]
    y2 = [float(v) for v in data.get(y2_key, [])]
    if not thr or len(y1) != len(thr) or len(y2) != len(thr):
        raise ValueError(f"Invalid or missing arrays in {sweep_json} for keys: {y1_key}, {y2_key}")

    # Default x_min: respect prefiltering if present
    if x_min is None:
        x_min = float(data.get("min_conf_available", 0.001))

    xs: List[float] = []
    a: List[float] = []
    b: List[float] = []
    for t, v1, v2 in zip(thr, y1, y2):
        if float(x_min) <= t <= float(x_max):
            xs.append(t)
            a.append(v1)
            b.append(v2)

    if not xs:
        raise ValueError(f"No points in range [{x_min}, {x_max}] for {sweep_json}")

    a_sm = _moving_average(a, smooth_window)
    b_sm = _moving_average(b, smooth_window)

    fig, ax = plt.subplots()
    ax.plot(xs, a_sm, label=y1_label)
    ax.plot(xs, b_sm, label=y2_label)

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")# choose an informative y-label depending on what we plot
    if y1_key in ("f1", "f2") and y2_key in ("f1", "f2"):
        ylabel = "F-score"
    elif y1_key in ("tpr", "fpr"):
        ylabel = "TPR/FPR Rate"
    elif y2_key in ("sensitivity", "specificity"):
        ylabel = "Sensitivity/Specificity Rate"
    else:
        ylabel = "Score"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_afroc_curve(
    *,
    fpf: List[float],
    tpr_loc: List[float],
    out_path: Path,
    title: Optional[str] = "Alternative Free-Response ROC (AFROC)",
) -> Path:
    """
    Save an AFROC plot:
      x: false positive fraction on negative images (FPF)
      y: fraction of TP decisions with correct localization (lesion-level)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(fpf, tpr_loc)

    ax.set_xlabel("False positive fraction (FPF) on negative images")
    ax.set_ylabel("Fraction of TP decisions with correct localization")

    if title:
        ax.set_title(title)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.margins(x=0.0, y=0.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
