from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data containers + loading
# -----------------------------

@dataclass(frozen=True)
class CurveData:
    path: Path
    iou_thr: float
    conf_thresholds: List[float]
    # FROC payload
    froc_fp_per_image: Optional[List[float]] = None
    froc_tpr_loc: Optional[List[float]] = None
    # AFROC payload
    afroc_fpf: Optional[List[float]] = None
    afroc_tpr_loc: Optional[List[float]] = None


def _load_curve_json(p: Path) -> CurveData:
    if not p.exists():
        raise FileNotFoundError(f"curve.json not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    iou_thr = float(data.get("iou_thr", 0.0))

    conf_thr = data.get("conf_thresholds", None)
    if conf_thr is None:
        raise ValueError(
            f"{p} does not contain 'conf_thresholds'. "
            "This script expects fixed-confidence FROC/AFROC outputs."
        )
    conf_thresholds = [float(x) for x in conf_thr]

    def _extract_payload(root: Any) -> Dict[str, Any]:
        if not isinstance(root, dict):
            return {}
        if "payload" in root and isinstance(root["payload"], dict):
            return root["payload"]
        return root

    # FROC
    froc_fp = None
    froc_y = None
    if "froc" in data:
        payload = _extract_payload(data["froc"])
        if "fp_per_image" in payload and "tpr_loc" in payload:
            froc_fp = [float(v) for v in payload["fp_per_image"]]
            froc_y = [float(v) for v in payload["tpr_loc"]]

    # AFROC
    afroc_x = None
    afroc_y = None
    if "afroc" in data:
        payload = _extract_payload(data["afroc"])
        if "fpf" in payload and "tpr_loc" in payload:
            afroc_x = [float(v) for v in payload["fpf"]]
            afroc_y = [float(v) for v in payload["tpr_loc"]]

    return CurveData(
        path=p,
        iou_thr=iou_thr,
        conf_thresholds=conf_thresholds,
        froc_fp_per_image=froc_fp,
        froc_tpr_loc=froc_y,
        afroc_fpf=afroc_x,
        afroc_tpr_loc=afroc_y,
    )


def _nearest_index(xs: List[float], x: float) -> int:
    if not xs:
        return 0
    x = float(x)
    return min(range(len(xs)), key=lambda i: abs(xs[i] - x))


def _assert_same_conf_thresholds(curves: List[CurveData], model_name: str) -> List[float]:
    if not curves:
        return []
    base = curves[0].conf_thresholds
    for c in curves[1:]:
        if len(c.conf_thresholds) != len(base) or any(abs(a - b) > 1e-12 for a, b in zip(c.conf_thresholds, base)):
            raise ValueError(
                f"[{model_name}] conf_thresholds mismatch across seeds.\n"
                f"  first: {curves[0].path}\n"
                f"  other: {c.path}\n"
                "Make sure all were computed with the same conf grid."
            )
    return base


def _pick_iou_for_title(model_curves: Dict[str, List[CurveData]]) -> float:
    all_iou = sorted({float(c.iou_thr) for curves in model_curves.values() for c in curves})
    if len(all_iou) > 1:
        print(f"[WARN] Multiple IoU thresholds found in inputs: {all_iou}. Using {all_iou[0]:.3f} for the title.")
    return all_iou[0] if all_iou else 0.0


# -----------------------------
# Plotting (mean only)
# -----------------------------

def plot_multi_model_froc_afroc_mean_only(
    *,
    model_to_curve_paths: Dict[str, List[Path]],
    model_to_highlight_conf: Dict[str, float],
    out_froc: Path,
    out_afroc: Path,
    linewidth: float = 2.4,
    marker_size: float = 70,
) -> None:
    """
    Overlay FROC/AFROC for multiple models.
    - Aggregates across seeds internally (mean), but plots only the mean curve.
    - Highlights one operating point per model (conf) with a marker in the curve color.
    - Legend shows model name + conf used, e.g. "YOLOv8 (conf=0.2)".
    """

    # Load per-model curves
    model_curves: Dict[str, List[CurveData]] = {}
    for model, paths in model_to_curve_paths.items():
        clean = [Path(p) for p in paths if p is not None]
        if not clean:
            continue
        model_curves[model] = [_load_curve_json(p) for p in clean]


    print("[INFO] Curve inputs:")
    if not model_curves:
        print("  (no models provided)")
        return
    for model, curves in model_curves.items():
        print(f"  - {model}: {len(curves)} file(s)")
        for c in curves:
            print(f"      {c.path}")

    iou_for_title = _pick_iou_for_title(model_curves)

    # --------
    # FROC
    # --------
    out_froc.parent.mkdir(parents=True, exist_ok=True)
    fig_f, ax_f = plt.subplots()

    for model in model_curves.keys():
        curves = model_curves[model]
        curves_with_froc = [c for c in curves if c.froc_fp_per_image and c.froc_tpr_loc]
        if not curves_with_froc:
            print(f"[INFO] {model}: no FROC payload found -> skipping in FROC plot.")
            continue

        conf_grid = _assert_same_conf_thresholds(curves_with_froc, model)

        X = np.array([c.froc_fp_per_image for c in curves_with_froc], dtype=float)
        Y = np.array([c.froc_tpr_loc for c in curves_with_froc], dtype=float)

        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)

        conf_sel = model_to_highlight_conf.get(model, None)
        label = model if conf_sel is None else f"{model} (conf={float(conf_sel):g})"

        line = ax_f.plot(mean_x, mean_y, linewidth=linewidth, label=label)[0]
        color = line.get_color()

        if conf_sel is not None:
            j = _nearest_index(conf_grid, float(conf_sel))
            ax_f.scatter([mean_x[j]], [mean_y[j]], s=marker_size, color=color, zorder=6)

    ax_f.set_title(f"FROC (IoU > {iou_for_title:.2f})")
    ax_f.set_xlabel("False positives per image (FPPI)")
    ax_f.set_ylabel("Fraction of GT boxes detected (≥1 match)")
    ax_f.set_xlim(left=0.0)
    ax_f.set_ylim(0.0, 1.0)
    ax_f.margins(x=0.0)
    ax_f.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_f.legend()
    fig_f.tight_layout()
    fig_f.savefig(out_froc, dpi=200)
    plt.close(fig_f)
    print(f"[INFO] Wrote: {out_froc}")

    # --------
    # AFROC
    # --------
    out_afroc.parent.mkdir(parents=True, exist_ok=True)
    fig_a, ax_a = plt.subplots()

    for model in model_curves.keys():
        curves = model_curves[model]
        curves_with_afroc = [c for c in curves if c.afroc_fpf and c.afroc_tpr_loc]
        if not curves_with_afroc:
            print(f"[INFO] {model}: no AFROC payload found -> skipping in AFROC plot.")
            continue

        conf_grid = _assert_same_conf_thresholds(curves_with_afroc, model)

        X = np.array([c.afroc_fpf for c in curves_with_afroc], dtype=float)
        Y = np.array([c.afroc_tpr_loc for c in curves_with_afroc], dtype=float)

        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)

        conf_sel = model_to_highlight_conf.get(model, None)
        label = model if conf_sel is None else f"{model} (conf={float(conf_sel):g})"

        line = ax_a.plot(mean_x, mean_y, linewidth=linewidth, label=label)[0]
        color = line.get_color()

        if conf_sel is not None:
            j = _nearest_index(conf_grid, float(conf_sel))
            ax_a.scatter([mean_x[j]], [mean_y[j]], s=marker_size, color=color, zorder=6)

    ax_a.set_title(f"AFROC (IoU > {iou_for_title:.2f})")
    ax_a.set_xlabel("False positive fraction (FPF) on negative images")
    ax_a.set_ylabel("Fraction of GT boxes detected")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.margins(x=0.0)
    ax_a.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_a.legend()
    fig_a.tight_layout()
    fig_a.savefig(out_afroc, dpi=200)
    plt.close(fig_a)
    print(f"[INFO] Wrote: {out_afroc}")

def main() -> None:
    # Any missing model or empty list => skipped.
    # Put 1..3 curve.json paths per model.
    MODEL_CURVES: Dict[str, List[Path]] = {
        "Faster R-CNN": [
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s0_b96/i00_c00020/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s42_b96/i00_c00020/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s123_b96/i00_c00020/curve.json"),
        ],
        "YOLOv8": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s0_b208/i00_c00006/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s42_b208/i00_c00006/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s123_b208/i00_c00006/curve.json"),
        ],
        "YOLOv11": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s0_b208/i00_c00005/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s42_b208/i00_c00005/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s123_b208/i00_c00005/curve.json"),
        ],
        "RT-DETR": [
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s0/i00_c00032/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s42/i00_c00032/curve.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s123/i00_c00032/curve.json"),
        ],
    }

    # Confidence points to highlight (only used if that model exists above)
    MODEL_CONF: Dict[str, float] = {
        "Faster R-CNN": 0.20,
        "YOLOv8": 0.06,
        "YOLOv11": 0.05,
        "RT-DETR": 0.32,
    }

    plot_multi_model_froc_afroc_mean_only(
        model_to_curve_paths=MODEL_CURVES,
        model_to_highlight_conf=MODEL_CONF,
        out_froc=Path("froc_overlay.png"),
        out_afroc=Path("afroc_overlay.png"),
        marker_size=70,       # marker size
    )


if __name__ == "__main__":
    main()

