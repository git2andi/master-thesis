#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Sample mean + sample std (ddof=1 when n>=2; else std=0.0).
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def get_in(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# ----------------------------
# Metric typing / formatting
# ----------------------------

COUNT_KEYS = {
    "n_frames",
    "n_preds",
    "detection_metrics.tp",
    "detection_metrics.fp",
    "detection_metrics.fn",
    "detection_metrics.n_gt",
    "detection_metrics.n_pred",
    "frame_metrics.tp",
    "frame_metrics.fp",
    "frame_metrics.tn",
    "frame_metrics.fn",
}

# In lesion summary, these are NOT simple counts (rates/means). Everything else is count-like.
LESION_NONCOUNT_KEYS = {
    "pct_detected_any",
    "pct_detected_25pct",
    "pct_detected_50pct",
    "mean_det_fraction",
    "mean_det_fraction_pct",
    "pct_detected_within_1s",
    "pct_detected_within_3s",
    "pct_detected_within_5s",
}

COUNT_SUFFIXES = (".tp", ".fp", ".tn", ".fn", ".n_gt", ".n_pred")


def is_count_metric(metric_key: str) -> bool:
    if metric_key in COUNT_KEYS:
        return True
    if metric_key.startswith("lesion_metrics.summary."):
        leaf = metric_key.split("lesion_metrics.summary.", 1)[1]
        return leaf not in LESION_NONCOUNT_KEYS
    if metric_key.endswith(COUNT_SUFFIXES):
        return True
    return False


def is_lesion_pct_metric(metric_key: str) -> bool:
    """
    True for lesion_metrics.summary.pct_*
    """
    return metric_key.startswith("lesion_metrics.summary.pct_")


def lesion_pct_to_count_key(metric_key: str) -> Optional[str]:
    """
    lesion_metrics.summary.pct_detected_50pct -> lesion_metrics.summary.detected_50pct
    lesion_metrics.summary.pct_detected_within_3s -> lesion_metrics.summary.detected_within_3s
    """
    if not is_lesion_pct_metric(metric_key):
        return None
    leaf = metric_key.split("lesion_metrics.summary.", 1)[1]
    if not leaf.startswith("pct_"):
        return None
    count_leaf = leaf.replace("pct_", "", 1)
    return f"lesion_metrics.summary.{count_leaf}"


def is_lesion_count_key(metric_key: str) -> bool:
    """
    Lesion-level count metrics that should be reported as a RANGE [min--max].
    """
    if not metric_key.startswith("lesion_metrics.summary."):
        return False
    leaf = metric_key.split("lesion_metrics.summary.", 1)[1]
    return leaf.startswith("detected_") or leaf == "detected_any" or leaf.startswith("detected_within_")


def fmt_rate_pm(mean: float, std: float, decimals: int) -> str:
    """
    LaTeX-ish string with SINGLE backslash: '\pm' (not '\\pm').
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "nan"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} \\pm {fmt.format(std)}"


def fmt_range_int(min_v: int, max_v: int, *, compact_equal: bool = True) -> str:
    """
    Lesion-count range formatting.
      - If compact_equal: [15--15] becomes "15".
      - Else: always "[min--max]".
    """
    if compact_equal and min_v == max_v:
        return str(min_v)
    return f"[{min_v}--{max_v}]"


def format_metric_pm(
    metric_key: str,
    mean: float,
    std: float,
    *,
    decimals_rates: int,
) -> Tuple[str, float, float]:
    """
    Default formatting:
      - counts -> integer mean ± integer std
      - rates  -> float mean ± float std
    """
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "nan", float("nan"), float("nan")

    if is_count_metric(metric_key):
        dm = float(int(round(mean)))
        ds = float(int(round(std)))
        latex = f"{int(dm)} \\pm {int(ds)}"
        return latex, dm, ds

    latex = fmt_rate_pm(mean, std, decimals_rates)
    return latex, mean, std


# ----------------------------
# Extraction from results.json
# ----------------------------

def flatten_selected_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k in ["n_frames", "n_preds"]:
        v = data.get(k, None)
        if is_number(v):
            out[k] = float(v)

    cm = data.get("coco_metrics", {})
    if isinstance(cm, dict):
        for k, v in cm.items():
            if is_number(v):
                out[f"coco_metrics.{k}"] = float(v)

    dm = data.get("detection_metrics", {})
    if isinstance(dm, dict):
        for k, v in dm.items():
            if is_number(v):
                out[f"detection_metrics.{k}"] = float(v)

    fm = data.get("frame_metrics", {})
    if isinstance(fm, dict):
        for k, v in fm.items():
            if is_number(v):
                out[f"frame_metrics.{k}"] = float(v)

    lm_sum = get_in(data, ["lesion_metrics", "summary"])
    if isinstance(lm_sum, dict):
        for k, v in lm_sum.items():
            if is_number(v):
                out[f"lesion_metrics.summary.{k}"] = float(v)

    return out


@dataclass
class FileRecord:
    path: Path
    metrics: Dict[str, float]
    file_conf: Optional[float]
    file_iou: Optional[float]


def load_results(path: Path) -> FileRecord:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_conf = data.get("conf", None)
    raw_iou = data.get("iou_thr", None)
    file_conf = float(raw_conf) if is_number(raw_conf) else None
    file_iou = float(raw_iou) if is_number(raw_iou) else None

    return FileRecord(
        path=path,
        metrics=flatten_selected_metrics(data),
        file_conf=file_conf,
        file_iou=file_iou,
    )


# ----------------------------
# Aggregation
# ----------------------------

def aggregate_model(
    model_name: str,
    records: List[FileRecord],
    *,
    decimals_rates: int,
    warn_on_missing_metrics: bool,
    warnings: List[str],
    tol_conf_iou: float,
    compact_equal_ranges: bool,
) -> Dict[str, Any]:
    keys = sorted({k for r in records for k in r.metrics.keys()})

    confs = [r.file_conf for r in records if r.file_conf is not None]
    ious = [r.file_iou for r in records if r.file_iou is not None]

    # Warn if mixed conf/iou across files (no manual validation; only report).
    if confs and (max(confs) - min(confs) > tol_conf_iou):
        warnings.append(f"[WARN] {model_name}: mixed conf across files: {confs}")
    if ious and (max(ious) - min(ious) > tol_conf_iou):
        warnings.append(f"[WARN] {model_name}: mixed iou_thr across files: {ious}")

    conf_mean, conf_std = (mean_std([float(x) for x in confs]) if confs else (None, None))
    iou_mean, iou_std = (mean_std([float(x) for x in ious]) if ious else (None, None))

    out: Dict[str, Any] = {
        "n_seeds": len(records),
        "paths": [str(r.path) for r in records],
        "conf": {"values": confs, "mean": conf_mean, "std": conf_std},
        "iou_thr": {"values": ious, "mean": iou_mean, "std": iou_std},
        "metrics": {},
    }

    for k in keys:
        vals = [r.metrics[k] for r in records if k in r.metrics and is_number(r.metrics[k])]
        if not vals:
            continue

        if warn_on_missing_metrics and len(vals) != len(records):
            warnings.append(
                f"[WARN] {model_name}: metric '{k}' present in {len(vals)}/{len(records)} files; "
                f"aggregating over available values."
            )

        # Lesion pct metrics: convert to COUNT per file, then report RANGE [min--max].
        if is_lesion_pct_metric(k):
            count_key = lesion_pct_to_count_key(k)
            per_file_counts: List[float] = []

            for r in records:
                # Prefer explicit detected_* count
                if count_key is not None and count_key in r.metrics and is_number(r.metrics[count_key]):
                    per_file_counts.append(float(r.metrics[count_key]))
                    continue

                # Otherwise derive from pct and n_lesions
                pct = r.metrics.get(k, None)
                n_les = r.metrics.get("lesion_metrics.summary.n_lesions", None)
                if is_number(pct) and is_number(n_les):
                    per_file_counts.append(float(pct) / 100.0 * float(n_les))

            if warn_on_missing_metrics and len(per_file_counts) != len(records):
                warnings.append(
                    f"[WARN] {model_name}: lesion pct metric '{k}' could be converted to counts for "
                    f"{len(per_file_counts)}/{len(records)} files (needs detected_* or n_lesions)."
                )

            if per_file_counts:
                min_c = int(round(min(per_file_counts)))
                max_c = int(round(max(per_file_counts)))
                out["metrics"][k] = {
                    "min": float(min_c),
                    "max": float(max_c),
                    "latex": fmt_range_int(min_c, max_c, compact_equal=compact_equal_ranges),
                    "n": len(per_file_counts),
                    "type": "lesion_count_range_from_pct",
                    "source_pct_key": k,
                    "preferred_count_key": count_key,
                }
            continue

        # Lesion count metrics: report RANGE [min--max].
        if is_lesion_count_key(k):
            min_c = int(round(min(vals)))
            max_c = int(round(max(vals)))
            out["metrics"][k] = {
                "min": float(min_c),
                "max": float(max_c),
                "latex": fmt_range_int(min_c, max_c, compact_equal=compact_equal_ranges),
                "n": len(vals),
                "type": "lesion_count_range",
            }
            continue

        # Default path: mean ± std (counts as rounded ints, rates as floats)
        m, s = mean_std([float(v) for v in vals])
        latex, display_mean, display_std = format_metric_pm(
            k, m, s, decimals_rates=decimals_rates
        )

        out["metrics"][k] = {
            "mean": m,
            "std": s,
            "display_mean": display_mean,
            "display_std": display_std,
            "latex": latex,
            "n": len(vals),
            "type": "count" if is_count_metric(k) else "rate",
        }

    return out


def write_tsv(out_tsv: Path, per_model: Dict[str, Dict[str, Any]]) -> None:
    """
    Flat TSV, sorted by metric first, then model, so the same metric is grouped across models.

    For lesion range metrics, mean/std columns contain min/max to keep schema simple.
    """
    header = "model\tmetric\ttype\tmean\tstd\tlatex\tn\tconf_mean\tconf_std\tiou_mean\tiou_std"
    rows: List[Tuple[str, str, str, str, str, str, str, str, str, str, str]] = []

    for model, blob in per_model.items():
        conf_mean = blob.get("conf", {}).get("mean", "")
        conf_std = blob.get("conf", {}).get("std", "")
        iou_mean = blob.get("iou_thr", {}).get("mean", "")
        iou_std = blob.get("iou_thr", {}).get("std", "")

        metrics = blob.get("metrics", {})
        for metric_key, stats in metrics.items():
            mtype = stats.get("type", "")

            if mtype in ("lesion_count_range", "lesion_count_range_from_pct"):
                mean_v = stats.get("min", "")
                std_v = stats.get("max", "")
            else:
                mean_v = stats.get("display_mean", stats.get("mean", ""))
                std_v = stats.get("display_std", stats.get("std", ""))

            rows.append(
                (
                    model,
                    metric_key,
                    mtype,
                    str(mean_v),
                    str(std_v),
                    str(stats.get("latex", "")),
                    str(stats.get("n", "")),
                    str(conf_mean),
                    str(conf_std),
                    str(iou_mean),
                    str(iou_std),
                )
            )

    MODEL_ORDER = ["Faster R-CNN", "YOLOv8", "YOLOv11", "RT-DETR"]
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda r: (r[1], model_rank.get(r[0], 10**9)))


    lines = [header] + ["\t".join(r) for r in rows]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------
# Main (manual paths here)
# ----------------------------

def main() -> None:
    # No manual thresholding: conf/iou_thr are read from each results.json.
    TOL_CONF_IOU: float = 1e-6
    DECIMALS_RATES: int = 3
    WARN_ON_MISSING_METRICS: bool = True

    # If True: [15--15] becomes "15" (cleaner in tables)
    COMPACT_EQUAL_RANGES: bool = True

    #OUT_JSON: Path = Path("./summary_pm_c20.json")
    OUT_TSV: Optional[Path] = Path("./sc20.tsv")

    MODEL_RESULTS: Dict[str, List[Path]] = {
        "Faster R-CNN": [
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s0_b96/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s42_b96/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/fasterrcnn_realColon_640_s123_b96/i00_c00020/results.json"),
        ],
        "YOLOv8": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s0_b208/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s42_b208/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y8m_realColon_640_s123_b208/i00_c00020/results.json"),
        ],
        "YOLOv11": [
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s0_b208/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s42_b208/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/y11m_realColon_640_s123_b208/i00_c00020/results.json"),
        ],
        "RT-DETR": [
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s0/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s42/i00_c00020/results.json"),
            Path("/home/stud/aschwab/master-thesis/eval/results/filtered_rtdetr_realColon_640_s123/i00_c00020/results.json"),
        ],
    }

    warnings: List[str] = []
    per_model_out: Dict[str, Any] = {}

    for model_name, paths in MODEL_RESULTS.items():
        if not paths:
            continue
        if len(paths) > 3:
            raise ValueError(f"{model_name}: expected 1–3 paths, got {len(paths)}")

        records: List[FileRecord] = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"{model_name}: file not found: {p}")
            records.append(load_results(p))

        per_model_out[model_name] = aggregate_model(
            model_name,
            records,
            decimals_rates=DECIMALS_RATES,
            warn_on_missing_metrics=WARN_ON_MISSING_METRICS,
            warnings=warnings,
            tol_conf_iou=TOL_CONF_IOU,
            compact_equal_ranges=COMPACT_EQUAL_RANGES,
        )

        # Enforce model order in all outputs (JSON + TSV)
    MODEL_ORDER = ["Faster R-CNN", "YOLOv8", "YOLOv11", "RT-DETR"]
    per_model_out_ordered: Dict[str, Any] = {k: per_model_out[k] for k in MODEL_ORDER if k in per_model_out}

    # (optional) append any unexpected models at the end (keeps behavior robust)
    for k in per_model_out.keys():
        if k not in per_model_out_ordered:
            per_model_out_ordered[k] = per_model_out[k]

    out = {
        "input": {
            "decimals_rates": DECIMALS_RATES,
            "tol_conf_iou": TOL_CONF_IOU,
            "lesion_reporting": "range [min--max] across seeds (counts); pct metrics converted to counts first",
            "latex_range": "uses [min--max] with '--' as LaTeX en-dash",
            "latex_pm": "uses '\\pm' (single backslash) where applicable",
            "compact_equal_ranges": COMPACT_EQUAL_RANGES,
        },
        "warnings": warnings,
        "models": per_model_out_ordered,
    }

    #OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    #OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if OUT_TSV is not None:
        write_tsv(OUT_TSV, per_model_out_ordered)

    for w in warnings:
        print(w)
    #print(f"Wrote: {OUT_JSON}")
    if OUT_TSV is not None:
        print(f"Wrote: {OUT_TSV}")


if __name__ == "__main__":
    main()
