import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = "/data/local/aschwab/data/realColon/"
VIDEO_INFO_PATH = os.path.join(ROOT_DIR, "video_info.csv")
LESION_INFO_PATH = os.path.join(ROOT_DIR, "lesion_info.csv")
OUTPUT_DIR = "realColon_visualizations"

SPLIT_ORDER = ["Train", "Val", "Test"]
PALETTE_SPLIT = {"Train": "#5656e6", "Val": "#ffa030", "Test": "#ee3335"}
GLOBAL_COLOR = PALETTE_SPLIT["Train"]

FIGSIZE = (7.2, 4.8)
DPI = 300

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titleweight": "bold",
})

LABELS = {
    "sex": "Sex",
    "age": "Age (years)",
    "age_group": "Age group (years)",
    "endoscope_brand": "Endoscope brand",
    "duration_min": "Procedure duration (minutes)",
    "duration_group": "Procedure duration group (minutes)",
    "num_lesions": "Number of lesions per video",
    "bbps": "BBPS score",
    "size [mm]": "Lesion size (mm)",
    "size_group": "Lesion size group (mm)",
    "histology_class": "Histology",
    "site": "Anatomical location",
    "fps": "Frames per second (FPS)",
    "Split": "Dataset split",
}

# =========================================================
# DATA
# =========================================================
def load_and_prep_data():
    df_vid = pd.read_csv(VIDEO_INFO_PATH)
    df_lesion = pd.read_csv(LESION_INFO_PATH)

    df_vid["fps"] = pd.to_numeric(df_vid["fps"], errors="coerce")
    df_vid["num_frames"] = pd.to_numeric(df_vid["num_frames"], errors="coerce")
    df_vid.loc[df_vid["fps"] <= 0, "fps"] = np.nan
    df_vid["duration_min"] = df_vid["num_frames"] / df_vid["fps"] / 60.0


    dur_bins = [0, 10, 30, 50, 70, 90]
    dur_labels = ["(0, 10]", "(10, 30]", "(30, 50]", "(50, 70]", "(70, 90]"]
    df_vid["duration_group"] = pd.cut(df_vid["duration_min"], bins=dur_bins, labels=dur_labels)

    age_bins = [40, 50, 60, 70, 80, 90]
    age_labels = ["(40, 50]", "(50, 60]", "(60, 70]", "(70, 80]", "(80, 90]"]
    df_vid["age_group"] = pd.cut(df_vid["age"], bins=age_bins, labels=age_labels)

    def group_size(s):
        try:
            s = float(s)
        except Exception:
            return np.nan
        return "10-17" if s >= 10 else str(int(s))

    df_lesion["size_group"] = df_lesion["size [mm]"].apply(group_size)
    size_order = [str(i) for i in range(1, 10)] + ["10-17"]
    df_lesion["size_group"] = pd.Categorical(df_lesion["size_group"], categories=size_order, ordered=True)

    df_lesion["site"] = df_lesion["site"].astype(str).str.lower().str.title()

    def get_split(video_name: str) -> str:
        s = str(video_name).strip()

        # Try common patterns, from most specific to most general.
        candidates = []

        # Pattern A: "001-011_014436.jpg" -> token "011"
        if "-" in s:
            try:
                after_dash = s.split("-", 1)[1]
                token = after_dash.split("_", 1)[0]
                candidates.append(token)
            except Exception:
                pass

        # Pattern B: plain index like "12" or "12.0"
        candidates.append(s)

        idx = None
        for tok in candidates:
            tok = str(tok).strip()
            try:
                # float handles "12.0"; int(float(..)) handles both "12" and "12.0"
                idx = int(float(tok))
                break
            except Exception:
                continue

        if idx is None:
            return "Unknown"

        if idx <= 10:
            return "Train"
        elif idx <= 12:
            return "Val"
        elif idx <= 15:
            return "Test"
        return "Unknown"



    df_vid["Split"] = df_vid["unique_video_name"].apply(get_split)
    split_map = df_vid.set_index("unique_video_name")["Split"].to_dict()
    df_lesion["Split"] = df_lesion["unique_video_name"].map(split_map)

    df_vid = df_vid[df_vid["Split"].isin(SPLIT_ORDER)].copy()
    df_lesion = df_lesion[df_lesion["Split"].isin(SPLIT_ORDER)].copy()
    return df_vid, df_lesion

# =========================================================
# PLOT UTIL
# =========================================================
def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    _ensure_outdir()
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def style_axes(ax):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(width=1.0)

def annotate_bars(ax, orient="v", show_pct=False, fontsize=9, pad=2):
    if not ax.containers:
        return

    totals = None
    if show_pct:
        vals = []
        for c in ax.containers:
            for p in c:
                vals.append(p.get_height() if orient == "v" else p.get_width())
        s = float(np.sum(vals)) if len(vals) else 0.0
        totals = s if s > 0 else None

    for c in ax.containers:
        labels = []
        for p in c:
            v = p.get_height() if orient == "v" else p.get_width()
            if v <= 0:
                labels.append("")
            elif totals:
                labels.append(f"{int(v)} ({(v / totals) * 100:.1f}%)")
            else:
                labels.append(f"{int(v)}")
        ax.bar_label(c, labels=labels, fontsize=fontsize, padding=pad)

def add_n_split_labels(ax, df, y_col, split_col="Split", order=SPLIT_ORDER):
    d = df[[split_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna(subset=[split_col, y_col])
    counts = d[split_col].value_counts().reindex(order).fillna(0).astype(int)

    # Force fixed ticks and labels (robust across matplotlib/seaborn versions)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{s}\n(n={counts[s]})" for s in order])



def proportion_bar(ax, data, x, hue="Split", order_x=None, order_hue=SPLIT_ORDER, palette=PALETTE_SPLIT):
    df = data[[x, hue]].dropna().copy()
    if order_x is None:
        order_x = df[x].value_counts().index.tolist()

    ct = pd.crosstab(df[x], df[hue]).reindex(index=order_x, columns=order_hue).fillna(0)
    denom = ct.sum(axis=1).replace(0, np.nan)
    prop = ct.div(denom, axis=0).fillna(0)

    bottoms = np.zeros(len(prop))
    xs = np.arange(len(prop.index))
    for col in prop.columns:
        vals = prop[col].values
        ax.bar(xs, vals, bottom=bottoms, color=palette[col], label=col)
        bottoms += vals

    ax.set_xticks(xs)
    ax.set_xticklabels(prop.index)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share")
    ax.legend(title=LABELS.get(hue, hue), frameon=True)

# =========================================================
# GENERIC PLOTTERS (no repetition)
# =========================================================
def plot_global_count(df, col, title, fname, order=None, horizontal=False, show_pct=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if horizontal:
        sns.countplot(data=df, y=col, order=order, color=GLOBAL_COLOR, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(LABELS.get(col, col))
        annotate_bars(ax, orient="h", show_pct=show_pct)
    else:
        sns.countplot(data=df, x=col, order=order, color=GLOBAL_COLOR, ax=ax)
        ax.set_xlabel(LABELS.get(col, col))
        ax.set_ylabel("Count")
        annotate_bars(ax, orient="v", show_pct=show_pct)
    ax.set_title(title)
    style_axes(ax)
    save(fig, fname)

def plot_split_proportion(df, col, title, fname, order=None, rotate=0):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    proportion_bar(ax, df, x=col, order_x=order)
    ax.set_title(title)
    ax.set_xlabel(LABELS.get(col, col))
    if rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate, ha="right")
    style_axes(ax)
    save(fig, fname)

def plot_split_violin(df, y, title, fname, ylabel=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.violinplot(
        data=df, x="Split", y=y,
        order=SPLIT_ORDER,
        palette=PALETTE_SPLIT,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(LABELS["Split"])
    ax.set_ylabel(ylabel or LABELS.get(y, y))
    add_n_split_labels(ax, df, y_col=y)
    style_axes(ax)
    save(fig, fname)

# =========================================================
# PLOT SPEC (10 topics; global + split each)
# =========================================================
def plot_all(df_vid, df_lesion):
    # 1 Sex
    plot_global_count(df_vid, "sex", "Global: Sex distribution", "01_sex_global.png", horizontal=True)
    plot_split_proportion(df_vid, "sex", "Split: Sex composition", "01_sex_split.png")

    # 2 Age
    age_order = list(df_vid["age_group"].cat.categories)
    plot_global_count(df_vid, "age_group", "Global: Age groups", "02_age_global.png", order=age_order, horizontal=True)
    plot_split_violin(df_vid, "age", "Split: Age distribution", "02_age_split.png")

    # 3 Endoscope brand
    brand_order = df_vid["endoscope_brand"].value_counts().index.tolist()
    plot_global_count(df_vid, "endoscope_brand", "Global: Endoscope brand", "03_endoscope_global.png",
                      order=brand_order, horizontal=True)
    plot_split_proportion(df_vid, "endoscope_brand", "Split: Endoscope brand composition",
                          "03_endoscope_split.png", order=brand_order, rotate=20)

    # 4 Duration
    dur_order = list(df_vid["duration_group"].cat.categories)
    plot_global_count(df_vid, "duration_group", "Global: Procedure duration groups", "04_duration_global.png",
                      order=dur_order, horizontal=True)
    plot_split_violin(df_vid, "duration_min", "Split: Procedure duration", "04_duration_split.png",
                      ylabel=LABELS["duration_min"])

    # 5 Lesions per video (discrete)
    lesions_order = sorted(df_vid["num_lesions"].dropna().unique())
    plot_global_count(df_vid, "num_lesions", "Global: Lesions per video", "05_polyp_count_global.png",
                      order=lesions_order, horizontal=False)
    plot_split_proportion(df_vid, "num_lesions", "Split: Lesions-per-video composition",
                          "05_polyp_count_split.png", order=lesions_order)

    # 6 BBPS (ordinal)
    bbps_order = sorted(df_vid["bbps"].dropna().unique())
    plot_global_count(df_vid, "bbps", "Global: BBPS score", "06_bbps_global.png",
                      order=bbps_order, horizontal=False)
    plot_split_proportion(df_vid, "bbps", "Split: BBPS composition", "06_bbps_split.png",
                          order=bbps_order)

    # 7 Lesion size
    size_order = list(df_lesion["size_group"].cat.categories)
    plot_global_count(df_lesion, "size_group", "Global: Lesion size groups", "07_size_global.png",
                      order=size_order, horizontal=True)
    plot_split_violin(df_lesion, "size [mm]", "Split: Lesion size distribution", "07_size_split.png",
                      ylabel=LABELS["size [mm]"])

    # 8 Histology
    hist_order = df_lesion["histology_class"].value_counts().index.tolist()
    plot_global_count(df_lesion, "histology_class", "Global: Histology distribution", "08_histology_global.png",
                      order=hist_order, horizontal=True)
    plot_split_proportion(df_lesion, "histology_class", "Split: Histology composition",
                          "08_histology_split.png", order=hist_order, rotate=20)

    # 9 Site
    site_order = ["Caecum", "Ascending", "Hepatic_Flexure", "Transverse",
                  "Splenic_Flexure", "Descending", "Sigma", "Rectum"]
    plot_global_count(df_lesion, "site", "Global: Anatomical location", "09_location_global.png",
                      order=site_order[::-1], horizontal=True)
    plot_split_proportion(df_lesion, "site", "Split: Location composition",
                          "09_location_split.png", order=site_order, rotate=25)

    # 10 FPS
    fps_order = sorted(df_vid["fps"].dropna().unique())
    plot_global_count(df_vid, "fps", "Global: Video FPS distribution", "10_fps_global.png",
                      order=fps_order, horizontal=False)
    plot_split_proportion(df_vid, "fps", "Split: FPS composition", "10_fps_split.png",
                          order=fps_order)

if __name__ == "__main__":
    df_v, df_l = load_and_prep_data()

    for y, d in [("age", df_v), ("duration_min", df_v), ("size [mm]", df_l)]:
        print("\n", y)
        print("All rows per split:\n", d["Split"].value_counts())

        d2 = (
            d.replace([np.inf, -np.inf], np.nan)
             .dropna(subset=["Split", y])
        )
        print("Valid y rows per split:\n", d2["Split"].value_counts())

    plot_all(df_v, df_l)
