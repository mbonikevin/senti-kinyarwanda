"""charts for the report"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import LABEL_ORDER

FIGURES_DIR = Path("reports/figures")
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
# blue and red rather than green and red so colourblind readers can separate them
SENTIMENT_COLOR = {"positive": "#2a78d6", "neutral": "#8a8882", "negative": "#e34948"}
SEQ_BLUE = "#3987e5"

def _frame(ax, title, xlabel=None, ylabel=None, grid_axis="y"):
    """apply the shared chart style"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=SECONDARY)
    return ax

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved reports/figures/{name}.png")
    return fig

def class_distribution(splits, name="class_distribution"):
    """bars of class counts the gap here is why we lead with macro f1"""
    names = ["train", "validation", "test"]
    x, width = np.arange(len(names)), 0.26

    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    for i, label in enumerate(LABEL_ORDER):
        counts = [int((splits[s]["label"] == i).sum()) for s in names]
        bars = ax.bar(x + (i - 1) * width, counts, width - 0.02,
                      label=label, color=SENTIMENT_COLOR[label])
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, count, str(count),
                    ha="center", va="bottom", fontsize=8, color=SECONDARY)

    _frame(ax, "Class counts by split", ylabel="Tweets")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={len(splits[s])})" for s in names], fontsize=9)
    for text in ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right").get_texts():
        text.set_color(SECONDARY)
    plt.tight_layout()
    return _save(fig, name)

def length_distribution(splits, max_len=128, name="length_distribution"):
    """tweet lengths proving max_len barely cuts anything off"""
    lengths = splits["train"]["text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    ax.hist(lengths, bins=range(0, int(lengths.max()) + 3, 2),
            color=SEQ_BLUE, edgecolor=SURFACE, linewidth=0.8)
    p95 = int(lengths.quantile(0.95))
    ax.axvline(p95, color=SECONDARY, linewidth=1.2, linestyle=":")
    ax.text(p95 + 0.6, ax.get_ylim()[1] * 0.92, f"95th pct = {p95}",
            fontsize=8, color=SECONDARY)

    _frame(ax, f"Tweet length (max {int(lengths.max())} words)",
           xlabel="Words", ylabel="Tweets")
    ax.text(0.98, 0.72, f"max_len = {max_len} tokens", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED)
    plt.tight_layout()
    return _save(fig, name)

def experiment_results(log_df, name="experiment_results", baseline_f1=None):
    """dot plot of each run dots not bars because the gaps are small"""
    df = log_df.sort_values("val_macro_f1")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.6), facecolor=SURFACE)
    ax.hlines(y, df["val_macro_f1"].min() - 0.01, df["val_macro_f1"],
              color=GRID, linewidth=1.4)
    ax.plot(df["val_macro_f1"], y, "o", markersize=9, color=SEQ_BLUE,
            markeredgecolor=SURFACE, markeredgewidth=2)
    if baseline_f1 is not None:
        ax.axvline(baseline_f1, color="#e34948", linewidth=1.4, linestyle="--")
        ax.text(baseline_f1, len(df) - 0.4, f" baseline {baseline_f1:.3f}",
                fontsize=8, color="#e34948", va="top")
    for yi, value in zip(y, df["val_macro_f1"]):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8, color=SECONDARY)

    _frame(ax, "Validation macro-F1 by run", xlabel="Macro-F1", grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.exp_id}  {r.stage}" for r in df.itertuples()], fontsize=9)
    ax.set_xlim(df["val_macro_f1"].min() - 0.02, df["val_macro_f1"].max() + 0.04)
    plt.tight_layout()
    return _save(fig, name)

def error_feature_chart(feature_df, overall_error_rate, name="error_features"):
    """error rate for each tag set against the overall rate"""
    df = feature_df.sort_values("err_with")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(df) + 1.5), facecolor=SURFACE)
    bars = ax.barh(y, df["err_with"], height=0.6, color=SEQ_BLUE)
    ax.axvline(overall_error_rate, color=SECONDARY, linewidth=1.4, linestyle="--")
    ax.text(overall_error_rate, len(df) - 0.3, f" overall {overall_error_rate:.2f}",
            fontsize=8, color=SECONDARY, va="top")
    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{row.err_with:.2f}  (n={row.n_with})", va="center",
                fontsize=8, color=SECONDARY)

    _frame(ax, "Error rate by tweet trait", xlabel="Error rate", grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlim(0, max(df["err_with"].max(), overall_error_rate) * 1.35)
    plt.tight_layout()
    return _save(fig, name)
