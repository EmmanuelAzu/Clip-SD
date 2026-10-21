# NOTE: this file previously also contained plot_hierarchical_breakdown_suite
# (redundant with plot_category_breakdown_suite's own fine-to-coarse framing),
# plot_signal_noise_distribution_shift, and plot_concept_retrieval_heatmap --
# all removed as out-of-scope extras for the current focused pipeline
# (curated 10-class subset; cross-category breakdown + Bozeat + one tSNE
# grid + joint-vs-vision-only scenario comparison). Only the one function
# actually still in use remains here.

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import clip

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _normalize_columns(df):
    """Normalizes column names to standard lowercase for internal key checking."""
    df_copy = df.copy()
    df_copy.columns = [str(col).strip() for col in df_copy.columns]
    return df_copy


def plot_accuracy_curve(df, output_dir=None):
    """Top-1/Top-5/Top-10 specific accuracy, all three on one plot, vs.
    pruning level. Split out from the old combined
    plot_category_breakdown_suite (which crammed an accuracy line and an
    error-taxonomy heatmap into one untidy two-panel figure) into its own
    standalone plot per explicit request.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    series_specs = [
        ("top1_specific_acc", "Top-1 Accuracy", "#1f77b4", "o"),
        ("top5_specific_acc", "Top-5 Accuracy", "#ff7f0e", "s"),
        ("top10_specific_acc", "Top-10 Accuracy", "#2ca02c", "^"),
    ]
    plotted_any = False
    for col, label, color, marker in series_specs:
        if col in df.columns:
            ax.plot(
                prune_pcts, df[col], marker=marker, markersize=4,
                color=color, linewidth=2.2, label=label,
            )
            plotted_any = True

    if not plotted_any and "i2t_top1" in df.columns:
        ax.plot(prune_pcts, df["i2t_top1"], marker="o", color="#1f77b4",
                 linewidth=2.5, label="Top-1 Accuracy")

    ax.set_xlabel("Pruning Level (%)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Top-1 / Top-5 / Top-10 Accuracy vs. Pruning Level", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved accuracy curve to: {save_path}")
    return save_path


def plot_error_taxonomy_heatmap(df, output_dir=None):
    """The clinical error-taxonomy heatmap (Coordinate/Superordinate/
    Domain Error/Domain Collapse), on its own, standalone -- split out of
    the old combined plot per explicit request. Proportions, not raw
    counts (bounded 0-1, comparable across runs with different dataset
    sizes); annotation labels shown only when the grid is narrow enough
    to stay readable.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    acc_col = next((c for c in ["top1_specific_acc", "i2t_top1"] if c in df.columns), None)
    acc_series = df[acc_col] if acc_col else None

    err_cols = [
        c for c in ["coordinate error", "superordinate error", "domain error", "domain collapse"]
        if c in df.columns
    ]
    if not err_cols:
        print("[!] No error-taxonomy columns found -- skipping heatmap.")
        return None

    err_sum = df[err_cols].sum(axis=1)
    if acc_series is not None:
        total_samples = err_sum / (1 - acc_series).replace(0, np.nan)
    else:
        total_samples = err_sum.replace(0, np.nan)
    heatmap_data = (df[err_cols].div(total_samples, axis=0)).T
    heatmap_data.columns = [f"{p:.1f}%" for p in prune_pcts]
    heatmap_data.index = [c.replace("_", " ").title() for c in err_cols]

    n_cols = len(heatmap_data.columns)
    show_annot = n_cols <= 20
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.4), 5), dpi=200)
    sns.heatmap(
        heatmap_data,
        annot=show_annot,
        fmt=".2f",
        annot_kws={"size": 7} if show_annot else None,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        ax=ax,
        cbar=True,
        cbar_kws={"label": "Proportion of outcomes"},
    )
    ax.set_title("Clinical Error Taxonomy (proportion of outcomes)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pruning Level (%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=90)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_taxonomy_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved error taxonomy heatmap to: {save_path}")
    return save_path
