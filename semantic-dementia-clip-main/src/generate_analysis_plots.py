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


def plot_category_breakdown_suite(df, output_dir=None):
    """THE cross-category plot: Top-1 accuracy trajectory + a proportion-based
    clinical error-taxonomy heatmap (Coordinate/Superordinate/Domain
    Error/Domain Collapse) across pruning levels.

    NOTE: previously plotted a "Vision CKA" line (now removed -- CKA
    computation was cut from run_eval as out-of-scope for the current
    focused pipeline) and a decorative, non-empirical "Expected Theory
    Bound" dashed curve (removed -- it wasn't derived from anything, just
    a quadratic decay placeholder). The heatmap previously showed raw
    error COUNTS with annotation labels that visually overlapped once
    there were more than a handful of pruning-level columns; it now shows
    PROPORTIONS (bounded 0-1, comparable across runs with different
    dataset sizes) with annotation font scaled down and made optional for
    wide grids.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Top-1 Specific Accuracy
    acc_col = next(
        (c for c in ["top1_specific_acc", "i2t_top1", "Correct"] if c in df.columns),
        None,
    )
    if acc_col == "Correct":
        total = df[["Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse"]].sum(axis=1)
        acc_series = df["Correct"] / total
    elif acc_col is not None:
        acc_series = df[acc_col]
    else:
        acc_series = None

    if acc_series is not None:
        ax1.plot(
            prune_pcts, acc_series, marker="o", color="#1f77b4",
            linewidth=2.5, label="Top-1 Specific Accuracy",
        )

    ax1.set_xlabel("Pruning Level (%)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Top-1 Accuracy vs. Pruning Level", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True)
    ax1.set_ylim(-0.02, 1.02)

    # Panel 2: Error Taxonomy Heatmap (proportions, not raw counts)
    err_cols = [
        c for c in [
            "coordinate error", "superordinate error", "domain error", "domain collapse",
        ] if c in df.columns
    ]

    if err_cols:
        err_sum = df[err_cols].sum(axis=1)
        # total_samples derived from: err_sum = total * (1 - accuracy)
        if acc_series is not None:
            total_samples = err_sum / (1 - acc_series).replace(0, np.nan)
        else:
            total_samples = err_sum.replace(0, np.nan)
        heatmap_data = (df[err_cols].div(total_samples, axis=0)).T
        heatmap_data.columns = [f"{p:.1f}%" for p in prune_pcts]
        heatmap_data.index = [c.replace("_", " ").title() for c in err_cols]

        n_cols = len(heatmap_data.columns)
        show_annot = n_cols <= 20  # avoid unreadable overlap on wide grids
        sns.heatmap(
            heatmap_data,
            annot=show_annot,
            fmt=".2f",
            annot_kws={"size": 7} if show_annot else None,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            ax=ax2,
            cbar=True,
            cbar_kws={"label": "Proportion of outcomes"},
        )
        ax2.set_title("Clinical Error Taxonomy (proportion of outcomes)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Pruning Level (%)", fontsize=11)
        ax2.tick_params(axis="x", labelsize=7, rotation=90)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cross_category_breakdown.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved cross-category breakdown plot to: {save_path}")
