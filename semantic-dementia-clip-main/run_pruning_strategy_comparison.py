"""Compares three distinct ways of destroying semantic information, at
matched nominal severity levels, to separate two different questions the
proposal conflates:

1. l1_unstructured  -- "zero-weight" MAGNITUDE pruning (the pipeline's
   main/default method): deterministically removes the weakest-magnitude
   (most redundant) connections first. Biological analogy: targeted loss
   of weak/underused synapses.
2. random_unstructured -- RANDOM weight pruning: removes an unbiased random
   sample of connections regardless of magnitude. Biological analogy:
   undirected, non-selective neuronal damage.
3. Bfloor -- Gaussian noise floor (Sec. 3.3.3): does NOT touch model
   weights at all; replaces the query embedding directly with random
   noise. This is not a "pruning strategy" in the same sense as (1)/(2) --
   it's the chance-level reference every pruning curve should be judged
   against, plotted here as a constant horizontal line for direct
   comparison.

If magnitude pruning degrades performance markedly more slowly than random
pruning at the same nominal sparsity, that is itself a substantive finding
supporting the proposal's claim that CLIP's embedding space has meaningful,
non-uniformly-distributed structure (i.e. not every weight is equally
load-bearing) -- exactly the kind of result Bfloor is needed to contextualize.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator

PRUNING_LEVELS_5PCT = [round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()]


def main(pruning_levels=PRUNING_LEVELS_5PCT, output_dir=None, sample_frac=1.0, target_n=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "pruning_strategy_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" PRUNING STRATEGY COMPARISON: MAGNITUDE vs RANDOM vs NOISE FLOOR ")
    print("=" * 60)

    evaluator = JointSpaceEvaluator(sample_frac=sample_frac, target_n=target_n)

    strategies = [
        ("l1_unstructured", "Magnitude (zero-weight) pruning"),
        ("random_unstructured", "Random weight pruning"),
    ]

    all_results = []
    for pruning_method, label in strategies:
        print(f"\n================ {label} ================")
        df = evaluator.run_eval(
            pruning_levels=pruning_levels,
            target_area="full",
            scenario="joint",
            pruning_method=pruning_method,
            masking_mode="static",
            # Bfloor is identical regardless of pruning_method (it doesn't
            # touch weights), so only compute it once to avoid wasted work.
            include_noise_floor=(pruning_method == strategies[0][0]),
        )
        df["strategy"] = label
        run_csv = os.path.join(output_dir, f"{pruning_method}_metrics.csv")
        df.to_csv(run_csv, index=False)
        print(f"[+] Saved: {run_csv}")
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(output_dir, "pruning_strategy_comparison_full.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\n[+] Combined comparison saved to: {combined_csv}")

    _plot_comparison(combined, output_dir)
    return combined_csv


def _plot_comparison(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    colors = {"Magnitude (zero-weight) pruning": "#1f77b4", "Random weight pruning": "#d62728"}

    noise_floor_acc = df["noise_floor_top1_acc"].dropna()
    noise_floor_acc = noise_floor_acc.iloc[0] if len(noise_floor_acc) else None
    noise_floor_ent = df["noise_floor_entropy"].dropna()
    noise_floor_ent = noise_floor_ent.iloc[0] if len(noise_floor_ent) else None

    for ax, col, title, floor_val in [
        (axes[0], "top1_specific_acc", "Top-1 Specific Accuracy", noise_floor_acc),
        (axes[1], "semantic_entropy", "Semantic Vector Entropy (bits)", noise_floor_ent),
    ]:
        for strategy, color in colors.items():
            sub = df[df["strategy"] == strategy].sort_values("pruning_level")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(
                sub["pruning_level"] * 100, sub[col],
                color=color, marker="o", markersize=3, linewidth=1.8, label=strategy,
            )
        if floor_val is not None:
            ax.axhline(
                floor_val, color="gray", linestyle=":", linewidth=1.5,
                label="Bfloor (Gaussian noise)",
            )
        ax.set_xlabel("Pruning Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Magnitude vs. Random Weight Pruning, against the Gaussian Noise Floor (Bfloor)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "pruning_strategy_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparison plot saved to: {save_path}")


if __name__ == "__main__":
    main()
