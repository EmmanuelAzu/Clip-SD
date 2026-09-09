"""Compares masking_mode="static" (Directive 6.1's non-iterative
simplification: every pruning level computed fresh from the pristine
model) against masking_mode="iterative" (damage compounds on top of the
previous level, matching a literal "progressive neurodegeneration"
narrative).

As documented in CLIPPruningEngine.get_pruned_model, these two modes are
mathematically PROVEN EQUIVALENT for magnitude/norm-ranked pruning methods
(l1_unstructured, structured_channel, structured_head) -- because no
retraining happens between pruning levels, so weight magnitudes never
change and the same global rank-ordering is recovered either way. This
script runs both modes for l1_unstructured explicitly as an empirical
sanity check of that claim (the two curves should be visually
indistinguishable), and then runs both modes for random_unstructured,
where they genuinely diverge, to show what "damage accumulates on top of
prior damage" actually looks like when the pruning mechanism doesn't rank
by magnitude.
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
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "masking_mode_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" STATIC vs. ITERATIVE (COMPOUNDING) PRUNING COMPARISON ")
    print("=" * 60)

    evaluator = JointSpaceEvaluator(sample_frac=sample_frac, target_n=target_n)

    runs = [
        ("l1_unstructured", "static"),
        ("l1_unstructured", "iterative"),
        ("random_unstructured", "static"),
        ("random_unstructured", "iterative"),
    ]

    all_results = []
    for pruning_method, masking_mode in runs:
        label = f"{pruning_method}_{masking_mode}"
        print(f"\n================ {label} ================")
        df = evaluator.run_eval(
            pruning_levels=pruning_levels,
            target_area="full",
            scenario="joint",
            pruning_method=pruning_method,
            masking_mode=masking_mode,
            include_noise_floor=False,
        )
        df["run_label"] = label
        run_csv = os.path.join(output_dir, f"{label}_metrics.csv")
        df.to_csv(run_csv, index=False)
        print(f"[+] Saved: {run_csv}")
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(output_dir, "masking_mode_comparison_full.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\n[+] Combined comparison saved to: {combined_csv}")

    _plot_comparison(combined, output_dir)
    return combined_csv


def _plot_comparison(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    styles = {
        "l1_unstructured_static": ("#1f77b4", "-"),
        "l1_unstructured_iterative": ("#1f77b4", "--"),
        "random_unstructured_static": ("#d62728", "-"),
        "random_unstructured_iterative": ("#d62728", "--"),
    }

    for ax, col, title in [
        (axes[0], "top1_specific_acc", "Top-1 Specific Accuracy"),
        (axes[1], "semantic_entropy", "Semantic Vector Entropy (bits)"),
    ]:
        for label, (color, linestyle) in styles.items():
            sub = df[df["run_label"] == label].sort_values("pruning_level")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(
                sub["pruning_level"] * 100, sub[col],
                color=color, linestyle=linestyle, marker="o", markersize=3,
                linewidth=1.8, label=label,
            )
        ax.set_xlabel("Pruning Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Static vs. Iterative Masking: identical for l1 (solid==dashed), "
        "diverges for random (solid!=dashed)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "masking_mode_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparison plot saved to: {save_path}")


if __name__ == "__main__":
    main()
