"""RQ2: "How does the hierarchical breakdown of semantic categories vary
when pruning is targeted at different layer depths (early, middle, or
deep) of the CLIP architecture compared to random global pruning?"

Previously CLIPPruningEngine fully supported depth_zone/sub_module
targeting, but no run_*.py script ever actually exercised it -- every real
pipeline run used the "global" default, so RQ2 had no experiment producing
data for it despite being a named research question. This script closes
that gap, mirroring run_bozeat_grid.py's structure.

Each depth_zone is run as its own separate sweep (own CSV, own model
copies -- they are NOT mutually exclusive pruning masks layered on one
model) and the results are then combined into one comparison CSV/plot, per
the "separate them but compare results" design.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator

DEPTH_ZONES = ["early", "middle", "deep", "global"]
PRUNING_LEVELS_5PCT = [round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()]


def main(
    pruning_levels=PRUNING_LEVELS_5PCT,
    output_dir=None,
    sub_module="all",
    sample_frac=1.0,
    target_n=None,
):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "depth_zone_grid")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" RQ2: DEPTH-ZONE TARGETED PRUNING COMPARISON ")
    print("=" * 60)
    print(f"[*] Depth zones: {DEPTH_ZONES}  |  sub_module='{sub_module}'")

    evaluator = JointSpaceEvaluator(sample_frac=sample_frac, target_n=target_n)

    all_results = []
    for zone in DEPTH_ZONES:
        print(f"\n================ Depth Zone: {zone.upper()} ================")
        df = evaluator.run_eval(
            pruning_levels=pruning_levels,
            target_area="full",
            depth_zone=zone,
            sub_module=sub_module,
            scenario="joint",
            masking_mode="static",
            include_noise_floor=(zone == DEPTH_ZONES[0]),  # compute once, not 4x
        )
        df["depth_zone"] = zone
        zone_csv = os.path.join(output_dir, f"depth_zone_{zone}_metrics.csv")
        df.to_csv(zone_csv, index=False)
        print(f"[+] {zone} sweep saved to: {zone_csv}")
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(output_dir, "depth_zone_comparison_full.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\n[+] Combined comparison saved to: {combined_csv}")

    _plot_depth_zone_comparison(combined, output_dir)
    return combined_csv


def _plot_depth_zone_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Overlays each depth_zone's accuracy/entropy trajectory for direct
    visual comparison -- answers RQ2: does hierarchical decay depend on
    WHERE in the network damage occurs, or only on how much?
    """
    metrics_to_plot = [
        ("top1_specific_acc", "Top-1 Specific Accuracy"),
        ("top1_coordinate_acc", "Top-1 Coordinate (Basic-Level) Accuracy"),
        ("top1_super_acc", "Top-1 Superordinate Accuracy"),
        ("semantic_entropy", "Semantic Vector Entropy (bits)"),
    ]
    colors = {"early": "#1f77b4", "middle": "#ff7f0e", "deep": "#d62728", "global": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics_to_plot):
        if col not in df.columns:
            continue
        for zone in DEPTH_ZONES:
            sub = df[df["depth_zone"] == zone].sort_values("pruning_level")
            if sub.empty:
                continue
            ax.plot(
                sub["pruning_level"] * 100, sub[col],
                marker="o", markersize=3, linewidth=1.8,
                label=zone, color=colors.get(zone),
            )
        ax.set_xlabel("Pruning Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("RQ2: Depth-Zone Targeted Pruning vs. Global Pruning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "depth_zone_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparison plot saved to: {save_path}")


if __name__ == "__main__":
    main()
