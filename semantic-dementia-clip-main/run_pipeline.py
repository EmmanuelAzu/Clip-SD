import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_analysis_plots import plot_accuracy_curve, plot_error_taxonomy_heatmap
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS


def _plot_scenario_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Joint (both encoders pruned) vs. Vision-Only (only the vision
    encoder pruned, text stays pristine) -- the two scenarios currently in
    scope. Text-only was dropped from the active comparison per the
    project's current focus, though JointSpaceEvaluator.run_eval still
    accepts scenario="text_only" if it's ever needed again.
    """
    scenario_labels = {
        "joint": "Joint (both encoders pruned)",
        "vision_only": "Vision-Only (vision encoder pruned, text pristine)",
    }
    colors = {"joint": "#2ca02c", "vision_only": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    for ax, col, title in [
        (axes[0], "top1_specific_acc", "Top-1 Specific Accuracy"),
        (axes[1], "semantic_entropy", "Semantic Vector Entropy (bits)"),
    ]:
        for scenario, label in scenario_labels.items():
            sub = df[df["scenario"] == scenario].sort_values("pruning_level")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(
                sub["pruning_level"] * 100, sub[col],
                color=colors[scenario], marker="o", markersize=3,
                linewidth=1.8, label=label,
            )
        ax.set_xlabel("Pruning Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Joint vs. Vision-Only Pruning Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "scenario_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Scenario comparison plot saved to: {save_path}")


def run_joint_pipeline(
    output_dir: str | None = None,
    curated_classes: list[str] | None = None,
    pruning_levels: list[float] | None = None,
) -> str:
    """Core evaluation: curated 10-class subset, joint + vision_only
    scenarios, 2.5% pruning increments from 0% to 70%.
    """
    if curated_classes is None:
        curated_classes = CURATED_CLASSES
    if pruning_levels is None:
        pruning_levels = PRUNING_LEVELS_FOCUSED

    print("=" * 60)
    print(" CORE EVALUATION PIPELINE (curated 10-class subset) ")
    print("=" * 60)
    print(f"[*] Classes ({len(curated_classes)}): {curated_classes}")
    print(f"[*] Pruning Grid ({len(pruning_levels)} stages, 2.5% steps, 0-70%)")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    evaluator = JointSpaceEvaluator(
        restrict_classes=curated_classes,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    print("\n[*] Running JOINT scenario (both encoders pruned)...")
    joint_df = evaluator.run_eval(
        pruning_levels=pruning_levels, target_area="full", scenario="joint"
    )

    print("\n[*] Running VISION-ONLY scenario (vision encoder pruned, text pristine)...")
    vision_only_df = evaluator.run_eval(
        pruning_levels=pruning_levels, target_area="full", scenario="vision_only"
    )

    csv_path = os.path.join(output_dir, "joint_space_metrics.csv")
    joint_df.to_csv(csv_path, index=False)
    print(f"\n[+] Joint-scenario metrics saved to:\n    {csv_path}")

    scenario_comparison_df = pd.concat([joint_df, vision_only_df], ignore_index=True)
    scenario_csv = os.path.join(output_dir, "scenario_comparison.csv")
    scenario_comparison_df.to_csv(scenario_csv, index=False)
    print(f"[+] Scenario comparison metrics saved to:\n    {scenario_csv}")
    _plot_scenario_comparison(scenario_comparison_df, output_dir)

    # THE cross-category plots: (1) Top-1/5/10 accuracy vs pruning, and
    # (2) the 4-tier clinical error-taxonomy heatmap -- now two separate,
    # clean figures rather than one crowded two-panel plot.
    plot_accuracy_curve(joint_df, output_dir=output_dir)
    plot_error_taxonomy_heatmap(joint_df, output_dir=output_dir)

    print("\n[+] Core evaluation pipeline complete.")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the core curated-subset CLIP pruning evaluation."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Custom directory to save output results and plots"
    )
    args = parser.parse_args()

    run_joint_pipeline(output_dir=args.output_dir)
