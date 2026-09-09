import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_analysis_plots import (
    plot_category_breakdown_suite,
    plot_hierarchical_breakdown_suite,
    plot_signal_noise_distribution_shift,
    plot_concept_retrieval_heatmap,
)
from src.generate_tsne import generate_joint_hierarchical_tsne
from src.extended_hierarchical_eval import (
    plot_hierarchical_confusion_matrices,
    generate_hierarchical_hsv_tsne,
    compute_and_plot_top10_rank_taxonomy,
)

PRUNING_LEVELS_5PCT = [round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()]


def _plot_scenario_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Overlays joint / text_only / vision_only trajectories -- the core
    empirical comparison behind RQ1/RQ3: does damaging only the linguistic
    stream (Aphasia proxy) look different from damaging only the visual
    stream (Agnosia proxy) or damaging both (Dementia proxy)?
    """
    scenario_labels = {
        "joint": "Joint (Combined Transmodal Atrophy)",
        "text_only": "Text-Only (Progressive Aphasia proxy)",
        "vision_only": "Vision-Only (Visual Agnosia proxy)",
    }
    colors = {"joint": "#2ca02c", "text_only": "#1f77b4", "vision_only": "#d62728"}

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
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Three-Scenario Comparison: Aphasia vs. Agnosia vs. Combined Dementia Proxy",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "scenario_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Scenario comparison plot saved to: {save_path}")


def run_joint_pipeline(
    sample_frac: float = 1.0,
    target_n: int | None = None,
    balance_taxonomically: bool = True,
    output_dir: str | None = None,
) -> str:
    """Executes the full multimodal semantic dementia CLIP evaluation pipeline."""
    print("=" * 60)
    print(" RUNNING MULTIMODAL SEMANTIC DEMENTIA EVALUATION PIPELINE ")
    print("=" * 60)
    print(f"[*] Pruning Grid ({len(PRUNING_LEVELS_5PCT)} stages): {PRUNING_LEVELS_5PCT}")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    # JointSpaceEvaluator already builds a real CLIPPruningEngine on
    # self.base_model and a working self._reindex_visual_memory() in its own
    # __init__ (src/joint_evaluator.py) -- both used to be discarded here and
    # replaced with generic, attribute-guessing fallback implementations
    # that never matched this class's actual interface and always fell
    # through to their slowest, most fragile path. Use the evaluator as-is.
    evaluator = JointSpaceEvaluator(
        sample_frac=sample_frac,
        target_n=target_n,
        balance_taxonomically=balance_taxonomically,
    )

    # target_area="full" (whole encoder backbone) is the single definition of
    # "pruning level p" used everywhere in this pipeline -- every plotting
    # helper below defaults to the same target_area via CLIPPruningEngine.
    #
    # scenario="joint" is the Combined Transmodal Atrophy / Semantic
    # Dementia Hub Failure proxy (Methodology Sec. 3.3.4, Scenario 3) --
    # this remains the headline run that feeds every plotting helper below,
    # unchanged from before.
    results_df = evaluator.run_eval(
        pruning_levels=PRUNING_LEVELS_5PCT, target_area="full", scenario="joint"
    )

    csv_path = os.path.join(output_dir, "joint_space_metrics_full.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Full Evaluation metrics saved to:\n    {csv_path}")

    # -------------------------------------------------------------------
    # Three-Scenario Comparison (Methodology Sec. 3.3.4, Stage 7.5):
    # Scenario 1 (Isolated Linguistic Atrophy / Progressive Aphasia proxy)
    # and Scenario 2 (Isolated Visual Atrophy / Visual Object Agnosia
    # proxy), run with the SAME full metrics suite (CKA, NPR, entropy,
    # hierarchical breakdown, typicality) as the "joint" run above.
    # Previously this three-scenario architecture was described in detail
    # in the proposal but never actually reachable from run_full_pipeline.py
    # -- the only implementation (TestingHarness.compare_pruning_locations)
    # was an orphaned script with its own simpler metric set. Both now
    # exist: this richer version lives here; the faster
    # TestingHarness comparison is wired into run_full_pipeline.py as its
    # own separate stage.
    # -------------------------------------------------------------------
    print("\n[*] Running Scenario Comparison (Aphasia / Agnosia / Dementia proxies)...")
    text_only_df = evaluator.run_eval(
        pruning_levels=PRUNING_LEVELS_5PCT, target_area="full",
        scenario="text_only", include_noise_floor=False,
    )
    vision_only_df = evaluator.run_eval(
        pruning_levels=PRUNING_LEVELS_5PCT, target_area="full",
        scenario="vision_only", include_noise_floor=False,
    )

    scenario_comparison_df = pd.concat(
        [results_df, text_only_df, vision_only_df], ignore_index=True
    )
    scenario_csv = os.path.join(output_dir, "scenario_comparison_full.csv")
    scenario_comparison_df.to_csv(scenario_csv, index=False)
    print(f"[+] Scenario comparison metrics saved to:\n    {scenario_csv}")
    _plot_scenario_comparison(scenario_comparison_df, output_dir)

    # Generate baseline visualization suite
    plot_category_breakdown_suite(results_df, output_dir=output_dir)
    plot_hierarchical_breakdown_suite(results_df, output_dir=output_dir)
    plot_signal_noise_distribution_shift(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )
    plot_concept_retrieval_heatmap(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )
    generate_joint_hierarchical_tsne(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )

    # Extended Evaluation Metrics
    plot_hierarchical_confusion_matrices(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )
    generate_hierarchical_hsv_tsne(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )
    compute_and_plot_top10_rank_taxonomy(
        evaluator, pruning_levels=PRUNING_LEVELS_5PCT, output_dir=output_dir
    )

    print("\n[+] Honors project pipeline with extended metrics executed successfully end-to-end!")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Multimodal Semantic Dementia CLIP Pruning Evaluation."
    )
    parser.add_argument(
        "--sample_frac", type=float, default=1.0, help="Fraction of dataset to evaluate"
    )
    parser.add_argument(
        "--target_n", type=int, default=None, help="Explicit sample count for evaluation"
    )
    parser.add_argument(
        "--no_balance", action="store_true", help="Disable taxonomic class balancing"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Custom directory to save output results and plots"
    )
    args = parser.parse_args()

    run_joint_pipeline(
        sample_frac=args.sample_frac,
        target_n=args.target_n,
        balance_taxonomically=not args.no_balance,
        output_dir=args.output_dir,
    )
