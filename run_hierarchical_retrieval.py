import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.hierarchical_retrieval import run_hierarchical_retrieval_experiment
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS


def main(output_dir=None, scenario="joint", query_mode="text", joint_alpha=0.5):
    print("=" * 60)
    print(" HIERARCHICAL (CLUSTER-ROUTED) RETRIEVAL EXPERIMENT ")
    print("=" * 60)
    print("[*] Tests the approximate/indexed kNN framing from Proposal Sec. 2.3")
    print("    against the flat/brute-force search actually used in Bozeat (Sec. 3.3.4).")
    print(f"[*] Query mode: {query_mode}" + (f" (alpha={joint_alpha})" if query_mode == "joint" else ""))

    if output_dir is None:
        # Joint-query output goes to a separate subfolder, same reasoning
        # as run_bozeat_grid.py -- comparable side by side, not overwritten.
        subdir = "hierarchical_retrieval" if query_mode == "text" else "hierarchical_retrieval_joint_query"
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", subdir)

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    csv_path, grid_path = run_hierarchical_retrieval_experiment(
        evaluator, pruning_levels=PRUNING_LEVELS_FOCUSED, scenario=scenario,
        query_mode=query_mode, joint_alpha=joint_alpha, output_dir=output_dir,
    )
    print(f"\n[+] Done.\n    Decomposition CSV: {csv_path}\n    Image grid: {grid_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hierarchical cluster-routed retrieval experiment.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    parser.add_argument("--query_mode", type=str, default="text", choices=["text", "joint"])
    parser.add_argument("--joint_alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario, query_mode=args.query_mode, joint_alpha=args.joint_alpha)
