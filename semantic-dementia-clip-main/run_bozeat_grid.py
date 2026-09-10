import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.bozeat_experiment import run_bozeat_experiment
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED


def main(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")

    print("=" * 60)
    print(" BOZEAT TEXT->IMAGE RETRIEVAL EXPERIMENT (curated 10-class subset) ")
    print("=" * 60)
    print(f"[*] Prompts ({len(CURATED_CLASSES)}): {CURATED_CLASSES}")
    print(f"[*] Pruning Grid ({len(PRUNING_LEVELS_FOCUSED)} stages, 2.5% steps, 0-70%)")

    # restrict_classes here makes BOTH the query prompts and the retrieval
    # candidate pool the curated 10-class subset -- this is what makes the
    # experiment fast (re-encoding ~30-50 images per level instead of the
    # full balanced dataset).
    evaluator = JointSpaceEvaluator(restrict_classes=CURATED_CLASSES, balance_taxonomically=True)

    csv_path, grid_path = run_bozeat_experiment(
        evaluator,
        target_prompts=CURATED_CLASSES,
        pruning_levels=PRUNING_LEVELS_FOCUSED,
        output_dir=output_dir,
    )

    print(f"\n[+] Done.\n    Results CSV: {csv_path}\n    Image grid: {grid_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bozeat retrieval experiment.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
