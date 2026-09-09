import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.generate_tsne_subcat_no_error import generate_tsne_clean_color_grid
from src.joint_evaluator import JointSpaceEvaluator

# 19-stage pruning schedule (5% increments from 0% to 90%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Clean Concept-Dispersion t-SNE Plot Generation."
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate",
    )
    parser.add_argument(
        "--target_n",
        type=int,
        default=None,
        help="Explicit sample count for evaluation",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Disable taxonomic class balancing",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=15,
        help="Number of image samples per concept class to visualize",
    )
    parser.add_argument(
        "--single_word_prompts",
        action="store_true",
        help="Use single-word concept prompts instead of contextual natural language templates",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for t-SNE plots",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" RUNNING CLEAN CONCEPT DISPERSION t-SNE GENERATION ")
    print("=" * 60)

    if args.output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "tsne")
    else:
        output_dir = args.output_dir

    evaluator = JointSpaceEvaluator(
        sample_frac=args.sample_frac,
        target_n=args.target_n,
        balance_taxonomically=not args.no_balance,
    )

    save_path = generate_tsne_clean_color_grid(
        evaluator=evaluator,
        pruning_levels=PRUNING_LEVELS_5PCT,
        samples_per_class=args.samples_per_class,
        output_dir=output_dir,
        use_single_word_prompts=args.single_word_prompts,
    )

    print(
        f"\n[+] Concept-dispersion t-SNE grid generated successfully:\n    {save_path}"
    )


if __name__ == "__main__":
    main()