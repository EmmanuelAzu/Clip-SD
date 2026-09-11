import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_hierarchy_graph import generate_hierarchy_dendrogram
from src.curated_config import CURATED_CLASSES, IMAGES_PER_CLASS


def main(output_dir=None, pruning_levels=None, scenario="joint"):
    print("=" * 60)
    print(" HIERARCHICAL DENDROGRAM (class-centroid clustering) ")
    print("=" * 60)

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    if pruning_levels is None:
        pruning_levels = [0.0, 0.25, 0.50, 0.75]

    print(f"[*] Pruning levels shown: {pruning_levels}")

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    save_path = generate_hierarchy_dendrogram(
        evaluator, pruning_levels_to_show=pruning_levels, scenario=scenario, output_dir=output_dir
    )
    print(f"\n[+] Done: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the hierarchical dendrogram visualization.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario)
