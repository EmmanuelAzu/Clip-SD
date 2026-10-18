import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_tsne_joint import generate_joint_tsne_grid
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS


def main(output_dir=None, scenario="joint"):
    print("=" * 60)
    print(" JOINT (IMAGE + TEXT) t-SNE GRID ")
    print("=" * 60)
    print(f"[*] Classes ({len(CURATED_CLASSES)}): {CURATED_CLASSES}")
    print(f"[*] Pruning levels: {len(PRUNING_LEVELS_FOCUSED)} (0-75%, 2.5% steps)")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    save_path = generate_joint_tsne_grid(evaluator, scenario=scenario, output_dir=output_dir)
    print(f"\n[+] Done: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the joint (image+text) embedding tSNE grid.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario)
