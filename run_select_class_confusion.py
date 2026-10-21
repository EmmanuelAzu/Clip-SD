import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.select_class_confusion import generate_select_class_confusion_matrices, DEFAULT_SELECT_CLASSES
from src.curated_config import CURATED_CLASSES, IMAGES_PER_CLASS


def main(output_dir=None, select_classes=None, pruning_levels=None, scenario="joint"):
    print("=" * 60)
    print(" SELECT-CLASS CONFUSION MATRIX (0-35% pruning) ")
    print("=" * 60)
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )
    save_path = generate_select_class_confusion_matrices(
        evaluator, select_classes=select_classes, pruning_levels=pruning_levels,
        scenario=scenario, output_dir=output_dir,
    )
    print(f"\n[+] Done: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a simplified confusion matrix for select classes.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="Override the default select classes.")
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, select_classes=args.classes, scenario=args.scenario)
