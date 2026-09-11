import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.bozeat_experiment import run_bozeat_experiment
from src.curated_config import (
    CURATED_CLASSES,
    CURATED_TAXONOMY,
    PRUNING_LEVELS_FOCUSED,
    IMAGES_PER_CLASS,
)

# The full retrieval CURVE covers all 33 curated classes (cheap regardless
# of count, since the candidate pool is small either way). The qualitative
# image GRID needs a small row count to stay readable, so it uses one
# representative breed/species per coordinate group (9 rows) by default.
DEFAULT_DISPLAY_PROMPTS = [members[0] for members in CURATED_TAXONOMY.values()]


def main(output_dir=None, display_prompts=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")
    if display_prompts is None:
        display_prompts = DEFAULT_DISPLAY_PROMPTS

    print("=" * 60)
    print(" BOZEAT TEXT->IMAGE RETRIEVAL EXPERIMENT (curated multi-breed subset) ")
    print("=" * 60)
    print(f"[*] Full prompt set ({len(CURATED_CLASSES)} classes, {len(CURATED_TAXONOMY)} coordinate groups)")
    print(f"[*] Image grid display prompts (1/group): {display_prompts}")
    print(f"[*] Pruning Grid ({len(PRUNING_LEVELS_FOCUSED)} stages, 2.5% steps, 0-75%)")

    # restrict_classes here makes BOTH the query prompts and the retrieval
    # candidate pool the curated multi-breed subset -- this is what makes
    # the experiment fast. fixed_samples_per_class=5 keeps every specific
    # class equally weighted regardless of how many images are actually
    # available, per the within-group-collapse study design.
    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    csv_path, grid_path = run_bozeat_experiment(
        evaluator,
        target_prompts=CURATED_CLASSES,
        pruning_levels=PRUNING_LEVELS_FOCUSED,
        display_prompts=display_prompts,
        output_dir=output_dir,
    )

    print(f"\n[+] Done.\n    Results CSV: {csv_path}\n    Image grid: {grid_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bozeat retrieval experiment.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
