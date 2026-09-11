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

# Image grid COLUMNS (pruning levels rendered as thumbnails): 2.5% steps
# from 0% to 50% -- densest coverage in the transition zone where the
# interesting breed-level collapse dynamics actually happen, per explicit
# request. NOTE: this intentionally does NOT extend to 60-75%, so the
# "total collapse onto one repeated image" endpoint pattern (seen in
# earlier runs) will not appear in this grid -- the full retrieval CURVE
# still covers 0-75% and captures that region numerically.
DEFAULT_DISPLAY_PRUNING_LEVELS = [p for p in PRUNING_LEVELS_FOCUSED if p <= 0.50]


def main(output_dir=None, display_prompts=None, display_pruning_levels=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")
    if display_prompts is None:
        display_prompts = DEFAULT_DISPLAY_PROMPTS
    if display_pruning_levels is None:
        display_pruning_levels = DEFAULT_DISPLAY_PRUNING_LEVELS

    print("=" * 60)
    print(" BOZEAT TEXT->IMAGE RETRIEVAL EXPERIMENT (curated multi-breed subset) ")
    print("=" * 60)
    print(f"[*] Full prompt set ({len(CURATED_CLASSES)} classes, {len(CURATED_TAXONOMY)} coordinate groups)")
    print(f"[*] Image grid display prompts (1/group): {display_prompts}")
    print(f"[*] Image grid display pruning levels ({len(display_pruning_levels)} cols, 2.5% steps, 0-50%): {display_pruning_levels}")
    print(f"[*] Full retrieval curve ({len(PRUNING_LEVELS_FOCUSED)} stages, 2.5% steps, 0-75%)")

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
        display_pruning_levels=display_pruning_levels,
        output_dir=output_dir,
    )

    print(f"\n[+] Done.\n    Results CSV: {csv_path}\n    Image grid: {grid_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bozeat retrieval experiment.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
