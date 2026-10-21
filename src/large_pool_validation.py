"""Larger candidate pool validation.

Every result so far (Bozeat, hierarchical retrieval) searches only the
curated 33-class x 5-images pool (165 candidates) -- a real, repeatedly-
flagged limitation, since a small, mostly-easy-to-distinguish distractor
set makes correct retrieval structurally easier than genuine open-world
search. This module re-runs BOTH qualitative grids against all ~72
downloaded classes instead (not just the curated 33), at a small number
of pruning levels (not the full 31-level sweep -- this pool is ~2x
larger, so the full sweep would cost proportionally more; a handful of
checkpoints is enough to see whether the SAME collapse patterns hold up).

Design notes:
  - The evaluator is built with restrict_classes=None, which naturally
    picks up every class actually present in metadata_processed.csv (no
    hardcoded class list needed, robust to however many of the 72
    intended classes actually downloaded).
  - Bozeat's flat search has no dependency on the curated 9-group
    taxonomy at all, so it's reused completely unmodified here -- only
    the evaluator's candidate pool changes.
  - Hierarchical retrieval's routing DOES depend on coordinate groups,
    which the curated 9-group scheme (src/curated_config.py) doesn't
    cover for the +44 non-curated classes. Rather than inventing a new
    grouping, this uses the FULL taxonomy's own "coordinate" column
    (data/taxonomy_full.json) directly -- every class already has a real
    assignment there (e.g. "Ungulates", "Arachnids", "Marine
    Invertebrates"), just a different, broader granularity than the
    curated scheme. This means "routing group" means something
    different in this validation than in the main pipeline -- noted
    directly on both output plots so this isn't misread as the same
    metric.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.bozeat_experiment import run_bozeat_experiment
from src.hierarchical_retrieval import run_hierarchical_retrieval_experiment

DISPLAY_PROMPTS = [
    "German Shepherd", "Golden Retriever", "Tabby Cat", "Persian Cat",
    "Monarch Butterfly", "Albatross", "Chimpanzee", "Bullfrog", "Lemon",
]
# A smaller pruning schedule than the main pipeline's 31 levels -- this
# pool is roughly 2x larger, so full-resolution sweeps would cost
# proportionally more for validation-only output.
VALIDATION_PRUNING_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


def run_large_pool_validation(output_dir: str | None = None, scenario: str = "joint") -> dict[str, str]:
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "large_pool_validation")
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Building evaluator over the FULL downloaded class set (restrict_classes=None)...")
    evaluator = JointSpaceEvaluator(
        restrict_classes=None,
        balance_taxonomically=True,
        fixed_samples_per_class=5,
    )
    n_classes = evaluator.valid_metadata["specific"].nunique()
    n_images = len(evaluator.valid_metadata)
    print(f"[*] Large pool: {n_classes} classes, {n_images} candidate images "
          f"(vs. 33 classes / 165 images in the main curated pipeline).")

    results = {}

    print("\n[*] Running Bozeat retrieval grid against the large pool...")
    bozeat_csv, bozeat_grid = run_bozeat_experiment(
        evaluator,
        target_prompts=DISPLAY_PROMPTS,
        pruning_levels=VALIDATION_PRUNING_LEVELS,
        display_prompts=DISPLAY_PROMPTS,
        display_pruning_levels=VALIDATION_PRUNING_LEVELS,
        output_dir=os.path.join(output_dir, "bozeat_large_pool"),
    )
    results["bozeat_csv"] = bozeat_csv
    results["bozeat_grid"] = bozeat_grid

    print("\n[*] Deriving coordinate groups from the full taxonomy's own metadata...")
    meta = evaluator.valid_metadata
    group_of_concept = dict(zip(meta["specific"], meta["coordinate"]))
    groups = sorted(meta["coordinate"].unique())
    print(f"[*] {len(groups)} coordinate groups present in the large pool "
          f"(vs. 9 in the curated 33-class scheme): {groups}")

    print("\n[*] Running hierarchical retrieval grid against the large pool...")
    hier_csv, hier_grid = run_hierarchical_retrieval_experiment(
        evaluator,
        pruning_levels=VALIDATION_PRUNING_LEVELS,
        scenario=scenario,
        display_prompts=DISPLAY_PROMPTS,
        display_pruning_levels=VALIDATION_PRUNING_LEVELS,
        group_of_concept=group_of_concept,
        groups=groups,
        output_dir=os.path.join(output_dir, "hierarchical_large_pool"),
    )
    results["hierarchical_csv"] = hier_csv
    results["hierarchical_grid"] = hier_grid

    print(f"\n[+] Large pool validation complete. Outputs in: {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Bozeat + hierarchical retrieval against the full ~72-class pool.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    run_large_pool_validation(output_dir=args.output_dir, scenario=args.scenario)
