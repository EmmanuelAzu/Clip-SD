import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.generate_tsne_subcategories import (
    PRUNING_LEVELS_5PCT,
    generate_tsne_grid_with_key,
)
from src.joint_evaluator import JointSpaceEvaluator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[*] Executing {len(PRUNING_LEVELS_5PCT)}-stage t-SNE grid pipeline on: {device}"
    )

    metadata_path = os.path.join(
        PROJECT_ROOT, "data", "processed", "metadata_processed.csv"
    )
    output_dir = os.path.join(PROJECT_ROOT, "data", "results", "tsne")

    evaluator = JointSpaceEvaluator(
        metadata_path=metadata_path,
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True,
    )

    generate_tsne_grid_with_key(
        evaluator,
        pruning_levels=PRUNING_LEVELS_5PCT,
        samples_per_class=15,
        output_dir=output_dir,
        n_cols=6,
    )


if __name__ == "__main__":
    main()