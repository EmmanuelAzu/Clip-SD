import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import src.bozeat_experiment as bozeat_module
from src.joint_evaluator import JointSpaceEvaluator

PRUNING_LEVELS_5PCT = getattr(
    bozeat_module,
    "PRUNING_LEVELS_5PCT",
    [round(x, 2) for x in np.arange(0.00, 0.91, 0.05)],
)
DEFAULT_TARGET_PROMPTS = getattr(bozeat_module, "DEFAULT_TARGET_PROMPTS", None)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[*] Executing {len(PRUNING_LEVELS_5PCT)}-stage Bozeat Visual Retrieval Grid on: {device}"
    )

    metadata_path = os.path.join(
        PROJECT_ROOT, "data", "processed", "metadata_processed.csv"
    )
    output_dir = os.path.join(
        PROJECT_ROOT, "data", "results", "bozeat_experiment"
    )

    # JointSpaceEvaluator already builds a real CLIPPruningEngine on
    # self.base_model in its own __init__ (src/joint_evaluator.py) -- no
    # extra resolution/attachment step is needed.
    evaluator = JointSpaceEvaluator(
        metadata_path=metadata_path,
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True,
    )

    kwargs = {
        "pruning_levels": PRUNING_LEVELS_5PCT,
        "output_dir": output_dir,
    }
    if DEFAULT_TARGET_PROMPTS is not None:
        kwargs["target_prompts"] = DEFAULT_TARGET_PROMPTS

    bozeat_module.run_bozeat_visual_grid(evaluator, **kwargs)


if __name__ == "__main__":
    main()
