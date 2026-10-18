"""Simplified confusion matrix for a handful of representative classes.

The full error-taxonomy heatmap (src/generate_analysis_plots.py) answers
"what KIND of hierarchy-level error happens" across all 33 classes at
once. This answers the more literal, easier-to-read question: for a
handful of familiar classes specifically, which classes get confused with
which, at a few pruning levels in the 0-35% transition zone -- a classic
NxN confusion matrix (true label rows, predicted label columns), not a
taxonomy-tier breakdown.

Direction: image -> text, the SAME 33-way classification direction as the
core evaluation (Top-1/5/10, error taxonomy) -- each of the N selected
classes' images is classified against the full 33-class vocabulary (not
just the N selected), then the results are filtered down to a small,
readable NxN matrix for just the selected classes' rows.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEFAULT_SELECT_CLASSES = [
    "German Shepherd", "Tabby Cat", "Chimpanzee", "Bullfrog", "Albatross", "Lemon",
]


def generate_select_class_confusion_matrices(
    evaluator,
    select_classes: list[str] | None = None,
    pruning_levels: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
) -> str:
    if select_classes is None:
        select_classes = DEFAULT_SELECT_CLASSES
    if pruning_levels is None:
        pruning_levels = [0.0, 0.10, 0.20, 0.30, 0.35]
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    n_stages = len(pruning_levels)
    fig, axes = plt.subplots(1, n_stages, figsize=(4.2 * n_stages, 4.6), dpi=200)
    if n_stages == 1:
        axes = [axes]

    for ax, p_level in zip(axes, pruning_levels):
        print(f"[*] Select-class confusion matrix: extracting features at {p_level * 100:.1f}% pruning...")
        if scenario == "joint":
            amount = p_level
        elif scenario == "vision_only":
            amount = {"text": 0.0, "vision": p_level}
        else:
            amount = {"text": p_level, "vision": 0.0}

        pruned_model = evaluator.pruning_engine.get_pruned_model(
            amount=amount, encoder_type="joint", target_area="full"
        )
        img_feats, text_feats, labels, unique_concepts, _ = evaluator._extract_joint_features(pruned_model)

        # Full 33-way classification (same direction/vocabulary as core
        # eval) -- every image still competes against all 33 concepts,
        # not just the selected ones, so this reflects real confusion
        # with the FULL taxonomy, not an artificially easier N-way task.
        sim_matrix = img_feats @ text_feats.T
        preds = torch.argmax(sim_matrix, dim=1).numpy()
        pred_concepts = [unique_concepts[p] for p in preds]

        # Filter to only the selected classes' TRUE rows, for readability
        select_mask = [l in select_classes for l in labels]
        true_sel = [l for l, m in zip(labels, select_mask) if m]
        pred_sel = [p for p, m in zip(pred_concepts, select_mask) if m]

        # Build the NxN matrix: rows = true (selected classes only),
        # columns = predicted, but collapse any predicted class outside
        # the selected set into an explicit "Other" column so off-target
        # predictions aren't silently dropped.
        cm = pd.DataFrame(0, index=select_classes, columns=select_classes + ["Other"])
        for t, p in zip(true_sel, pred_sel):
            col = p if p in select_classes else "Other"
            cm.loc[t, col] += 1

        cm_norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)

        sns.heatmap(
            cm_norm, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
            cbar=(ax is axes[-1]), square=True, annot_kws={"size": 8},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{p_level * 100:.1f}% Pruned", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True" if ax is axes[0] else "", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle(
        f"Confusion Matrix, Select Classes ({', '.join(select_classes)})\n"
        "Proportion of each true class's predictions landing on each column (row-normalized)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "select_class_confusion_matrices.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Select-class confusion matrices saved to: {save_path}")
    return save_path
