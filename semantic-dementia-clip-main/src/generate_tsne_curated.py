"""The one tSNE visualization for this project.

Shows EVERY pruning level in the schedule (not a subsampled set), colored
by coordinate-group hue family with a distinct shade per specific
breed/species within that group (src/curated_config.py::build_class_colors)
-- so a viewer can tell at a glance which points belong to the same
species (near-identical shade), the same broader group (same hue family),
or a different group entirely (different hue), without needing to
cross-reference a class-by-class legend.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_TAXONOMY, CURATED_CLASSES, build_class_colors


def generate_curated_tsne_grid(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
    n_cols: int = 6,
) -> str:
    """Generates one tSNE grid: one panel per pruning level (default:
    every level in the schedule), points colored by specific class within
    a coordinate-group hue family, fit globally across all stacked stages
    so positions are directly comparable panel to panel. Legend is always
    included, per explicit request, organized by coordinate group.
    """
    if pruning_levels_to_show is None:
        from src.curated_config import PRUNING_LEVELS_FOCUSED
        pruning_levels_to_show = PRUNING_LEVELS_FOCUSED
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    class_colors = build_class_colors()
    meta = evaluator.valid_metadata.reset_index(drop=True)

    all_feats = []
    true_labels = None

    for p_level in pruning_levels_to_show:
        print(f"[*] tSNE: extracting features at {p_level * 100:.1f}% pruning...")
        if scenario == "joint":
            amount = p_level
        elif scenario == "vision_only":
            amount = {"text": 0.0, "vision": p_level}
        else:
            amount = {"text": p_level, "vision": 0.0}

        pruned_model = evaluator.pruning_engine.get_pruned_model(
            amount=amount, encoder_type="joint", target_area="full"
        )
        img_feats, _, labels, _, _ = evaluator._extract_joint_features(pruned_model)
        img_feats_np = img_feats.detach().cpu().numpy() if isinstance(img_feats, torch.Tensor) else img_feats
        all_feats.append(img_feats_np)
        if true_labels is None:
            true_labels = labels

    X_total = np.vstack(all_feats)
    n_per_stage = len(true_labels)
    perplexity = min(30, max(5, n_per_stage // 3))

    print(f"[*] Fitting global t-SNE across {len(pruning_levels_to_show)} stacked stages "
          f"({X_total.shape[0]} points total, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    X_2d = tsne.fit_transform(X_total)

    n_stages = len(pruning_levels_to_show)
    n_cols = min(n_cols, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    xlim = (X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5)
    ylim = (X_2d[:, 1].min() - 5, X_2d[:, 1].max() + 5)

    for stage_idx, (ax, p_level) in enumerate(zip(axes_flat, pruning_levels_to_show)):
        start = stage_idx * n_per_stage
        end = start + n_per_stage
        stage_points = X_2d[start:end]

        for cls in CURATED_CLASSES:
            cls_mask = [lbl == cls for lbl in true_labels]
            if not any(cls_mask):
                continue
            pts = stage_points[cls_mask]
            ax.scatter(
                pts[:, 0], pts[:, 1], s=22, color=class_colors[cls],
                alpha=0.85, edgecolor="white", linewidth=0.3,
            )

        ax.set_title(f"{p_level * 100:.1f}%", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    for ax in axes_flat[n_stages:]:
        ax.axis("off")

    # Legend organized by coordinate group -- shows the hue-family
    # structure directly (one legend "row" per group, shades within it),
    # rather than a flat 33-entry alphabetical list.
    handles = []
    labels_list = []
    for group, members in CURATED_TAXONOMY.items():
        handles.append(mpatches.Patch(color="white", alpha=0))  # spacer/header
        labels_list.append(f"— {group} —")
        for m in members:
            handles.append(mpatches.Patch(color=class_colors[m]))
            labels_list.append(m)

    fig.legend(
        handles, labels_list, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=7, frameon=False, ncol=1, title="Coordinate Group / Species",
        title_fontsize=8,
    )

    fig.suptitle(
        f"Joint Embedding Space t-SNE Across All Pruning Levels ({scenario} scenario)\n"
        "Color = coordinate group (hue family) + specific breed/species (shade)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    save_path = os.path.join(output_dir, "tsne_curated.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] tSNE grid saved to: {save_path}")
    return save_path
