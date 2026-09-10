"""The one tSNE visualization for this project.

Previously the codebase had FIVE separate, largely-duplicate tSNE
implementations: src/generate_tsne.py, src/extended_hierarchical_eval.py's
generate_hierarchical_hsv_tsne, and three standalone scripts/modules
(run_tsne_5stages.py + generate_tsne_subcategories.py, run_tSNE_nerr.py +
gen_tSNE_subcat_nerr.py, run_tsne_no_errors.py +
generate_tsne_subcat_no_error.py). All five are superseded by this single
module, scoped to the curated 10-class subset (src/curated_config.py) for
both speed (re-encoding ~30-50 images per pruning level instead of ~7,200)
and interpretability (10 named, well-known classes with a stable color
per class, rather than domain-grouped colormap families that needed a
legend of their own to decode).
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

from src.curated_config import CURATED_CLASSES

# tab10 gives 10 maximally-distinct qualitative colors -- exactly matching
# the 10 curated classes, one color each, no palette-family logic needed.
_CMAP = plt.get_cmap("tab10")
CLASS_COLORS = {cls: _CMAP(i) for i, cls in enumerate(CURATED_CLASSES)}


def generate_curated_tsne_grid(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
) -> str:
    """Generates one tSNE grid: a panel per pruning level (default 6
    representative stages spanning 0-70%), points colored by specific
    class, fit globally across all stacked stages so positions are
    directly comparable panel to panel.
    """
    if pruning_levels_to_show is None:
        pruning_levels_to_show = [0.0, 0.15, 0.30, 0.45, 0.60, 0.70]
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    spec_col = "specific" if "specific" in evaluator.valid_metadata.columns else "concept"
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
    n_cols = min(3, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.0 * n_rows))
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
                pts[:, 0], pts[:, 1], s=40, color=CLASS_COLORS[cls],
                alpha=0.85, edgecolor="white", linewidth=0.4, label=cls,
            )

        ax.set_title(f"{p_level * 100:.1f}% Pruned", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    for ax in axes_flat[n_stages:]:
        ax.axis("off")

    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CURATED_CLASSES]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Joint Embedding Space t-SNE Across Pruning Levels ({scenario} scenario, curated 10-class subset)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_path = os.path.join(output_dir, "tsne_curated.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] tSNE grid saved to: {save_path}")
    return save_path
