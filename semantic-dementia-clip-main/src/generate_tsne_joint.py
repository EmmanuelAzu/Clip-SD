"""The genuinely joint (image AND text) embedding tSNE.

src/generate_tsne_curated.py only ever projects IMAGE embeddings (its own
title previously claimed "Joint Embedding Space", which was misleading --
fixed there separately). This module fits ONE shared t-SNE across BOTH
the 165 image embeddings AND the 33 text-prompt embeddings together, so
you can see where each class's text anchor sits relative to its own
image cluster -- and, critically, whether that anchor visibly drifts away
from its cluster as pruning increases, which is a direct geometric
explanation for retrieval failures (Bozeat, hierarchical routing) at the
same pruning level.

Marker shape carries the image/text distinction; color still carries
class identity (same hue-family scheme as the image-only tSNE), so a
text anchor's color should match its own image cluster's color at 0%
pruning, by construction.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_TAXONOMY, CURATED_CLASSES, build_class_colors


def generate_joint_tsne_grid(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
    n_cols: int = 6,
) -> str:
    if pruning_levels_to_show is None:
        from src.curated_config import PRUNING_LEVELS_FOCUSED
        pruning_levels_to_show = PRUNING_LEVELS_FOCUSED
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    class_colors = build_class_colors()

    all_feats = []
    true_labels = None
    n_images = n_texts = None

    for p_level in pruning_levels_to_show:
        print(f"[*] Joint tSNE: extracting features at {p_level * 100:.1f}% pruning...")
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
        img_feats_np = img_feats.detach().cpu().numpy() if isinstance(img_feats, torch.Tensor) else img_feats
        text_feats_np = text_feats.detach().cpu().numpy() if isinstance(text_feats, torch.Tensor) else text_feats

        # Stack images first, then texts, per stage -- consistent order
        # lets us slice each stage's block apart cleanly after the fit.
        stage_feats = np.vstack([img_feats_np, text_feats_np])
        all_feats.append(stage_feats)
        if true_labels is None:
            true_labels = labels
            text_labels = unique_concepts
            n_images = len(labels)
            n_texts = len(unique_concepts)

    X_total = np.vstack(all_feats)
    n_per_stage = n_images + n_texts
    perplexity = min(30, max(5, n_per_stage // 3))

    print(f"[*] Fitting global joint t-SNE across {len(pruning_levels_to_show)} stacked stages "
          f"({X_total.shape[0]} points total: {n_images} images + {n_texts} text prompts per stage, "
          f"perplexity={perplexity})...")
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
        img_points = X_2d[start:start + n_images]
        text_points = X_2d[start + n_images:start + n_per_stage]

        for cls in CURATED_CLASSES:
            img_mask = [lbl == cls for lbl in true_labels]
            if any(img_mask):
                pts = img_points[img_mask]
                ax.scatter(pts[:, 0], pts[:, 1], s=20, color=class_colors[cls],
                           alpha=0.8, edgecolor="white", linewidth=0.3, marker="o", zorder=2)
            if cls in text_labels:
                t_idx = text_labels.index(cls)
                tp = text_points[t_idx]
                ax.scatter(tp[0], tp[1], s=90, color=class_colors[cls],
                           edgecolor="black", linewidth=0.8, marker="*", zorder=3)

        ax.set_title(f"{p_level * 100:.1f}%", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    for ax in axes_flat[n_stages:]:
        ax.axis("off")

    handles = []
    labels_list = []
    for group, members in CURATED_TAXONOMY.items():
        handles.append(mpatches.Patch(color="white", alpha=0))
        labels_list.append(f"— {group} —")
        for m in members:
            handles.append(mpatches.Patch(color=class_colors[m]))
            labels_list.append(m)

    marker_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=6, label="Image embedding"),
        mlines.Line2D([], [], color="gray", marker="*", linestyle="None", markersize=11,
                       markeredgecolor="black", label="Text-prompt embedding"),
    ]

    legend1 = fig.legend(
        handles, labels_list, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=7, frameon=False, ncol=1, title="Coordinate Group / Species",
        title_fontsize=8,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=marker_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
        fontsize=8, frameon=False, title="Marker Shape", title_fontsize=8.5,
    )

    fig.suptitle(
        f"Joint (Image + Text) Embedding Space t-SNE Across All Pruning Levels ({scenario} scenario)\n"
        "Circles = image embeddings, stars = text-prompt embeddings -- watch whether a class's star "
        "drifts away from its own circle cluster as pruning increases",
        fontsize=12.5, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    save_path = os.path.join(output_dir, "tsne_joint.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Joint tSNE grid saved to: {save_path}")
    return save_path
