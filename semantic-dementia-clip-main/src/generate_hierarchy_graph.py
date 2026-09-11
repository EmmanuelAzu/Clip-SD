"""Hierarchical dendrogram of the embedding space, at a few representative
pruning levels, to directly visualize the SHAPE of collapse -- distinct
from the t-SNE (geometric layout) and the cross-category plot (aggregate
error rates). This answers "do all dog breeds merge with each other
before any of them merges with a cat, and does the whole Felines group
then merge with Primates before Insects?" directly, as a tree, rather
than requiring that structure to be inferred from scatter positions or
error-rate numbers.

Method: at each selected pruning level, compute the per-class mean
embedding (centroid) across that class's images, then run agglomerative
hierarchical clustering (scipy) on the 33 centroids. Leaf labels are
colored by true coordinate group (same hue-family scheme as the tSNE and
Bozeat plots) so a reader can see at a glance whether early merges happen
WITHIN a color family (expected, healthy hierarchical structure) or
ACROSS color families (evidence the model has already lost the
coordinate-level boundary).
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_TAXONOMY, CURATED_CLASSES, build_class_colors


def _class_to_group(cls: str) -> str:
    for group, members in CURATED_TAXONOMY.items():
        if cls in members:
            return group
    return "Unknown"


def generate_hierarchy_dendrogram(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    linkage_method: str = "average",
    output_dir: str | None = None,
) -> str:
    if pruning_levels_to_show is None:
        pruning_levels_to_show = [0.0, 0.25, 0.50, 0.75]
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    class_colors = build_class_colors()

    n_stages = len(pruning_levels_to_show)
    fig, axes = plt.subplots(1, n_stages, figsize=(6.5 * n_stages, 9), dpi=200)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, p_level in zip(axes_flat, pruning_levels_to_show):
        print(f"[*] Hierarchy graph: extracting features at {p_level * 100:.1f}% pruning...")
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

        # Per-class centroid embeddings
        centroids = []
        centroid_labels = []
        for cls in CURATED_CLASSES:
            mask = [lbl == cls for lbl in labels]
            if not any(mask):
                continue
            centroids.append(img_feats_np[mask].mean(axis=0))
            centroid_labels.append(cls)
        centroids = np.vstack(centroids)

        # Cosine distance is the natural choice here since the embeddings
        # are L2-normalized and retrieval itself is cosine-similarity
        # based -- using it for clustering keeps this consistent with
        # what the rest of the pipeline actually measures.
        dist = pdist(centroids, metric="cosine")
        Z = linkage(dist, method=linkage_method)

        dendrogram(
            Z, labels=centroid_labels, ax=ax, orientation="left",
            leaf_font_size=8, color_threshold=0,
            above_threshold_color="#888888",
        )
        ax.set_title(f"{p_level * 100:.1f}% Pruned", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cosine Distance (linkage)", fontsize=9)

        # Recolor leaf tick labels by true coordinate group
        for tick_label in ax.get_ymajorticklabels():
            cls_name = tick_label.get_text()
            if cls_name in class_colors:
                tick_label.set_color(class_colors[cls_name])
                tick_label.set_fontweight("bold")

    fig.suptitle(
        "Hierarchical Clustering of Class Centroids Across Pruning Levels\n"
        "(leaf label color = true coordinate group -- early within-color merges = "
        "healthy hierarchy retained; early cross-color merges = structure already lost)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "hierarchy_dendrogram.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Hierarchy dendrogram saved to: {save_path}")
    return save_path
