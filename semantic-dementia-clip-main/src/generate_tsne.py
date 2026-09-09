import os
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 19-stage pruning schedule (5% increments from 0% to 90%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()
]


def generate_joint_hierarchical_tsne(
    harness,
    pruning_levels: list[float] = PRUNING_LEVELS_5PCT,
    output_dir: str | None = None,
    n_cols: int = 6,
) -> str:
    """Plots joint space t-SNE trajectories dynamically scaled across atrophy levels using HSV color mapping."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "tsne")
    os.makedirs(output_dir, exist_ok=True)

    meta = harness.metadata.reset_index(drop=True)

    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in meta.columns),
        "specific",
    )
    coord_col = next(
        (c for c in ["coordinate", "category"] if c in meta.columns),
        "coordinate",
    )
    super_col = next(
        (c for c in ["superordinate", "domain", "macro"] if c in meta.columns),
        "superordinate",
    )

    super_classes = sorted(list(meta[super_col].unique()))
    coord_classes = sorted(list(meta[coord_col].unique()))
    spec_classes = sorted(list(meta[spec_col].unique()))

    hue_map = {
        s: i / max(1, len(super_classes)) for i, s in enumerate(super_classes)
    }

    color_lut = {}
    for _, row in meta.iterrows():
        sp, co, su = row[spec_col], row[coord_col], row[super_col]
        if sp not in color_lut:
            h = hue_map[su]
            co_idx = coord_classes.index(co)
            s = 0.5 + 0.5 * (co_idx / max(1, len(coord_classes)))
            sp_idx = spec_classes.index(sp)
            v = 0.35 + 0.55 * (sp_idx / max(1, len(spec_classes)))
            color_lut[sp] = mcolors.hsv_to_rgb((h, s, v))

    n_plots = len(pruning_levels)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for idx, p_level in enumerate(pruning_levels):
        if hasattr(harness, "pruning_engine"):
            pruned_model = harness.pruning_engine.get_pruned_model(
                amount=p_level, encoder_type="joint", target_area="full"
            )
        else:
            pruned_model = harness._apply_pruning(harness.base_model, p_level)

        img_feats, text_feats, labels, _, _ = harness._extract_joint_features(
            pruned_model
        )

        all_feats = torch.cat([img_feats, text_feats], dim=0).detach().cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        tsne_coords = tsne.fit_transform(all_feats)

        n_img = len(img_feats)
        img_coords = tsne_coords[:n_img]
        text_coords = tsne_coords[n_img:]

        ax = axes_flat[idx]
        img_colors = [color_lut[c] for c in labels]

        ax.scatter(
            img_coords[:, 0],
            img_coords[:, 1],
            c=img_colors,
            marker="o",
            alpha=0.7,
            s=25,
            label="Image Embeddings",
        )
        ax.scatter(
            text_coords[:, 0],
            text_coords[:, 1],
            c="black",
            marker="X",
            s=70,
            edgecolors="white",
            linewidth=1,
            label="Text Prompts",
        )

        ax.set_title(f"Atrophy: {int(p_level * 100)}%", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for unused in range(n_plots, len(axes_flat)):
        axes_flat[unused].axis("off")

    plt.suptitle(
        "Full Dataset Joint Feature Space Trajectory Under Hierarchical HSV Color Mapping",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Distinct filename from extended_hierarchical_eval.generate_hierarchical_hsv_tsne():
    # both functions previously wrote to the same path and silently overwrote
    # each other when run_pipeline.py called both with the same output_dir.
    save_path = os.path.join(output_dir, "joint_tsne_with_text_prompts.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved hierarchical HSV t-SNE plot to: {save_path}")
    return save_path