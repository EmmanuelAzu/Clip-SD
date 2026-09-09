import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import procrustes
from sklearn.manifold import TSNE
import torch
import clip

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def build_distinct_category_palette(categories: list[str]) -> dict[str, tuple]:
    """Assigns maximally distinct colors across taxonomy categories."""
    cmaps = [plt.cm.tab10, plt.cm.Set2, plt.cm.Set1, plt.cm.Dark2]
    colors = []
    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])

    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}


def extract_joint_image_embeddings(model, metadata, preprocess, device):
    """Extracts L2-normalized image embeddings residing in the joint multimodal projection space."""
    model.eval()

    img_tensors = []
    valid_indices = []

    for idx, row in metadata.iterrows():
        img_path = row.get("filepath", None)
        if not img_path or pd.isna(img_path):
            if "filename" in row and pd.notna(row["filename"]):
                img_path = os.path.join(PROJECT_ROOT, "data", "raw", row["filename"])
            else:
                continue

        if not os.path.isabs(img_path):
            img_path = os.path.join(PROJECT_ROOT, img_path)

        if os.path.exists(img_path):
            try:
                img_tensors.append(preprocess(Image.open(img_path).convert("RGB")))
                valid_indices.append(idx)
            except Exception:
                pass

    filtered_meta = metadata.iloc[valid_indices].copy().reset_index(drop=True)
    if len(img_tensors) == 0:
        raise ValueError("[!] No valid image files were found to extract joint embeddings.")

    img_batch = torch.stack(img_tensors).to(device)

    with torch.no_grad():
        img_feats = model.encode_image(img_batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    return img_feats.float().cpu().numpy(), filtered_meta


def generate_shared_progression_grid(
    harness=None,
    metadata_path=None,
    output_plot_path=None,
    pruning_levels=[0.0, 0.25, 0.50, 0.75],
):
    """Generates a multi-panel t-SNE trajectory grid of image embeddings in joint space.

    Uses Procrustes alignment against pristine baseline to prevent coordinate rotation artifacts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_plot_path is None:
        output_plot_path = os.path.join(
            PROJECT_ROOT, "data", "results", "joint_space_progression.png"
        )

    if harness is not None:
        metadata = harness.metadata.reset_index(drop=True)
        base_model = harness.base_model
        preprocess = harness.preprocess
    else:
        if metadata_path is None:
            metadata_path = os.path.join(
                PROJECT_ROOT, "data", "metadata_processed.csv"
            )
        metadata = pd.read_csv(metadata_path)
        base_model, preprocess = clip.load("ViT-B/32", device=device)

    # Color grouping tier: domain or coordinate
    cat_col = next((c for c in ["coordinate", "domain", "superordinate"] if c in metadata.columns), "coordinate")
    unique_cats = sorted(list(metadata[cat_col].unique()))
    cat_palette = build_distinct_category_palette(unique_cats)

    print(f"[*] Extracting joint image space progression across pruning levels: {pruning_levels}...")
    feats_by_level = {}

    for p in pruning_levels:
        if harness is not None and hasattr(harness, "pruning_engine"):
            pruned = harness.pruning_engine.get_pruned_model(
                amount=p, encoder_type="joint", target_area="full"
            )
        else:
            from src.pruning_engine import CLIPPruningEngine
            import copy
            model_copy = copy.deepcopy(base_model)
            engine = CLIPPruningEngine(model_copy)
            pruned = engine.get_pruned_model(amount=p, encoder_type="joint")

        V_img, meta = extract_joint_image_embeddings(pruned, metadata, preprocess, device)
        feats_by_level[p] = (V_img, meta)

    # Base t-SNE computation at 0% atrophy
    V0, _ = feats_by_level[pruning_levels[0]]
    perp = min(30, max(5, (len(V0) - 1) // 3))

    tsne_base = TSNE(
        n_components=2,
        perplexity=perp,
        metric="cosine",
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    base_raw = tsne_base.fit_transform(V0)

    # Center and normalize baseline
    base_centered = base_raw - np.mean(base_raw, axis=0)
    norm = np.linalg.norm(base_centered)
    base_coords = base_centered / (norm if norm > 0 else 1.0)

    coords_by_level = {pruning_levels[0]: base_coords}

    # Procrustes alignment for subsequent pruning levels
    for p in pruning_levels[1:]:
        V_p, _ = feats_by_level[p]
        tsne_p = TSNE(
            n_components=2,
            perplexity=perp,
            metric="cosine",
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        raw_p = tsne_p.fit_transform(V_p)

        _, aligned_p, _ = procrustes(base_coords, raw_p)
        coords_by_level[p] = aligned_p

    # Compute global axis boundaries across all panels
    all_x = np.concatenate([c[:, 0] for c in coords_by_level.values()])
    all_y = np.concatenate([c[:, 1] for c in coords_by_level.values()])
    x_margin = (max(all_x) - min(all_x)) * 0.08
    y_margin = (max(all_y) - min(all_y)) * 0.08
    xlim = (min(all_x) - x_margin, max(all_x) + x_margin)
    ylim = (min(all_y) - y_margin, max(all_y) + y_margin)

    # Render multi-panel progression grid
    num_plots = len(pruning_levels)
    ncols = min(4, num_plots)
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, p in enumerate(pruning_levels):
        ax = axes_flat[idx]
        img_coords = coords_by_level[p]
        _, meta = feats_by_level[p]

        for cat in unique_cats:
            mask = (meta[cat_col] == cat).values
            if mask.any():
                ax.scatter(
                    img_coords[mask, 0],
                    img_coords[mask, 1],
                    c=[cat_palette[cat]],
                    alpha=0.85,
                    s=35,
                    zorder=2,
                    edgecolors="black",
                    linewidths=0.3,
                    label=cat,
                )

        ax.set_title(
            f"Joint Space Atrophy: {int(p * 100)}%",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("Aligned Cosine Dim 1", fontsize=9)
        ax.set_ylabel("Aligned Cosine Dim 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for extra_idx in range(num_plots, len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    # Category Legend
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cat_palette[cat],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
            label=cat,
        )
        for cat in unique_cats
    ]

    fig.legend(
        handles=legend_handles,
        title=f"Categories ({cat_col.capitalize()})",
        title_fontsize="10",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(6, len(unique_cats)),
        frameon=True,
        facecolor="white",
        fontsize=9,
    )

    plt.suptitle(
        "Joint Multimodal Space Dissolution Across Pruning Levels (Procrustes Aligned)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved joint space progression plot to: {output_plot_path}")
    return output_plot_path