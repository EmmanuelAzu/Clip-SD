import os
import sys
import matplotlib.cm as cm
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

# Preferred colormaps per superordinate domain
DOMAIN_CMAP_PREFERENCES = {
    "animal": "Greens",
    "fauna": "Greens",
    "living": "Greens",
    "vehicle": "Reds",
    "transport": "Reds",
    "artifact": "Blues",
    "object": "Blues",
    "tool": "Oranges",
    "fruit": "Purples",
    "food": "YlOrBr",
    "vegetable": "YlGn",
}

FALLBACK_CMAPS = [
    "Greens", "Reds", "Blues", "Oranges", "Purples",
    "YlOrBr", "PuBu", "YlGn", "RdPu", "GnBu",
]


def _get_taxonomy_mapping(metadata: pd.DataFrame) -> tuple[str, str, str]:
    """Resolves taxonomy column names across dataset schema variants."""
    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in metadata.columns),
        "specific",
    )
    coord_col = next(
        (c for c in ["coordinate", "basic", "category"] if c in metadata.columns),
        "coordinate",
    )
    super_col = next(
        (c for c in ["superordinate", "domain", "macro"] if c in metadata.columns),
        "superordinate",
    )
    return spec_col, coord_col, super_col


def build_hierarchical_concept_palette(
    meta: pd.DataFrame, spec_col: str, super_col: str
) -> tuple[dict[str, tuple], dict[str, list[str]]]:
    """Assigns distinct colormap families to superordinate domains and shades to concepts."""
    unique_super = sorted(list(meta[super_col].unique()))
    concept_to_color = {}
    super_to_concepts = {}
    used_cmaps = set()

    for idx, super_cat in enumerate(unique_super):
        super_str = str(super_cat).lower()
        matched_cmap = next(
            (
                cmap
                for key, cmap in DOMAIN_CMAP_PREFERENCES.items()
                if key in super_str and cmap not in used_cmaps
            ),
            None,
        )

        if matched_cmap is None:
            matched_cmap = next(
                (cmap for cmap in FALLBACK_CMAPS if cmap not in used_cmaps), None
            )
        if matched_cmap is None:
            matched_cmap = FALLBACK_CMAPS[idx % len(FALLBACK_CMAPS)]

        used_cmaps.add(matched_cmap)

        concepts_in_domain = sorted(
            list(meta[meta[super_col] == super_cat][spec_col].unique())
        )
        super_to_concepts[super_cat] = concepts_in_domain
        cmap_func = cm.get_cmap(matched_cmap)
        n_concepts = len(concepts_in_domain)

        shades = (
            [0.60] if n_concepts == 1 else np.linspace(0.38, 0.85, n_concepts)
        )
        for concept, shade in zip(concepts_in_domain, shades):
            concept_to_color[concept] = cmap_func(shade)

    return concept_to_color, super_to_concepts


def generate_tsne_clean_color_grid(
    harness,
    pruning_levels: list[float] = PRUNING_LEVELS_5PCT,
    samples_per_class: int = 5,
    output_dir: str | None = None,
    n_cols: int = 6,
) -> str:
    """Generates a 19-stage t-SNE grid using global coordinate fitting to track feature drift."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "tsne")
    os.makedirs(output_dir, exist_ok=True)

    meta = harness.metadata.reset_index(drop=True)
    spec_col, _, super_col = _get_taxonomy_mapping(meta)

    concept_to_color, super_to_concepts = build_hierarchical_concept_palette(
        meta, spec_col, super_col
    )
    unique_concepts = sorted(list(meta[spec_col].unique()))

    # Deterministic stratified sampling across concepts
    rng = np.random.default_rng(42)
    selected_indices = []
    for concept in unique_concepts:
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = rng.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)
    n_samples_per_stage = len(selected_indices)

    # Extract features across all pruning stages
    all_img_feats = []
    true_labels = []

    for p_level in pruning_levels:
        if hasattr(harness, "pruning_engine"):
            pruned_model = harness.pruning_engine.get_pruned_model(
                amount=p_level, encoder_type="joint", target_area="full"
            )
        else:
            pruned_model = harness._apply_pruning(harness.base_model, p_level)

        img_feats_full, _, true_labels_full, _, _ = harness._extract_joint_features(
            pruned_model
        )

        if isinstance(img_feats_full, torch.Tensor):
            img_feats = img_feats_full[selected_indices].detach().cpu().numpy()
        else:
            img_feats = img_feats_full[selected_indices]

        all_img_feats.append(img_feats)

        if not true_labels:
            true_labels = [true_labels_full[i] for i in selected_indices]

    # Fit global t-SNE across all stacked stages
    X_total = np.vstack(all_img_feats)
    calc_perp = min(30, max(5, n_samples_per_stage // 4))

    tsne = TSNE(n_components=2, perplexity=calc_perp, random_state=42, init="pca")
    X_total_2d = tsne.fit_transform(X_total)

    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    x_min, x_max = X_total_2d[:, 0].min(), X_total_2d[:, 0].max()
    y_min, y_max = X_total_2d[:, 1].min(), X_total_2d[:, 1].max()
    padding_x = (x_max - x_min) * 0.05
    padding_y = (y_max - y_min) * 0.05

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]

        start_idx = idx * n_samples_per_stage
        end_idx = start_idx + n_samples_per_stage
        coords_2d = X_total_2d[start_idx:end_idx]

        for i, true_spec in enumerate(true_labels):
            color = concept_to_color.get(true_spec, "gray")
            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                c=[color],
                marker="o",
                s=35,
                alpha=0.85,
                linewidths=0.3,
                edgecolors="black",
            )

        ax.set_title(f"Atrophy: {int(p_level * 100)}%", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    legend_handles = []
    for super_cat, concepts in super_to_concepts.items():
        legend_handles.append(
            plt.Line2D([0], [0], color="w", label=f"[{str(super_cat).upper()}]", markersize=0)
        )
        for concept in concepts:
            col = concept_to_color[concept]
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=col,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    markersize=7,
                    label=f"  {concept}",
                )
            )

    fig.legend(
        handles=legend_handles,
        title="Domains & Object Concepts",
        title_fontsize="11",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(6, len(super_to_concepts) + 2),
        frameon=True,
        facecolor="#f9f9f9",
        fontsize=8,
    )

    plt.suptitle(
        f"Global Feature Trajectory Across 5% Atrophy Increments (0% to 90%)\n"
        f"Domain-Grouped Palettes (N={samples_per_class} samples/class) | Shared Global t-SNE Space",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_concept_dispersion_global_align.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved global trajectory t-SNE grid to:\n    {save_path}")
    return save_path


generate_tsne_grid_with_key = generate_tsne_clean_color_grid