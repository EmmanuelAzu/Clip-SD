import os
import clip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.86, 0.05).tolist()
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
    "Greens",
    "Reds",
    "Blues",
    "Oranges",
    "Purples",
    "YlOrBr",
    "PuBu",
    "YlGn",
    "RdPu",
    "GnBu",
]


def _get_taxonomy_mapping(metadata):
    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in metadata.columns),
        "specific",
    )
    coord_col = next(
        (
            c
            for c in ["coordinate", "basic", "category"]
            if c in metadata.columns
        ),
        "coordinate",
    )
    super_col = next(
        (
            c
            for c in ["superordinate", "domain", "macro"]
            if c in metadata.columns
        ),
        "superordinate",
    )
    return spec_col, coord_col, super_col


def build_hierarchical_concept_palette(meta, spec_col, super_col):
    """Assigns distinct colormap families to superordinate categories and distinct shades to specific concepts within each domain."""
    unique_super = sorted(list(meta[super_col].unique()))
    concept_to_color = {}
    super_to_concepts = {}

    used_cmaps = set()

    for idx, super_cat in enumerate(unique_super):
        super_str = str(super_cat).lower()

        # Match domain keyword to preferred colormap or pick from fallback list
        matched_cmap = None
        for key, cmap in DOMAIN_CMAP_PREFERENCES.items():
            if key in super_str and cmap not in used_cmaps:
                matched_cmap = cmap
                break

        if matched_cmap is None:
            for cmap in FALLBACK_CMAPS:
                if cmap not in used_cmaps:
                    matched_cmap = cmap
                    break

        if matched_cmap is None:
            matched_cmap = FALLBACK_CMAPS[idx % len(FALLBACK_CMAPS)]

        used_cmaps.add(matched_cmap)

        concepts_in_domain = sorted(
            list(meta[meta[super_col] == super_cat][spec_col].unique())
        )
        super_to_concepts[super_cat] = concepts_in_domain

        cmap_func = cm.get_cmap(matched_cmap)
        n_concepts = len(concepts_in_domain)

        # Distribute shades along the colormap (range 0.35 to 0.85 avoids extreme whites/blacks)
        if n_concepts == 1:
            shades = [0.60]
        else:
            shades = np.linspace(0.38, 0.85, n_concepts)

        for concept, shade in zip(concepts_in_domain, shades):
            concept_to_color[concept] = cmap_func(shade)

    return concept_to_color, super_to_concepts


def generate_tsne_clean_color_grid(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5PCT,
    samples_per_class=5,  # Reduced to 5 images per class for clearer point separation
    output_dir="./data/results/tsne",
    n_cols=6,
    use_single_word_prompts=False,
):
    """Generates an 18-stage t-SNE grid with domain-structured color palettes and decluttered point sampling."""
    print(
        f"[*] Generating {len(pruning_levels)}-stage hierarchical color t-SNE grid (N={samples_per_class} samples/class)..."
    )
    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, _, super_col = _get_taxonomy_mapping(meta)

    # Build domain-grouped color mapping
    concept_to_color, super_to_concepts = build_hierarchical_concept_palette(
        meta, spec_col, super_col
    )
    unique_concepts = sorted(list(meta[spec_col].unique()))

    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows)
    )
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    # Stratified sample selection across concepts
    np.random.seed(42)
    selected_indices = []
    for concept in unique_concepts:
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = np.random.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)

        (
            img_feats_full,
            text_feats,
            true_labels_full,
            eval_concepts,
            _,
        ) = evaluator._extract_joint_features(pruned_model)

        if use_single_word_prompts:
            text_tokens = evaluator.tokenizer(eval_concepts).to(
                evaluator.device
            )
            with torch.no_grad():
                text_feats = pruned_model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(
                    dim=-1, keepdim=True
                )

        img_feats = img_feats_full[selected_indices]
        true_labels = [true_labels_full[i] for i in selected_indices]

        tsne = TSNE(
            n_components=2, perplexity=18, random_state=42, init="pca"
        )
        coords_2d = tsne.fit_transform(img_feats.numpy())

        # Render uniform markers colored strictly by concept shade
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

        ax.set_title(
            f"Atrophy: {p_level * 100:.0f}%", fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=7)

    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Grouped Legend by Superordinate Category
    legend_handles = []
    for super_cat, concepts in super_to_concepts.items():
        # Header entry for superordinate group
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="w",
                label=f"[{str(super_cat).upper()}]",
                markersize=0,
            )
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
        f"Joint Space Feature Mixing Across 5% Atrophy Increments (0% to 85%)\n"
        f"Domain-Grouped Palettes (N={samples_per_class} samples/class) | "
        f"Prompting: {'Single-Word' if use_single_word_prompts else 'Contextual Template'}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, "tsne_concept_dispersion_hierarchical_colors.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"[+] Saved hierarchical concept-dispersion t-SNE grid to:\n    {save_path}"
    )
    return save_path


generate_tsne_grid_with_key = generate_tsne_clean_color_grid