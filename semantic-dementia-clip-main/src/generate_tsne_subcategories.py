import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 19-stage pruning schedule (5% increments from 0% to 90%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()
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


def generate_tsne_grid_with_key(
    harness,
    pruning_levels: list[float] = PRUNING_LEVELS_5PCT,
    samples_per_class: int = 15,
    output_dir: str | None = None,
    n_cols: int = 6,
    use_single_word_prompts: bool = False,
) -> str:
    """Generates a 19-stage t-SNE grid plot with concept-level color coding and error-type markers."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "tsne")
    os.makedirs(output_dir, exist_ok=True)

    meta = harness.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_taxonomy_mapping(meta)
    concept_lookup = meta.drop_duplicates(spec_col).set_index(spec_col)

    # Color palette mapped directly to Specific Object Concepts
    unique_concepts = sorted(list(meta[spec_col].unique()))
    palette = sns.color_palette("husl", n_colors=len(unique_concepts))
    concept_to_color = {
        concept: palette[i] for i, concept in enumerate(unique_concepts)
    }

    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    marker_map = {
        "Correct": ("o", 40, 0.75, 0.0, "none"),
        "Coordinate Error": ("s", 50, 0.85, 1.0, "black"),
        "Superordinate Error": ("^", 60, 0.90, 1.0, "black"),
        "Domain Collapse": ("X", 70, 0.95, 1.2, "darkred"),
    }

    # Stratified sampling across dataset indices
    rng = np.random.default_rng(42)
    selected_indices = []
    for concept in unique_concepts:
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = rng.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]

        if hasattr(harness, "pruning_engine"):
            pruned_model = harness.pruning_engine.get_pruned_model(
                amount=p_level, encoder_type="joint", target_area="full"
            )
        else:
            pruned_model = harness._apply_pruning(harness.base_model, p_level)

        (
            img_feats_full,
            text_feats,
            true_labels_full,
            eval_concepts,
            _,
        ) = harness._extract_joint_features(pruned_model)

        # Optional single-word prompt override for baseline evaluation
        if use_single_word_prompts and hasattr(harness, "tokenizer"):
            device = getattr(harness, "device", "cuda" if torch.cuda.is_available() else "cpu")
            text_tokens = harness.tokenizer(eval_concepts).to(device)
            with torch.no_grad():
                text_feats = pruned_model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        img_feats = img_feats_full[selected_indices]
        true_labels = [true_labels_full[i] for i in selected_indices]

        sim_matrix = torch.matmul(img_feats, text_feats.T)
        top1_indices = torch.argmax(sim_matrix, dim=1).detach().cpu().numpy()

        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init="pca")
        raw_img_feats = img_feats.detach().cpu().numpy() if torch.is_tensor(img_feats) else img_feats
        coords_2d = tsne.fit_transform(raw_img_feats)

        for i, true_spec in enumerate(true_labels):
            pred_spec = eval_concepts[top1_indices[i]]

            if pred_spec == true_spec:
                err_type = "Correct"
            else:
                true_co = (
                    concept_lookup.loc[true_spec, coord_col]
                    if true_spec in concept_lookup.index
                    else None
                )
                pred_co = (
                    concept_lookup.loc[pred_spec, coord_col]
                    if pred_spec in concept_lookup.index
                    else None
                )
                true_su = (
                    concept_lookup.loc[true_spec, super_col]
                    if true_spec in concept_lookup.index
                    else None
                )
                pred_su = (
                    concept_lookup.loc[pred_spec, super_col]
                    if pred_spec in concept_lookup.index
                    else None
                )

                if true_co == pred_co:
                    err_type = "Coordinate Error"
                elif true_su == pred_su:
                    err_type = "Superordinate Error"
                else:
                    err_type = "Domain Collapse"

            color = concept_to_color.get(true_spec, "gray")
            marker, size, alpha, lw, edge_color = marker_map[err_type]

            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                c=[color],
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=lw,
                edgecolors=edge_color if err_type != "Correct" else "none",
            )

        ax.set_title(f"Atrophy: {int(p_level * 100)}%", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Construct Legends
    concept_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=7,
            label=concept,
        )
        for concept, col in concept_to_color.items()
    ]

    error_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Correct",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="Coordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="Superordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=8,
            label="Domain Collapse",
        ),
    ]

    leg1 = fig.legend(
        handles=concept_handles,
        title="Object Concepts",
        title_fontsize="10",
        loc="upper center",
        bbox_to_anchor=(0.40, -0.01),
        ncol=min(6, len(unique_concepts)),
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.legend(
        handles=error_handles,
        title="Error Categories",
        title_fontsize="10",
        loc="upper center",
        bbox_to_anchor=(0.82, -0.01),
        ncol=2,
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.add_artist(leg1)

    plt.suptitle(
        f"Joint Space Feature Trajectory Across 5% Atrophy Increments (0% to 90%)\n"
        f"Prompting: {'Single-Word' if use_single_word_prompts else 'Contextual Template'}",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_trajectory_specific_colors.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved concept-colored t-SNE grid to: {save_path}")
    return save_path


generate_tsne_5levels_with_key = generate_tsne_grid_with_key