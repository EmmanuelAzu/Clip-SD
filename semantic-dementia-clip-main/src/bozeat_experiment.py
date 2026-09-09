import os
import sys
import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

PRUNING_LEVELS_5PCT = [round(x, 2) for x in np.arange(0.00, 0.91, 0.05).tolist()]

LIVING_KEYWORDS = [
    "living", "animal", "fruit", "vegetable", "bird", "mammal",
    "fauna", "flora", "organism", "amphibian", "reptile", "dog",
    "cat", "fish", "insect", "flower", "tree"
]


def load_image_safely(img_path):
    """Attempts multi-path resolution to open image files reliably."""
    if not isinstance(img_path, str) or not img_path.strip():
        return None

    search_paths = [
        img_path,
        os.path.abspath(img_path),
        os.path.join(PROJECT_ROOT, img_path),
        os.path.join(PROJECT_ROOT, "data", img_path),
        os.path.join(PROJECT_ROOT, "data", "processed", img_path),
        os.path.join(PROJECT_ROOT, "data", "processed", "images", img_path),
        os.path.join(PROJECT_ROOT, "data", "images", img_path),
    ]

    for path in search_paths:
        if os.path.exists(path) and not os.path.isdir(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    return None


def select_living_taxonomy_prompts(metadata, num_prompts=4):
    """Filters dataset taxonomy strictly for living entities."""
    spec_col = "specific" if "specific" in metadata.columns else ("concept" if "concept" in metadata.columns else "category")
    meta_unique = metadata.drop_duplicates(subset=[spec_col]).copy()

    def _is_living(row):
        combined_text = " ".join([
            str(row.get("domain", "")),
            str(row.get("superordinate", "")),
            str(row.get("coordinate", "")),
            str(row.get(spec_col, ""))
        ]).lower()
        return any(kw in combined_text for kw in LIVING_KEYWORDS)

    meta_unique["is_living"] = meta_unique.apply(_is_living, axis=1)
    living_meta = meta_unique[meta_unique["is_living"]]

    if len(living_meta) == 0:
        living_meta = meta_unique

    selected_prompts = living_meta[spec_col].head(num_prompts).tolist()
    return selected_prompts


def run_bozeat_visual_grid(
    harness,
    pruning_levels=PRUNING_LEVELS_5PCT,
    target_prompts=None,
    output_dir=None,
    num_prompts=4,
):
    """Executes Bozeat 'Draw from Prompt' task on living items and saves rendered visual grid."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")

    os.makedirs(output_dir, exist_ok=True)

    selected_prompts = target_prompts or select_living_taxonomy_prompts(
        harness.valid_metadata, num_prompts=num_prompts
    )
    print(f"[*] Bozeat Selected Living Prompts ({len(selected_prompts)} items):\n    {selected_prompts}")

    spec_col = "specific" if "specific" in harness.valid_metadata.columns else ("concept" if "concept" in harness.valid_metadata.columns else "category")
    # Bare single-word Bozeat query (Directive 3.1) -- no conversational template.
    tokens = clip.tokenize([p.lower() for p in selected_prompts]).to(harness.device)

    retrieval_grid_data = {p: [] for p in selected_prompts}

    for p_level in pruning_levels:
        # Always derive a fresh pruned copy from the pristine base model
        # (Non-Iterative Masking, Directive 6.1) rather than mutating
        # harness.base_model in place -- matches every other script's
        # convention and uses the real CLIPPruningEngine API.
        pruned_model = harness.pruning_engine.get_pruned_model(
            amount=p_level, encoder_type="joint", target_area="full"
        )
        pruned_model.eval()

        with torch.no_grad():
            text_feats = pruned_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            visual_memory = harness._reindex_visual_memory(pruned_model)

            sim_matrix = text_feats @ visual_memory.to(harness.device).T
            best_match_indices = torch.argmax(sim_matrix, dim=-1).cpu().numpy()

        for idx, prompt_concept in enumerate(selected_prompts):
            best_img_idx = best_match_indices[idx]
            retrieved_row = harness.valid_metadata.iloc[best_img_idx]
            retrieved_concept = retrieved_row[spec_col]

            img_path = retrieved_row.get("resolved_filepath", "")
            if not img_path or not os.path.exists(img_path):
                path_col = next((c for c in ["filepath", "filename", "image_path", "path"] if c in retrieved_row.index), None)
                img_path = retrieved_row[path_col] if path_col else ""

            is_correct = (retrieved_concept.lower() == prompt_concept.lower())
            retrieval_grid_data[prompt_concept].append((p_level, img_path, retrieved_concept, is_correct))

    n_rows = len(selected_prompts)
    n_cols = len(pruning_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.8 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, prompt_concept in enumerate(selected_prompts):
        records = retrieval_grid_data[prompt_concept]

        for c_idx, (p_level, img_path, ret_concept, is_correct) in enumerate(records):
            ax = axes[r_idx, c_idx]
            img = load_image_safely(img_path)

            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(0.5, 0.5, f"{ret_concept}", ha="center", va="center", fontsize=6, color="black")

            ax.set_xticks([])
            ax.set_yticks([])

            if is_correct:
                title_text = f"'{ret_concept}' ✓"
                title_color = "darkgreen"
                box_color = "#e6f4ea"
            else:
                title_text = f"'{ret_concept}' ✗"
                title_color = "darkred"
                box_color = "#fce8e6"

            ax.set_title(
                title_text,
                fontsize=6.5,
                color=title_color,
                fontweight="bold",
                pad=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=box_color, edgecolor=title_color, lw=0.6),
            )

            if r_idx == 0:
                ax.set_xlabel(f"{int(p_level * 100)}%", fontsize=9.0, fontweight="bold", labelpad=6)
                ax.xaxis.set_label_position("top")

            if c_idx == 0:
                ax.set_ylabel(
                    f'Prompt:\n"{prompt_concept}"',
                    fontsize=9.0,
                    fontweight="bold",
                    rotation=0,
                    labelpad=40,
                    ha="right",
                    va="center",
                )

    plt.suptitle("Bozeat 'Draw from Prompt' Task: Visual Retrieval Trajectory Across Atrophy Increments (Living Things)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "bozeat_retrieved_images_living.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved Bozeat living items visual grid to: {save_path}")
    return save_path