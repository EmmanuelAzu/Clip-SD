"""Bozeat-style text -> image retrieval experiment ("a duck with four
legs" analogue), scoped to the curated 10-class subset (both as query
prompts and as the retrieval candidate pool -- src/curated_config.py).

Restricting the candidate pool this way is what actually makes this fast:
previously every pruning level re-encoded the full balanced dataset
(~7,200 images) just to answer 4 text queries. Now it re-encodes ~30-50
images (10 curated classes x a handful of samples each) to answer 10
queries -- a ~150-200x reduction in the dominant cost, while covering
more than twice as many prompts.

Two outputs:
  1. A full-resolution retrieval CSV/plot across every pruning level in
     the schedule (cheap now, so no need to subsample for the numbers).
  2. A readable qualitative image-grid, which -- unlike the numeric
     curve -- genuinely needs a small number of columns to stay legible,
     so it uses a representative subset of pruning levels.
"""

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


def run_bozeat_experiment(
    harness,
    target_prompts: list[str],
    pruning_levels: list[float],
    display_pruning_levels: list[float] | None = None,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Runs Bozeat-style retrieval for `target_prompts` at every level in
    `pruning_levels`, using `harness.valid_metadata` as BOTH the source of
    query prompts and the retrieval candidate pool -- pass an evaluator
    already restricted to the curated subset (JointSpaceEvaluator(
    restrict_classes=CURATED_CLASSES)) for this to be fast and meaningful.

    Returns (csv_path, image_grid_path).
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")
    os.makedirs(output_dir, exist_ok=True)

    if display_pruning_levels is None:
        # A readable subset for the qualitative image grid -- the full
        # numeric curve still uses every level in `pruning_levels`.
        display_pruning_levels = [p for p in pruning_levels if round(p * 1000) % 100 == 0]
        if pruning_levels[-1] not in display_pruning_levels:
            display_pruning_levels.append(pruning_levels[-1])

    spec_col = "specific" if "specific" in harness.valid_metadata.columns else "concept"
    tokens = clip.tokenize([p.lower() for p in target_prompts]).to(harness.device)

    records = []  # one row per (prompt, pruning_level)

    for p_level in pruning_levels:
        pruned_model = harness.pruning_engine.get_pruned_model(
            amount=p_level, encoder_type="joint", target_area="full"
        )
        pruned_model.eval()

        with torch.no_grad():
            text_feats = pruned_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            visual_memory = harness._reindex_visual_memory(pruned_model)

            sim_matrix = text_feats @ visual_memory.to(harness.device).T
            best_scores, best_match_indices = torch.max(sim_matrix, dim=-1)
            best_match_indices = best_match_indices.cpu().numpy()
            best_scores = best_scores.cpu().numpy()

        for idx, prompt_concept in enumerate(target_prompts):
            best_img_idx = best_match_indices[idx]
            retrieved_row = harness.valid_metadata.iloc[best_img_idx]
            retrieved_concept = retrieved_row[spec_col]

            img_path = retrieved_row.get("resolved_filepath", "")
            if not img_path or not os.path.exists(img_path):
                path_col = next((c for c in ["filepath", "filename", "image_path", "path"] if c in retrieved_row.index), None)
                img_path = retrieved_row[path_col] if path_col else ""

            is_correct = str(retrieved_concept).lower() == str(prompt_concept).lower()
            records.append({
                "prompt": prompt_concept,
                "pruning_level": p_level,
                "retrieved_concept": retrieved_concept,
                "retrieved_img_path": img_path,
                "correct": is_correct,
                "top1_similarity": float(best_scores[idx]),
            })

    results_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "bozeat_retrieval_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[+] Bozeat retrieval results saved to: {csv_path}")

    _plot_retrieval_curve(results_df, target_prompts, output_dir)
    grid_path = _plot_image_grid(results_df, target_prompts, display_pruning_levels, output_dir)

    return csv_path, grid_path


def _plot_retrieval_curve(results_df: pd.DataFrame, target_prompts: list[str], output_dir: str) -> str:
    """Full-resolution (every pruning level) per-class retrieval accuracy
    and mean confidence, plus the aggregate across all 10 classes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200)
    cmap = plt.get_cmap("tab10")

    agg_acc = results_df.groupby("pruning_level")["correct"].mean()
    agg_conf = results_df.groupby("pruning_level")["top1_similarity"].mean()

    for i, prompt in enumerate(target_prompts):
        sub = results_df[results_df["prompt"] == prompt].sort_values("pruning_level")
        ax1.plot(sub["pruning_level"] * 100, sub["correct"].astype(float),
                  color=cmap(i), alpha=0.55, linewidth=1.2)
        ax2.plot(sub["pruning_level"] * 100, sub["top1_similarity"],
                  color=cmap(i), alpha=0.55, linewidth=1.2, label=prompt)

    ax1.plot(agg_acc.index * 100, agg_acc.values, color="black", linewidth=3, label="Mean (all 10 classes)")
    ax2.plot(agg_conf.index * 100, agg_conf.values, color="black", linewidth=3, linestyle="--")

    ax1.set_xlabel("Pruning Level (%)"); ax1.set_ylabel("Retrieval Correct (1/0)")
    ax1.set_title("Bozeat Retrieval Accuracy per Class", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Pruning Level (%)"); ax2.set_ylabel("Top-1 Cosine Similarity")
    ax2.set_title("Retrieval Confidence per Class", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=6.5, ncol=2); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "bozeat_retrieval_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Bozeat retrieval curve saved to: {save_path}")
    return save_path


def _plot_image_grid(
    results_df: pd.DataFrame,
    target_prompts: list[str],
    display_pruning_levels: list[float],
    output_dir: str,
) -> str:
    """The qualitative "what did it retrieve" grid, restricted to a
    readable subset of pruning levels.
    """
    n_rows = len(target_prompts)
    n_cols = len(display_pruning_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.8 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r_idx, prompt in enumerate(target_prompts):
        sub = results_df[results_df["prompt"] == prompt].set_index("pruning_level")
        for c_idx, p_level in enumerate(display_pruning_levels):
            ax = axes[r_idx, c_idx]
            row = sub.loc[p_level] if p_level in sub.index else None
            ret_concept = row["retrieved_concept"] if row is not None else "?"
            img_path = row["retrieved_img_path"] if row is not None else ""
            is_correct = bool(row["correct"]) if row is not None else False

            img = load_image_safely(img_path)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(0.5, 0.5, f"{ret_concept}", ha="center", va="center", fontsize=6, color="black")

            ax.set_xticks([]); ax.set_yticks([])

            check_mark = "\u2713" if is_correct else "\u2717"
            title_text = f"'{ret_concept}' {check_mark}"
            title_color = "darkgreen" if is_correct else "darkred"
            box_color = "#e6f4ea" if is_correct else "#fce8e6"
            ax.set_title(
                title_text, fontsize=6.5, color=title_color, fontweight="bold", pad=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=box_color, edgecolor=title_color, lw=0.6),
            )

            if r_idx == 0:
                ax.set_xlabel(f"{p_level * 100:.1f}%", fontsize=9.0, fontweight="bold", labelpad=6)
                ax.xaxis.set_label_position("top")
            if c_idx == 0:
                ax.set_ylabel(f'"{prompt}"', fontsize=9.0, fontweight="bold",
                               rotation=0, labelpad=48, ha="right", va="center")

    plt.suptitle(
        "Bozeat Text\u2192Image Retrieval Trajectory (curated 10-class subset)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "bozeat_retrieval_grid.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Bozeat image grid saved to: {save_path}")
    return save_path
