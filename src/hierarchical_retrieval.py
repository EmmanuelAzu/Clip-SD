"""Hierarchical / cluster-routed retrieval experiment.

Tests the retrieval mechanism your proposal's Background section (2.3)
actually describes -- biological memory retrieval as APPROXIMATE, indexed
search that "narrows the search space to relevant clusters... ignoring
the vast majority of irrelevant data" (citing HNSW-style ANN libraries) --
which the implemented Bozeat experiment (Methodology 3.3.4) deliberately
does NOT use, opting for brute-force flat search instead "rather than
using approximate nearest neighbor heuristics that introduce stochastic
noise." This module closes that gap and tests whether the theoretical
framing holds up empirically.

Method, at each pruning level:
  1. ROUTING: each of the 33 class-name text queries is compared against
     9 group CENTROIDS (mean image embedding per coordinate group:
     Domestic Dogs, Felines, Insects, ...) via cosine similarity. The
     highest-scoring group is the "routed" cluster -- this is the classic
     IVF/cluster-pruning approximate-kNN technique, not a literal forest
     of trees, but the same "narrow first, search within" idea.
  2. WITHIN-GROUP SEARCH: exact nearest-neighbor search, but restricted to
     only the images belonging to the ROUTED group (not necessarily the
     query's true group).
  3. FLAT BASELINE: for direct comparison, the same query's ordinary
     brute-force top-1 result across ALL 165 images (identical to the
     existing Bozeat mechanism), computed from the exact same embeddings
     so the comparison is apples-to-apples.

Because coordinate groups are a strict partition (each specific class
belongs to exactly one group), if routing picks the WRONG group, the
correct answer cannot possibly be found within it -- hierarchical final
accuracy is mathematically upper-bounded by routing accuracy. This module
also reports "within-group accuracy GIVEN correct routing" specifically
to isolate fine-grained discrimination from the routing step, since that
conditional decomposition is what directly tests RQ1 (does the FIRST
failure happen at the broad-category step or the fine-grained step?).

Two outputs, same pattern as src/bozeat_experiment.py:
  1. hierarchical_retrieval_decomposition.png -- the four-curve
     quantitative decomposition above, full resolution (every prompt,
     every pruning level).
  2. hierarchical_retrieval_grid.png -- a qualitative image grid (a
     representative subset of prompts/levels, since each cell is an
     actual photo) showing, for each query: which group it got routed to,
     and which specific image it retrieved within that group -- with a
     3-state outcome (correct / wrong breed / wrong group) so a routing
     failure is visually distinguishable from a within-group failure.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_TAXONOMY
from src.bozeat_experiment import load_image_safely
from src.joint_anchor import build_joint_anchor_queries


def run_hierarchical_retrieval_experiment(
    evaluator,
    pruning_levels: list[float],
    scenario: str = "joint",
    display_prompts: list[str] | None = None,
    display_pruning_levels: list[float] | None = None,
    group_of_concept: dict[str, str] | None = None,
    groups: list[str] | None = None,
    query_mode: str = "text",
    joint_alpha: float = 0.5,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Returns (csv_path, image_grid_path). display_prompts/
    display_pruning_levels control the qualitative image grid only (see
    Bozeat's identical pattern in src/bozeat_experiment.py) -- the
    decomposition CSV/plot always covers every prompt and every pruning
    level regardless of these.

    group_of_concept/groups: normally derived from the hardcoded 33-class
    CURATED_TAXONOMY (the default, used everywhere else in the pipeline).
    A caller validating against a LARGER candidate pool (e.g. all 72
    downloaded classes, not just the curated 33) can pass these in
    instead, typically derived directly from the evaluator's own metadata
    "coordinate" column -- see src/large_pool_validation.py.

    query_mode/joint_alpha: see src/bozeat_experiment.py's identical
    parameters and src/joint_anchor.py -- "text" (default) routes AND
    retrieves using pure text-prompt embeddings; "joint" uses a blended
    text+mean-image query instead, for both the routing step and the
    within-group retrieval step.
    """
    if query_mode not in ("text", "joint"):
        raise ValueError("query_mode must be 'text' or 'joint'.")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "hierarchical_retrieval")
    os.makedirs(output_dir, exist_ok=True)

    if display_prompts is None:
        # Same swap as run_bozeat_grid.py's DEFAULT_DISPLAY_PROMPTS: Banana
        # and Frying Pan replaced with a second Dog and second Cat breed,
        # so the grid shows a within-group comparison directly.
        display_prompts = [
            "German Shepherd", "Golden Retriever", "Tabby Cat", "Persian Cat",
            "Monarch Butterfly", "Albatross", "Chimpanzee", "Bullfrog", "Lemon",
        ]
    if display_pruning_levels is None:
        display_pruning_levels = [p for p in pruning_levels if p <= 0.50]

    group_of_concept = group_of_concept or {c: g for g, members in CURATED_TAXONOMY.items() for c in members}
    groups = groups or sorted(CURATED_TAXONOMY.keys())

    records = []

    for p_level in pruning_levels:
        print(f"[*] Hierarchical retrieval: extracting features at {p_level * 100:.1f}% pruning...")
        if scenario == "joint":
            amount = p_level
        elif scenario == "vision_only":
            amount = {"text": 0.0, "vision": p_level}
        else:
            amount = {"text": p_level, "vision": 0.0}

        pruned_model = evaluator.pruning_engine.get_pruned_model(
            amount=amount, encoder_type="joint", target_area="full"
        )
        img_feats, text_feats, labels, unique_concepts, eval_meta = evaluator._extract_joint_features(pruned_model)

        query_feats = text_feats
        if query_mode == "joint":
            query_feats = build_joint_anchor_queries(
                text_feats, unique_concepts, img_feats, labels, alpha=joint_alpha
            )

        img_groups = [group_of_concept[l] for l in labels]

        # Group centroids from IMAGE embeddings, re-normalized after averaging
        centroids = []
        for g in groups:
            mask = [gr == g for gr in img_groups]
            centroids.append(img_feats[mask].mean(dim=0))
        centroids = torch.stack(centroids)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True)

        sim_to_centroids = query_feats @ centroids.T          # (n_queries, 9)
        routed_group_idx = sim_to_centroids.argmax(dim=1)

        flat_sim = query_feats @ img_feats.T                  # (n_queries, n_images)
        flat_best_idx = flat_sim.argmax(dim=1)

        for i, query_concept in enumerate(unique_concepts):
            true_group = group_of_concept[query_concept]
            routed_group = groups[routed_group_idx[i].item()]
            routing_correct = routed_group == true_group

            # Within-group search: subset the ALREADY-COMPUTED flat_sim row
            # to only the routed group's columns -- same underlying scores
            # as the flat baseline, just a restricted candidate pool.
            mask = np.array([gr == routed_group for gr in img_groups])
            candidate_indices = np.where(mask)[0]
            within_group_sims = flat_sim[i, candidate_indices]
            best_within_local = within_group_sims.argmax().item()
            best_within_idx = candidate_indices[best_within_local]
            retrieved_hier = labels[best_within_idx]
            hier_correct = str(retrieved_hier).lower() == str(query_concept).lower()

            retrieved_flat_idx = flat_best_idx[i].item()
            retrieved_flat = labels[retrieved_flat_idx]
            flat_correct = str(retrieved_flat).lower() == str(query_concept).lower()

            # Resolve actual image file paths for the qualitative grid,
            # same resolution logic as Bozeat's image grid.
            hier_row = eval_meta.iloc[best_within_idx]
            hier_img_path = hier_row.get("resolved_filepath", "")
            if not hier_img_path or not os.path.exists(str(hier_img_path)):
                path_col = next((c for c in ["filepath", "filename", "image_path", "path"] if c in hier_row.index), None)
                hier_img_path = hier_row[path_col] if path_col else ""

            records.append({
                "prompt": query_concept,
                "pruning_level": p_level,
                "true_group": true_group,
                "routed_group": routed_group,
                "routing_correct": routing_correct,
                "retrieved_hierarchical": retrieved_hier,
                "retrieved_hierarchical_img_path": hier_img_path,
                "hierarchical_correct": hier_correct,
                "retrieved_flat": retrieved_flat,
                "flat_correct": flat_correct,
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "hierarchical_retrieval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Hierarchical retrieval results saved to: {csv_path}")

    _plot_decomposition(df, output_dir)
    grid_path = _plot_image_grid(df, display_prompts, display_pruning_levels, output_dir)

    return csv_path, grid_path


def _plot_image_grid(
    df: pd.DataFrame,
    display_prompts: list[str],
    display_pruning_levels: list[float],
    output_dir: str,
) -> str:
    """The qualitative 'what did routing choose, and what did it retrieve'
    grid -- same idea as Bozeat's image grid, but with a 3-state outcome
    instead of a plain correct/incorrect, since a hierarchical failure can
    happen at two genuinely different points:
      - GREEN: routed to the correct group AND retrieved the correct breed.
      - ORANGE: routed to the correct group, but retrieved the WRONG breed
        within it (a pure fine-grained/within-group failure).
      - RED: routed to the WRONG group entirely (a broad-category failure
        -- the retrieved image cannot possibly be correct, see the module
        docstring's mathematical bound).
    """
    n_rows = len(display_prompts)
    n_cols = len(display_pruning_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 3.0 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r_idx, prompt in enumerate(display_prompts):
        sub = df[df["prompt"] == prompt].set_index("pruning_level")
        for c_idx, p_level in enumerate(display_pruning_levels):
            ax = axes[r_idx, c_idx]
            row = sub.loc[p_level] if p_level in sub.index else None

            if row is None:
                ax.axis("off")
                continue

            routed_group = row["routed_group"]
            routing_ok = bool(row["routing_correct"])
            retrieved = row["retrieved_hierarchical"]
            hier_ok = bool(row["hierarchical_correct"])
            img_path = row["retrieved_hierarchical_img_path"]

            if hier_ok:
                state, color = "correct", "darkgreen"
                box_color = "#e6f4ea"
            elif routing_ok:
                state, color = "wrong breed", "#b8860b"
                box_color = "#fff4d6"
            else:
                state, color = "wrong group", "darkred"
                box_color = "#fce8e6"

            img = load_image_safely(img_path)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(0.5, 0.5, f"{retrieved}", ha="center", va="center", fontsize=6, color="black")

            ax.set_xticks([]); ax.set_yticks([])

            symbol = "\u2713" if hier_ok else "\u2717"
            title_text = f"'{retrieved}' {symbol}\n\u2192{routed_group}"
            ax.set_title(
                title_text, fontsize=6.3, color=color, fontweight="bold", pad=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=box_color, edgecolor=color, lw=0.6),
            )

            if r_idx == 0:
                ax.set_xlabel(f"{p_level * 100:.1f}%", fontsize=9.0, fontweight="bold", labelpad=6)
                ax.xaxis.set_label_position("top")
            if c_idx == 0:
                ax.set_ylabel(f'"{prompt}"', fontsize=9.0, fontweight="bold",
                               rotation=0, labelpad=48, ha="right", va="center")

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor="#e6f4ea", edgecolor="darkgreen", label="Correct (routing + breed)"),
        mpatches.Patch(facecolor="#fff4d6", edgecolor="#b8860b", label="Wrong breed (routing correct)"),
        mpatches.Patch(facecolor="#fce8e6", edgecolor="darkred", label="Wrong group (routing failed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        "Hierarchical Retrieval Trajectory: what routing chose, and what it retrieved",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "hierarchical_retrieval_grid.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Hierarchical retrieval image grid saved to: {save_path}")
    return save_path


def _plot_decomposition(df: pd.DataFrame, output_dir: str) -> None:
    agg = df.groupby("pruning_level").agg(
        routing_acc=("routing_correct", "mean"),
        hierarchical_acc=("hierarchical_correct", "mean"),
        flat_acc=("flat_correct", "mean"),
        n_routing_correct=("routing_correct", "sum"),
    ).reset_index()

    # Within-group accuracy GIVEN correct routing -- isolates fine-grained
    # discrimination from the routing step. Undefined (NaN) at any level
    # where routing was never correct for any query. IMPORTANT: this is a
    # conditional mean over only the routing_correct==True subset, so once
    # n_routing_correct gets small (routing rarely succeeds), the metric
    # can only land on a handful of discrete fractions (e.g. exactly 0.5
    # for n=2) -- that's a small-sample artifact, not a real trend. Points
    # built on n < LOW_N_THRESHOLD are marked with hollow markers below so
    # this isn't silently misread as a meaningful plateau.
    conditional = (
        df[df["routing_correct"]]
        .groupby("pruning_level")["hierarchical_correct"]
        .mean()
        .reindex(agg["pruning_level"])
        .values
    )
    agg["within_group_acc_given_correct_routing"] = conditional

    LOW_N_THRESHOLD = 5
    low_n_mask = agg["n_routing_correct"] < LOW_N_THRESHOLD
    if low_n_mask.any():
        low_n_levels = [f"{p*100:.1f}%" for p in agg.loc[low_n_mask, "pruning_level"]]
        print(f"[!] Within-group-given-correct-routing rests on n<{LOW_N_THRESHOLD} at: {low_n_levels} "
              f"-- marked with hollow markers on the plot, interpret those points with caution.")

    summary_path = os.path.join(output_dir, "hierarchical_retrieval_decomposition_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"[+] Decomposition summary (with n_routing_correct) saved to: {summary_path}")

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    pct = agg["pruning_level"] * 100

    ax.plot(pct, agg["routing_acc"], marker="o", markersize=4, linewidth=2.2,
             color="#1f77b4", label="Routing accuracy (correct coordinate group)")

    # Within-group|routing-correct line: hollow markers where n < threshold
    ax.plot(pct, agg["within_group_acc_given_correct_routing"], linewidth=2.2, color="#d62728",
             label=f"Within-group accuracy | routing correct (fine-grained, isolated; hollow = n<{LOW_N_THRESHOLD})")
    high_n = agg[~low_n_mask]
    low_n = agg[low_n_mask]
    ax.scatter(high_n["pruning_level"] * 100, high_n["within_group_acc_given_correct_routing"],
               marker="s", s=22, color="#d62728", zorder=5)
    ax.scatter(low_n["pruning_level"] * 100, low_n["within_group_acc_given_correct_routing"],
               marker="s", s=36, facecolors="none", edgecolors="#d62728", linewidths=1.6, zorder=5)

    ax.plot(pct, agg["hierarchical_acc"], marker="^", markersize=4, linewidth=1.8, linestyle="--",
             color="#9467bd", label="Overall hierarchical accuracy (routing x within-group)")
    ax.plot(pct, agg["flat_acc"], marker="d", markersize=4, linewidth=1.8, linestyle=":",
             color="#2ca02c", label="Flat brute-force accuracy (existing Bozeat baseline)")

    ax.set_xlabel("Pruning Level (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Hierarchical (Cluster-Routed) vs. Flat Retrieval\n"
        "Decomposing failure into routing (broad-category) vs. within-group (fine-grained) errors",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8.8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "hierarchical_retrieval_decomposition.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Decomposition plot saved to: {save_path}")
