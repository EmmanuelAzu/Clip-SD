import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import clip

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _normalize_columns(df):
    """Normalizes column names to standard lowercase for internal key checking."""
    df_copy = df.copy()
    df_copy.columns = [str(col).strip() for col in df_copy.columns]
    return df_copy


def plot_category_breakdown_suite(df, output_dir=None):
    """Plots empirical performance vs expected theoretical decay bound

    and renders a heatmap of clinical error taxonomy counts.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Empirically Measured Curves vs Theory
    if "Correct" in df.columns:
        total = df[["Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse"]].sum(axis=1)
        acc = df["Correct"] / total
        ax1.plot(
            prune_pcts,
            acc,
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            label="Empirical Accuracy",
        )
    elif "i2t_top1" in df.columns:
        ax1.plot(
            prune_pcts,
            df["i2t_top1"],
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            label="Empirical Top-1 Acc",
        )

    if "cka_vision" in df.columns:
        ax1.plot(
            prune_pcts,
            df["cka_vision"],
            marker="^",
            color="#2ca02c",
            linewidth=2,
            label="Vision CKA",
        )

    # Theoretical Expected Baseline Overlay (Semantic Dementia Quadratic Decay)
    exp_decay = [1.0 * (1.0 - (p / 100.0) ** 2) for p in prune_pcts]
    ax1.plot(
        prune_pcts,
        exp_decay,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label="Expected Theory Bound",
    )

    ax1.set_xlabel("Pruning Level (%)", fontsize=11)
    ax1.set_ylabel("Score / Accuracy", fontsize=11)
    ax1.set_title("Empirical Metrics vs Expected Theoretical Bound", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True)

    # Panel 2: Error Taxonomy Heatmap
    err_cols = [
        c for c in [
            "Coordinate Error",
            "Superordinate Error",
            "Domain Error",
            "Domain Collapse",
            "coordinate error",
            "superordinate error",
            "domain error",
            "domain collapse"
        ] if c in df.columns
    ]

    if err_cols:
        heatmap_data = df[err_cols].T
        heatmap_data.columns = [f"{int(p * 100)}%" for p in df[p_col]]
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="g",
            cmap="YlOrRd",
            ax=ax2,
            cbar=True,
        )
        ax2.set_title("Clinical Taxonomy Error Counts", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Pruning Level (%)", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "category_breakdown_suite.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved category breakdown suite to: {save_path}")


def plot_hierarchical_breakdown_suite(df, output_dir=None):
    """Plots fine-to-coarse hierarchical concept decay (Specific vs Superordinate)."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    plt.figure(figsize=(9, 5.5))

    # Empirical Metrics
    for col, color in [
        ("top1_specific_acc", "#d62728"),
        ("top1_coordinate_acc", "#ff7f0e"),
        ("top1_super_acc", "#2ca02c"),
    ]:
        if col in df.columns:
            name = col.replace("top1_", "").replace("_acc", "").capitalize()
            plt.plot(
                prune_pcts,
                df[col],
                marker="o",
                linewidth=2.5,
                color=color,
                label=f"Empirical {name}",
            )

    # Expected Fine-to-Coarse Decay Profiles
    exp_spec = [max(0.0, 1.0 - (p / 100.0) ** 1.5) for p in prune_pcts]
    exp_super = [max(0.0, 1.0 - 0.3 * (p / 100.0) ** 3) for p in prune_pcts]

    plt.plot(
        prune_pcts,
        exp_spec,
        "--",
        color="#d62728",
        alpha=0.5,
        label="Expected Specific Decay",
    )
    plt.plot(
        prune_pcts,
        exp_super,
        "--",
        color="#2ca02c",
        alpha=0.5,
        label="Expected Superordinate Decay",
    )

    plt.xlabel("Pruning Level (%)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Hierarchical Concept Decay: Empirical vs Expected Fine-to-Coarse", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hierarchical_breakdown_suite.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved hierarchical breakdown suite to: {save_path}")


def plot_signal_noise_distribution_shift(
    harness,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90],
    output_dir=None,
):
    """Plots cosine similarity density distribution shift between matching image-text queries across pruning checkpoints."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    test_queries = harness.metadata.drop_duplicates(subset=["specific"]).to_dict("records")
    prompts = [q["specific"].lower() for q in test_queries]
    tokens = clip.tokenize(prompts).to(harness.device)

    for p in pruning_levels:
        atrophied_model = harness.pruning_engine.get_pruned_model(
            amount=p, encoder_type="joint", target_area="full"
        )
        atrophied_model.eval()

        with torch.no_grad():
            text_feats = atrophied_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            visual_memory = harness._reindex_visual_memory(atrophied_model)
            
            # Diagonal/Matching Cosine Similarities
            num_queries = min(len(text_feats), len(visual_memory))
            sims = (text_feats[:num_queries] * visual_memory[:num_queries]).sum(dim=-1).cpu().numpy()

        sns.kdeplot(
            sims, label=f"Pruned {int(p*100)}%", fill=True, alpha=0.2
        )

    plt.title("Target Cosine Similarity Distribution Shift Across Pruning", fontsize=13, fontweight="bold")
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "signal_noise_distribution_shift.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved distribution shift plot to: {save_path}")


def plot_concept_retrieval_heatmap(
    harness,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90],
    output_dir=None,
):
    """Renders a heat map matrix of Top-1 retrieval accuracy broken down per individual concept across pruning levels."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)

    concepts = harness.metadata["specific"].unique().tolist()
    test_queries = harness.metadata.drop_duplicates(subset=["specific"]).to_dict("records")
    prompts = [q["specific"].lower() for q in test_queries]
    tokens = clip.tokenize(prompts).to(harness.device)

    concept_accs = {c: [] for c in concepts}

    for p in pruning_levels:
        atrophied_model = harness.pruning_engine.get_pruned_model(
            amount=p, encoder_type="joint", target_area="full"
        )
        atrophied_model.eval()

        with torch.no_grad():
            text_feats = atrophied_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            visual_memory = harness._reindex_visual_memory(atrophied_model)

            sim_matrix = (100.0 * text_feats @ visual_memory.T).softmax(dim=-1)
            top1_preds = torch.argmax(sim_matrix, dim=-1).cpu().numpy()

        for idx, query in enumerate(test_queries):
            concept_name = query["specific"]
            pred_row = harness.valid_metadata.iloc[top1_preds[idx]]
            acc = 1.0 if pred_row["specific"] == concept_name else 0.0
            concept_accs[concept_name].append(acc)

    df_heat = pd.DataFrame(
        concept_accs, index=[f"{int(p*100)}%" for p in pruning_levels]
    ).T

    plt.figure(figsize=(10, max(6, len(concepts) * 0.35)))
    sns.heatmap(df_heat, annot=True, cmap="viridis", vmin=0.0, vmax=1.0, fmt=".2f")
    plt.title("Per-Concept Accuracy Breakdown Across Pruning Levels", fontsize=13, fontweight="bold")
    plt.xlabel("Pruning Level (%)", fontsize=11)
    plt.ylabel("Concept Specific Category", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "concept_retrieval_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved concept retrieval heatmap to: {save_path}")


if __name__ == "__main__":
    results_csv = os.path.join(PROJECT_ROOT, "data", "results", "joint_hub_sd_simulation.csv")
    
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        plot_category_breakdown_suite(df_results)
        plot_hierarchical_breakdown_suite(df_results)
    else:
        print(f"[!] Simulation results file not found at: {results_csv}. Run TestingHarness simulation first.")