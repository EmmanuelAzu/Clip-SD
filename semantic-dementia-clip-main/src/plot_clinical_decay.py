import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def plot_clinical_decay(
    csv_path: str,
    encoder_type: str = "joint",
    output_dir: str | None = None,
):
    """Plots hierarchical semantic decay metrics across network pruning levels.

    Strictly supports Joint Projection Space ('joint') or Vision Encoder Hub
    ('vision') simulations.
    """
    encoder_type = encoder_type.lower()
    if encoder_type not in ["joint", "vision"]:
        raise ValueError(
            f"[!] Unsupported encoder type: '{encoder_type}'. "
            "This plot strictly supports 'joint' or 'vision' hub simulations."
        )

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[!] Could not locate metrics CSV at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Resolve pruning level column
    p_col = next(
        (c for c in ["Pruning_Level", "pruning_level", "pruning_pct", "atrophy_level"] if c in df.columns),
        None,
    )
    if p_col is None:
        raise KeyError("[!] Could not find a valid pruning level column in dataset.")

    # Convert fractional values (0.00-0.90) to integer percentages (0-90%)
    if df[p_col].max() <= 1.0:
        x_vals = (df[p_col] * 100).round().astype(int)
    else:
        x_vals = df[p_col].astype(int)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        ("Correct", "Correct Retrieval", "o", "forestgreen"),
        ("Coordinate Error", "Coordinate Error", "s", "darkorange"),
        ("Superordinate Error", "Superordinate Error", "^", "firebrick"),
        ("Domain Error", "Domain Error", "x", "purple"),
        ("Domain Collapse", "Domain Collapse", "d", "darkred"),
    ]

    for col, label, marker, color in metrics:
        if col in df.columns:
            ax.plot(
                x_vals,
                df[col],
                marker=marker,
                label=label,
                linewidth=2.5,
                markersize=6,
                color=color,
            )

    hub_label = "Vision Encoder Hub" if encoder_type == "vision" else "Joint Projection Hub"
    ax.set_title(
        f"Simulated Semantic Dementia: {hub_label}\nHierarchical Conceptual Breakdown Across Atrophy Increments",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Network Pruning Intensity (% Zeroed)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Response Count / Error Rate", fontsize=11, fontweight="bold")
    
    ax.set_xticks(range(0, int(x_vals.max()) + 5, 5))
    ax.legend(title="Clinical Response Type", loc="best", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{encoder_type}_hub_clinical_decay.png"
    out_path = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved {hub_label} clinical decay plot to:\n    {out_path}")
    return out_path


if __name__ == "__main__":
    # Example execution defaulting to Joint Space metrics
    default_csv = os.path.join(
        PROJECT_ROOT, "data", "results", "joint_space_metrics_full.csv"
    )
    if os.path.exists(default_csv):
        plot_clinical_decay(default_csv, encoder_type="joint")
    else:
        print(f"[!] Target metrics file not found: {default_csv}")