import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def plot_simulation_results(csv_path=None, output_dir=None):
    """Plots multi-panel visualization of Semantic Dementia decay trajectory

    and taxonomic cost metrics from TestingHarness output CSVs.
    """
    if csv_path is None:
        csv_path = os.path.join(
            PROJECT_ROOT, "data", "results", "joint_hub_sd_simulation.csv"
        )
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    if not os.path.exists(csv_path):
        print(f"[!] Target CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    p_col = (
        "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    )

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Taxonomic Error Distribution across Atrophy Levels
    ax1 = axes[0]
    error_cols = [
        "Correct",
        "Coordinate Error",
        "Superordinate Error",
        "Domain Error",
        "Domain Collapse",
    ]

    if all(col in df.columns for col in error_cols):
        # Filter for primary joint mode if multi-location comparison file is passed
        if "Modality_Mode" in df.columns:
            df_p1 = df[df["Modality_Mode"] == "joint"]
        else:
            df_p1 = df

        x_p1 = df_p1[p_col] * 100
        total_queries = df_p1[error_cols].sum(axis=1)

        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

        for col, color in zip(error_cols, colors):
            prop = (df_p1[col] / total_queries) * 100
            ax1.plot(
                x_p1, prop, marker="o", label=col, linewidth=2.5, color=color
            )

        ax1.set_title(
            "Taxonomic Retrieval Error Breakdown",
            fontsize=13,
            fontweight="bold",
        )
        ax1.set_xlabel("Pruning Intensity (%)", fontsize=11)
        ax1.set_ylabel("Query Proportion (%)", fontsize=11)
        ax1.set_ylim(-5, 105)
        ax1.legend(loc="best")
        ax1.grid(True, linestyle="--", alpha=0.7)

    # Panel 2: Expected Taxonomic Cost (0.0 to 4.0 Scale)
    ax2 = axes[1]
    if "Expected_Taxonomic_Cost" in df.columns:
        if "Modality_Mode" in df.columns:
            # Multi-location pruning comparison plot
            for mode_name, group_df in df.groupby("Modality_Mode"):
                ax2.plot(
                    group_df[p_col] * 100,
                    group_df["Expected_Taxonomic_Cost"],
                    marker="s",
                    label=f"Mode: {mode_name.replace('_', ' ').title()}",
                    linewidth=2.5,
                )
        else:
            ax2.plot(
                df[p_col] * 100,
                df["Expected_Taxonomic_Cost"],
                marker="s",
                label="Expected Taxonomic Cost",
                linewidth=2.5,
                color="#d62728",
            )

        ax2.set_title(
            "Expected Taxonomic Distance Cost", fontsize=13, fontweight="bold"
        )
        ax2.set_xlabel("Pruning Intensity (%)", fontsize=11)
        ax2.set_ylabel(
            "Taxonomic Cost (0.0 = Perfect, 4.0 = Collapse)", fontsize=11
        )
        ax2.set_ylim(-0.1, 4.2)
        ax2.legend(loc="best")
        ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(
        "Semantic Dementia Multimodal Atrophy Simulation Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(
        ".csv", "_simulation_results.png"
    )
    out_path = os.path.join(output_dir, filename)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Simulation results plot saved to: {out_path}")


if __name__ == "__main__":
    # Primary joint hub simulation output plot
    joint_csv = os.path.join(
        PROJECT_ROOT, "data", "results", "joint_hub_sd_simulation.csv"
    )
    if os.path.exists(joint_csv):
        plot_simulation_results(joint_csv)

    # Comparative pruning location output plot
    comp_csv = os.path.join(
        PROJECT_ROOT, "data", "results", "location_comparison_simulation.csv"
    )
    if os.path.exists(comp_csv):
        plot_simulation_results(comp_csv)