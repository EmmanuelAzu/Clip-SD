"""Typicality effect analysis.

Every core-evaluation run has always computed acc_typical, acc_atypical,
and delta_typicality (src/metrics.py compute_typicality_delta) into
joint_space_metrics.csv -- this is the project's direct, named test for
RQ3 and the prototyping/typicality effect Bozeat et al. (2003) documented
in real SD patients (typical category members are named correctly long
after atypical ones fail). That data has been computed and saved in every
run this session, but never actually plotted or examined. This script
does that, from an existing results CSV -- no new pipeline run required.

The paper's stated validation criterion (Sec 3.4.3): delta_typicality
should be positive and grow (roughly monotonically) as pruning increases
-- i.e. the gap between "still gets typical examples right" and "already
failing atypical examples" should widen as damage accumulates, mirroring
the human prototyping effect.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def plot_typicality_effect(csv_path: str, output_dir: str | None = None) -> str:
    df = pd.read_csv(csv_path)

    required = ["pruning_level", "acc_typical", "acc_atypical", "delta_typicality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[!] {csv_path} is missing column(s) {missing} -- typicality "
            f"metrics are only written when the run's metadata contains both "
            f"Typical and Atypical exemplars (see compute_typicality_delta)."
        )

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = df.sort_values("pruning_level")
    pct = df["pruning_level"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    ax = axes[0]
    ax.plot(pct, df["acc_typical"], marker="o", markersize=4, linewidth=2.2,
            color="#1f77b4", label="Typical exemplars")
    ax.plot(pct, df["acc_atypical"], marker="s", markersize=4, linewidth=2.2,
            color="#d62728", label="Atypical exemplars")
    ax.set_xlabel("Pruning Level (%)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax.set_title("Accuracy: Typical vs. Atypical Exemplars", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    ax2 = axes[1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["delta_typicality"]]
    ax2.bar(pct, df["delta_typicality"], width=(pct.diff().median() or 2.5) * 0.8, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Pruning Level (%)", fontsize=12)
    ax2.set_ylabel("Delta Typicality (Acc_typical - Acc_atypical)", fontsize=12)
    ax2.set_title("The Prototyping Effect Signal", fontsize=12.5, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "Typicality Effect: Do Typical Exemplars Survive Pruning Longer Than Atypical Ones?\n"
        "Validation criterion (Sec 3.4.3): delta should be positive and widen as pruning increases",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "typicality_effect.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Typicality effect plot saved to: {save_path}")

    # Quick, honest summary printed to console -- not just the plot
    n_positive = (df["delta_typicality"] > 0).sum()
    n_total = df["delta_typicality"].notna().sum()
    first_half = df[df["pruning_level"] <= df["pruning_level"].median()]["delta_typicality"].mean()
    second_half = df[df["pruning_level"] > df["pruning_level"].median()]["delta_typicality"].mean()
    print(f"[*] delta_typicality positive at {n_positive}/{n_total} pruning levels")
    print(f"[*] mean delta in lower half of pruning range: {first_half:.4f}")
    print(f"[*] mean delta in upper half of pruning range: {second_half:.4f}")
    if second_half > first_half:
        print("[*] Delta grows with pruning -- consistent with the paper's stated validation criterion.")
    else:
        print("[!] Delta does NOT grow with pruning as expected -- worth investigating before citing this as support.")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the typicality effect from an existing joint_space_metrics.csv.")
    parser.add_argument("csv_path", type=str, help="Path to joint_space_metrics.csv from a completed run.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    plot_typicality_effect(args.csv_path, args.output_dir)
