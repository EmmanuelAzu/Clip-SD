import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def visualize_random_samples(metadata_path=None, img_dir=None, num_samples=9):
    """Loads processed metadata and plots a grid of random images

    along with their 4-tier taxonomy mappings.
    """
    if metadata_path is None:
        metadata_path = os.path.join(
            PROJECT_ROOT, "data", "metadata_processed.csv"
        )
    if img_dir is None:
        img_dir = os.path.join(PROJECT_ROOT, "data", "images")

    print(f"[*] Reading dataset from: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(
            "[!] Error: Could not find processed metadata. Please run build_general_dataset.py first!"
        )
        return

    df = pd.read_csv(metadata_path)

    # Grab a random sample of images from the dataset
    samples = df.sample(n=min(num_samples, len(df))).reset_index(drop=True)

    # Calculate grid dimensions (e.g., 3x3 for 9 samples)
    grid_size = int(num_samples**0.5)
    if grid_size * grid_size < num_samples:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i, row in samples.iterrows():
        img_path = os.path.join(img_dir, row["filename"])

        if (
            not os.path.exists(img_path)
            and "filepath" in row
            and pd.notna(row["filepath"])
        ):
            img_path = str(row["filepath"])

        ax = axes[i]

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)

            # Format title to show the 4-tier hierarchical taxonomy path
            title_text = (
                f"Domain: {str(row['domain']).upper()}\n"
                f"Super: {str(row['superordinate']).capitalize()}\n"
                f"Coordinate: {str(row['coordinate']).capitalize()}\n"
                f"Specific: {str(row['specific']).capitalize()}"
            )
            ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8)
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing Image:\n{row['filename']}",
                ha="center",
                va="center",
                color="red",
                fontsize=10,
            )

        ax.axis("off")

    # Hide any unused subplot slots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Sample of Images & 4-Tier Taxonomy Mappings",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    # Save visualization output
    output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_samples.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved visualization sheet to: {output_path}")

    plt.show()


if __name__ == "__main__":
    visualize_random_samples(num_samples=9)