import os
import sys
from typing import Callable, Optional
import clip
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class ScaledImageDataset(Dataset):
    """Dataset class for loading and preprocessing images based on metadata."""

    def __init__(self, csv_path: str, img_dir: str, preprocess: Callable, restrict_domain: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.preprocess = preprocess

        # Domain scope is now configurable rather than hard-restricted to
        # "Living" (the old "Directive 1.1" behavior). Both domains are
        # kept by default so Domain-tier errors ("Domain Error"/"Domain
        # Collapse") are actually reachable in evaluation, matching the
        # proposal's 4-tier taxonomy. Pass restrict_domain="Living" (or
        # "Non-Living") to reproduce the old behavior deliberately.
        if restrict_domain is not None and "domain" in self.df.columns:
            before = len(self.df)
            self.df = self.df[
                self.df["domain"].astype(str).str.strip().str.casefold().eq(restrict_domain.strip().casefold())
            ].reset_index(drop=True)
            dropped = before - len(self.df)
            if dropped > 0:
                print(
                    f"[*] Domain scope filter (restrict_domain='{restrict_domain}'): "
                    f"dropped {dropped} row(s) from {os.path.basename(csv_path)}."
                )

        # Verify files exist to prevent runtime crashes
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, str(row["filename"]))
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))
        image = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(image)
        return tensor, idx


def run_indexing(
    batch_size: int = 64,
    num_workers: int = 4,
    raw_csv: Optional[str] = None,
    img_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    restrict_domain: Optional[str] = None,
) -> Optional[str]:
    """Generates L2-normalized image embeddings using CLIP and saves an offline visual index.

    NOTE on image_index.pt's actual role: this tensor is a PRISTINE
    (unpruned) reference index, useful for standalone inspection/sanity
    checks of the raw dataset (e.g. cluster visualization before any
    atrophy simulation begins). It is deliberately NOT reused by
    JointSpaceEvaluator/TestingHarness's main evaluation loops -- those
    re-encode every image fresh through whatever model state (pristine or
    pruned) is being evaluated at each pruning level, since the entire
    point of the experiment is to see how the VISION encoder's own output
    changes under atrophy; a cached pristine-only index would be actively
    wrong to reuse once vision pruning starts. If you don't need a
    standalone pristine snapshot, this stage is safe to skip via
    --skip_indexing in run_full_pipeline.py.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Scaled Indexer on Device: {device.upper()}")

    if raw_csv is None:
        raw_csv = os.path.join(PROJECT_ROOT, "tests", "metadata_raw.csv")
        if not os.path.exists(raw_csv):
            raw_csv = os.path.join(PROJECT_ROOT, "data", "raw", "metadata_raw.csv")

    if img_dir is None:
        img_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        print(
            f"[!] Error: Raw metadata not found at {raw_csv}. Run data download/builder first!"
        )
        return None

    # Load pre-trained CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Create dataset & loader
    dataset = ScaledImageDataset(raw_csv, img_dir, preprocess, restrict_domain=restrict_domain)

    # On Windows, num_workers > 0 can cause multiprocessing issues
    workers = num_workers if os.name != "nt" else 0

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    all_features = []
    print(
        f"[*] Visual Memory Bank: Indexing {len(dataset)} images in batches of {batch_size}..."
    )

    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Generating Visual Embeddings"):
            batch_imgs = batch_imgs.to(device)
            # Generate L2-normalized image features
            image_features = model.encode_image(batch_imgs)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            all_features.append(image_features.cpu())

    # Concatenate all batches into a single master tensor
    index_tensor = torch.cat(all_features, dim=0)

    # Save outputs
    tensor_path = os.path.join(output_dir, "image_index.pt")
    csv_path = os.path.join(output_dir, "metadata_processed.csv")

    torch.save(index_tensor, tensor_path)
    dataset.df.to_csv(csv_path, index=False)

    print("\n[+] SUCCESS: Offline visual memory bank built!")
    print(f"[+] Saved index tensor shape: {index_tensor.shape}")
    print(f"[+] Processed metadata saved to: {csv_path}")

    return tensor_path


if __name__ == "__main__":
    run_indexing()