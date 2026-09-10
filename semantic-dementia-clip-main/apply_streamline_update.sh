#!/usr/bin/env bash
# Applies the streamlined-pipeline update to the semantic-dementia-clip-main repo.
# Run this from the repo root on the cluster: bash apply_streamline_update.sh
set -euo pipefail

echo "[*] Deleting redundant files..."
rm -fv "run_tsne_5stages.py"
rm -fv "run_tSNE_nerr.py"
rm -fv "run_tsne_no_errors.py"
rm -fv "run_depth_zone_grid.py"
rm -fv "run_masking_mode_comparison.py"
rm -fv "run_pruning_strategy_comparison.py"
rm -fv "src/generate_tsne_subcategories.py"
rm -fv "src/gen_tSNE_subcat_nerr.py"
rm -fv "src/generate_tsne_subcat_no_error.py"
rm -fv "src/generate_tsne.py"
rm -fv "src/extended_hierarchical_eval.py"
rm -fv "src/baseline_conditions.py"

echo "[*] Writing new/updated files..."
mkdir -p src

cat > "src/curated_config.py" << 'APPLY_EOF_SRC_CURATED_CONFIG_PY'
"""Single source of truth for the curated evaluation subset and pruning
schedule, shared across every script in the (now streamlined) pipeline.

Replaces the earlier full-taxonomy (77-class, ~7,200-image balanced)
evaluation scope. The dominant compute cost throughout this pipeline has
always been re-encoding the candidate image pool through the vision
encoder at every pruning level -- restricting that pool to a small,
curated, taxonomically well-spread set of classes cuts that cost by
roughly two orders of magnitude, which matters far more than any other
lever (pruning schedule granularity, scenario count, etc).

CURATED_CLASSES: 10 well-known, easily-recognizable specific classes, one
per coordinate (basic-level) group for maximum taxonomic spread -- 8
animals (each from a different coordinate group: Domestic Dogs, Felines,
Primates, Birds, Reptiles, Amphibians, Fish, Insects) + 2 plants (Citrus,
Fruits). All 10 are already downloaded and verified against
data/taxonomy_full.json.
"""

import numpy as np

CURATED_CLASSES = [
    "German Shepherd",     # Domestic Dogs
    "Tabby Cat",            # Felines
    "Chimpanzee",           # Primates
    "King Penguin",         # Birds (atypical -- flightless, useful contrast)
    "American Alligator",   # Reptiles
    "Bullfrog",              # Amphibians
    "Goldfish",               # Fish
    "Monarch Butterfly",    # Insects
    "Lemon",                  # Plants: Citrus
    "Banana",                 # Plants: Fruits
]

# 2.5% increments, 0% to 70% inclusive (29 levels). Concentrates resolution
# in the region where the real pruning sweep showed the interesting
# hierarchical transitions happening (roughly 20-60%), and drops the
# 70-90% tail, which prior real results showed was already flat at floor
# performance -- i.e. redundant data points that cost compute without
# adding information.
PRUNING_LEVELS_FOCUSED = [round(x, 3) for x in np.arange(0.00, 0.701, 0.025).tolist()]
APPLY_EOF_SRC_CURATED_CONFIG_PY

cat > "src/generate_tsne_curated.py" << 'APPLY_EOF_SRC_GENERATE_TSNE_CURATED_PY'
"""The one tSNE visualization for this project.

Previously the codebase had FIVE separate, largely-duplicate tSNE
implementations: src/generate_tsne.py, src/extended_hierarchical_eval.py's
generate_hierarchical_hsv_tsne, and three standalone scripts/modules
(run_tsne_5stages.py + generate_tsne_subcategories.py, run_tSNE_nerr.py +
gen_tSNE_subcat_nerr.py, run_tsne_no_errors.py +
generate_tsne_subcat_no_error.py). All five are superseded by this single
module, scoped to the curated 10-class subset (src/curated_config.py) for
both speed (re-encoding ~30-50 images per pruning level instead of ~7,200)
and interpretability (10 named, well-known classes with a stable color
per class, rather than domain-grouped colormap families that needed a
legend of their own to decode).
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_CLASSES

# tab10 gives 10 maximally-distinct qualitative colors -- exactly matching
# the 10 curated classes, one color each, no palette-family logic needed.
_CMAP = plt.get_cmap("tab10")
CLASS_COLORS = {cls: _CMAP(i) for i, cls in enumerate(CURATED_CLASSES)}


def generate_curated_tsne_grid(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
) -> str:
    """Generates one tSNE grid: a panel per pruning level (default 6
    representative stages spanning 0-70%), points colored by specific
    class, fit globally across all stacked stages so positions are
    directly comparable panel to panel.
    """
    if pruning_levels_to_show is None:
        pruning_levels_to_show = [0.0, 0.15, 0.30, 0.45, 0.60, 0.70]
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    spec_col = "specific" if "specific" in evaluator.valid_metadata.columns else "concept"
    meta = evaluator.valid_metadata.reset_index(drop=True)

    all_feats = []
    true_labels = None

    for p_level in pruning_levels_to_show:
        print(f"[*] tSNE: extracting features at {p_level * 100:.1f}% pruning...")
        if scenario == "joint":
            amount = p_level
        elif scenario == "vision_only":
            amount = {"text": 0.0, "vision": p_level}
        else:
            amount = {"text": p_level, "vision": 0.0}

        pruned_model = evaluator.pruning_engine.get_pruned_model(
            amount=amount, encoder_type="joint", target_area="full"
        )
        img_feats, _, labels, _, _ = evaluator._extract_joint_features(pruned_model)
        img_feats_np = img_feats.detach().cpu().numpy() if isinstance(img_feats, torch.Tensor) else img_feats
        all_feats.append(img_feats_np)
        if true_labels is None:
            true_labels = labels

    X_total = np.vstack(all_feats)
    n_per_stage = len(true_labels)
    perplexity = min(30, max(5, n_per_stage // 3))

    print(f"[*] Fitting global t-SNE across {len(pruning_levels_to_show)} stacked stages "
          f"({X_total.shape[0]} points total, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    X_2d = tsne.fit_transform(X_total)

    n_stages = len(pruning_levels_to_show)
    n_cols = min(3, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.0 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    xlim = (X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5)
    ylim = (X_2d[:, 1].min() - 5, X_2d[:, 1].max() + 5)

    for stage_idx, (ax, p_level) in enumerate(zip(axes_flat, pruning_levels_to_show)):
        start = stage_idx * n_per_stage
        end = start + n_per_stage
        stage_points = X_2d[start:end]

        for cls in CURATED_CLASSES:
            cls_mask = [lbl == cls for lbl in true_labels]
            if not any(cls_mask):
                continue
            pts = stage_points[cls_mask]
            ax.scatter(
                pts[:, 0], pts[:, 1], s=40, color=CLASS_COLORS[cls],
                alpha=0.85, edgecolor="white", linewidth=0.4, label=cls,
            )

        ax.set_title(f"{p_level * 100:.1f}% Pruned", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    for ax in axes_flat[n_stages:]:
        ax.axis("off")

    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CURATED_CLASSES]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Joint Embedding Space t-SNE Across Pruning Levels ({scenario} scenario, curated 10-class subset)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_path = os.path.join(output_dir, "tsne_curated.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] tSNE grid saved to: {save_path}")
    return save_path
APPLY_EOF_SRC_GENERATE_TSNE_CURATED_PY

cat > "run_tsne.py" << 'APPLY_EOF_RUN_TSNE_PY'
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_tsne_curated import generate_curated_tsne_grid
from src.curated_config import CURATED_CLASSES


def main(output_dir=None, scenario="joint"):
    print("=" * 60)
    print(" CURATED t-SNE GRID (10-class subset) ")
    print("=" * 60)
    print(f"[*] Classes: {CURATED_CLASSES}")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    evaluator = JointSpaceEvaluator(restrict_classes=CURATED_CLASSES, balance_taxonomically=True)

    save_path = generate_curated_tsne_grid(evaluator, scenario=scenario, output_dir=output_dir)
    print(f"\n[+] Done: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the curated 10-class tSNE grid.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario)
APPLY_EOF_RUN_TSNE_PY

cat > "src/joint_evaluator.py" << 'APPLY_EOF_SRC_JOINT_EVALUATOR_PY'
import os
import sys
import clip
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pruning_engine import CLIPPruningEngine
from src.metrics import (
    compute_mrr,
    compute_entropy,
    compute_typicality_delta,
)


class EvaluationImageDataset(Dataset):
    """Dataset loader for evaluation images with recursive and relative path resolution."""

    def __init__(self, metadata, preprocess, concept_col="specific"):
        self.metadata = metadata.reset_index(drop=True)
        self.preprocess = preprocess

        if concept_col not in self.metadata.columns:
            for alt in ["specific", "coordinate", "concept", "label", "class", "category"]:
                if alt in self.metadata.columns:
                    concept_col = alt
                    break
        self.concept_col = concept_col

        path_col = None
        for alt in ["filename", "image_path", "filepath", "file_path", "path", "img_path", "resolved_filepath"]:
            if alt in self.metadata.columns:
                path_col = alt
                break

        if not path_col:
            raise ValueError(
                f"[!] Could not find an image path column in metadata. Present columns: {list(self.metadata.columns)}"
            )

        self.valid_indices = []
        self.samples = []

        cwd = os.getcwd()
        search_dirs = [
            cwd,
            os.path.join(cwd, "data"),
            os.path.join(cwd, "data", "processed"),
            os.path.join(cwd, "data", "processed", "images"),
            os.path.join(cwd, "data", "images"),
            os.path.join(cwd, "data", "raw"),
            os.path.join(cwd, "data", "tiny-imagenet-200"),
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, "data"),
        ]

        # Pre-build filename mapping across data directories
        file_map = {}
        data_dir = os.path.join(PROJECT_ROOT, "data")
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        if f not in file_map:
                            file_map[f] = os.path.join(root, f)

        for idx, row in self.metadata.iterrows():
            raw_path = str(row.get(path_col, "")).strip()
            if not raw_path or raw_path.lower() == "nan":
                continue

            cleaned_rel = raw_path.lstrip("./")
            fname = os.path.basename(raw_path)

            candidate_paths = [
                raw_path,
                os.path.abspath(raw_path),
                os.path.join(cwd, cleaned_rel),
                os.path.join(PROJECT_ROOT, cleaned_rel),
            ]
            for s_dir in search_dirs:
                candidate_paths.append(os.path.join(s_dir, cleaned_rel))
                candidate_paths.append(os.path.join(s_dir, fname))

            if fname in file_map:
                candidate_paths.append(file_map[fname])

            valid_path = None
            for p in candidate_paths:
                if os.path.isfile(p):
                    valid_path = p
                    break

            if valid_path:
                self.valid_indices.append(idx)
                self.samples.append((valid_path, row[self.concept_col]))

        if len(self.samples) == 0:
            raise ValueError("[!] No valid image paths found in metadata.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, concept = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return image_tensor, concept


class JointSpaceEvaluator:
    """Evaluates CLIP representation degradation under magnitude pruning."""

    def __init__(
        self,
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device="cpu",
        batch_size=32,
        sample_frac=1.0,
        target_n=None,
        balance_taxonomically=True,
        restrict_classes=None,
    ):
        if not os.path.exists(metadata_path) and os.path.exists(os.path.join(PROJECT_ROOT, metadata_path)):
            metadata_path = os.path.join(PROJECT_ROOT, metadata_path)

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size

        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        print(f"[*] Loading metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)

        spec_col = "specific" if "specific" in df.columns else ("concept" if "concept" in df.columns else "category")

        # Curated-subset restriction (src/curated_config.py): the dominant
        # compute cost throughout this pipeline is re-encoding the
        # candidate image pool at every pruning level, so restricting to a
        # small, well-spread, curated set of classes up front -- before
        # any balancing/sampling below -- is what actually makes runs
        # fast, independent of pruning-schedule granularity or scenario
        # count.
        if restrict_classes is not None and spec_col in df.columns:
            before = len(df)
            df = df[df[spec_col].isin(restrict_classes)].reset_index(drop=True)
            print(
                f"[*] Restricted to curated class subset ({len(restrict_classes)} classes): "
                f"{len(df)}/{before} rows kept."
            )
            missing = set(restrict_classes) - set(df[spec_col].unique())
            if missing:
                print(f"[!] WARNING: {missing} not found in metadata -- check spelling/availability.")

        # Taxonomic balancing / sampling
        if balance_taxonomically and spec_col in df.columns:
            n_classes = df[spec_col].nunique()
            if target_n is not None and target_n > 0:
                samples_per_class = max(1, target_n // n_classes)
            else:
                samples_per_class = df.groupby(spec_col).size().min()

            # NOTE: previously used df.groupby(spec_col, group_keys=False)
            # .apply(lambda x: x.sample(...)) -- on recent pandas versions,
            # .groupby().apply() SILENTLY EXCLUDES the grouping column from
            # the result by default (a real, confirmed behavior change, not
            # a hypothetical). That means spec_col ("specific") itself was
            # being dropped from `df` right here, for every single
            # JointSpaceEvaluator instantiation using the default
            # balance_taxonomically=True. Downstream code with a fallback
            # (EvaluationImageDataset) silently self-corrected to a coarser
            # column (e.g. "coordinate") instead of crashing -- meaning
            # evaluations were silently running at the wrong taxonomic
            # granularity with no error at all. Code without a fallback
            # (gen_tSNE_subcat_nerr.py) crashed with a confusing KeyError.
            # Fixed by iterating groups directly and reconstructing via
            # .loc[], which never triggers the grouping-column-exclusion
            # behavior since .apply() is never called.
            selected_idx = []
            for _, group in df.groupby(spec_col, group_keys=False):
                n = min(len(group), samples_per_class)
                selected_idx.extend(group.sample(n=n, random_state=42).index.tolist())
            df = df.loc[selected_idx].reset_index(drop=True)
            print(
                f"[*] Taxonomically balanced: {len(df)} total samples equalized across {n_classes} classes."
            )
        elif target_n is not None and target_n > 0:
            df = df.sample(n=min(len(df), target_n), random_state=42).reset_index(drop=True)
            print(f"[*] Sampled dataset to {len(df)} samples.")
        elif sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            print(f"[*] Downsampled dataset to {len(df)} samples.")

        print(f"[*] Loading CLIP model ({model_name}) on {self.device}...")
        self.base_model, self.preprocess = clip.load(model_name, device=self.device)
        self.base_model.eval()

        # Build pruning engine interface
        self.pruning_engine = CLIPPruningEngine(self.base_model)

        # Pre-validate image paths and construct dataset aliases required by downstream scripts
        self._dataset = EvaluationImageDataset(df, self.preprocess, spec_col)
        self.valid_metadata = df.iloc[self._dataset.valid_indices].reset_index(drop=True)
        self.valid_metadata["resolved_filepath"] = [s[0] for s in self._dataset.samples]

        # Set aliases expected across analysis modules
        self.df = self.valid_metadata
        self.metadata = self.valid_metadata
        self.image_paths = self.valid_metadata["resolved_filepath"].tolist()

    def _reindex_visual_memory(self, model=None) -> torch.Tensor:
        """Encodes images with the current/provided model state and returns normalized features."""
        target_model = model if model is not None else self.base_model
        target_model.eval()
        target_model.to(self.device)

        spec_col = "specific" if "specific" in self.valid_metadata.columns else "concept"
        dataset = EvaluationImageDataset(self.valid_metadata, self.preprocess, spec_col)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_feats_list = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                feats = target_model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats_list.append(feats.cpu())

        return torch.cat(img_feats_list, dim=0)

    def _extract_joint_features(self, model):
        model.eval()
        spec_col = (
            "specific" if "specific" in self.valid_metadata.columns else "concept"
        )
        dataset = EvaluationImageDataset(self.valid_metadata, self.preprocess, spec_col)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_feats_list, concept_labels = [], []
        with torch.no_grad():
            for imgs, concepts in tqdm(dataloader, desc="Extracting Features", leave=False):
                imgs = imgs.to(self.device)
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats_list.append(feats.cpu())
                concept_labels.extend(concepts)

            img_feats = torch.cat(img_feats_list, dim=0)
            unique_concepts = sorted(list(set(concept_labels)))
            # Bare single-word Bozeat query (Directive 3.1) -- no conversational template.
            text_prompts = [c.lower() for c in unique_concepts]
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            text_feats = model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return (
            img_feats,
            text_feats.cpu(),
            concept_labels,
            unique_concepts,
            self.valid_metadata,
        )

    def run_eval(
        self,
        pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95],
        target_area="full",
        scenario="joint",
        masking_mode="static",
        pruning_method="l1_unstructured",
        depth_zone="global",
        sub_module="all",
    ):
        """Runs the pruning sweep and returns a metrics DataFrame.

        Args:
            scenario: "joint" (both encoders pruned together) or
              "vision_only" (only the vision encoder is pruned, text stays
              pristine). "text_only" is still accepted by the pruning
              engine underneath if ever needed again, but the streamlined
              pipeline only exercises joint/vision_only -- the two
              scenarios currently in scope.
            masking_mode: "static" (default) or "iterative" -- see
              CLIPPruningEngine.get_pruned_model docstring.

        Metrics kept: Top-1 specific/coordinate/superordinate accuracy,
        MRR, Shannon entropy, the 4-tier clinical error breakdown
        (coordinate/superordinate/domain error/domain collapse), and
        typicality delta where applicable. CKA, neighborhood preservation,
        the Bfloor noise-floor reference, and the top-10 depth breakdown
        have been removed as out of scope for the current focused
        pipeline -- compute_cka/compute_neighborhood_preservation/
        compute_top10_breakdown_depth remain implemented in
        src/metrics.py if needed again later (src/baseline_conditions.py,
        which implemented Bfloor, was deleted since nothing else used it --
        reimplementing gaussian_noise_floor_features is a few lines if
        it's ever needed again; see git history).
        """
        if scenario not in ("joint", "text_only", "vision_only"):
            raise ValueError("scenario must be 'joint', 'text_only', or 'vision_only'.")

        if masking_mode == "iterative":
            self.pruning_engine.reset_accumulator()
            if list(pruning_levels) != sorted(pruning_levels):
                raise ValueError(
                    "masking_mode='iterative' requires pruning_levels to be ascending."
                )

        print("[*] Extracting base unpruned features...")
        ref_img, ref_text, labels, unique_concepts, eval_meta = (
            self._extract_joint_features(self.base_model)
        )

        spec_col = (
            "specific" if "specific" in self.valid_metadata.columns else "concept"
        )
        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        target_indices = torch.tensor([concept_to_idx[c] for c in labels])
        target_indices_np = target_indices.numpy()

        results = []

        for p_level in pruning_levels:
            print(f"\n[+] Evaluating Pruning Level: {p_level * 100:.1f}% (scenario={scenario}, masking_mode={masking_mode})")

            if scenario == "joint":
                amount_spec = p_level
            elif scenario == "text_only":
                amount_spec = {"text": p_level, "vision": 0.0}
            else:  # vision_only
                amount_spec = {"text": 0.0, "vision": p_level}

            pruned_model = self.pruning_engine.get_pruned_model(
                amount=amount_spec,
                encoder_type="joint",
                pruning_method=pruning_method,
                target_area=target_area,
                depth_zone=depth_zone,
                sub_module=sub_module,
                masking_mode=masking_mode,
            )
            p_img, p_text, _, _, _ = self._extract_joint_features(pruned_model)

            sim_matrix = torch.matmul(p_img, p_text.T)
            sim_matrix_np = sim_matrix.numpy()
            top1_preds = torch.argmax(sim_matrix, dim=1)
            correct_mask = top1_preds == target_indices

            spec_acc = correct_mask.float().mean().item()

            coordinate_acc, super_acc = spec_acc, spec_acc
            if (
                "coordinate" in eval_meta.columns
                and "superordinate" in eval_meta.columns
            ):
                coordinate_correct, super_correct = 0, 0
                for i, pred_idx in enumerate(top1_preds):
                    pred_c = unique_concepts[pred_idx.item()]
                    pred_match = self.valid_metadata[
                        self.valid_metadata[spec_col] == pred_c
                    ]
                    if not pred_match.empty:
                        p_row = pred_match.iloc[0]
                        t_row = eval_meta.iloc[i]
                        if p_row.get("coordinate") == t_row.get("coordinate"):
                            coordinate_correct += 1
                        if p_row.get("superordinate") == t_row.get("superordinate"):
                            super_correct += 1
                coordinate_acc = coordinate_correct / len(eval_meta)
                super_acc = super_correct / len(eval_meta)

            mrr = compute_mrr(sim_matrix_np, target_indices_np)
            entropy = compute_entropy(sim_matrix_np)

            coord_err, super_err, domain_err, collapse_err = 0, 0, 0, 0
            for i, is_corr in enumerate(correct_mask):
                if not is_corr:
                    pred_c = unique_concepts[top1_preds[i].item()]
                    t_row = eval_meta.iloc[i]
                    p_match = self.valid_metadata[self.valid_metadata[spec_col] == pred_c]
                    if not p_match.empty:
                        p_row = p_match.iloc[0]
                        if (
                            "coordinate" in t_row
                            and p_row.get("coordinate") == t_row.get("coordinate")
                        ):
                            coord_err += 1
                        elif (
                            "superordinate" in t_row
                            and p_row.get("superordinate") == t_row.get("superordinate")
                        ):
                            super_err += 1
                        elif (
                            "domain" in t_row
                            and p_row.get("domain") == t_row.get("domain")
                        ):
                            domain_err += 1
                        else:
                            collapse_err += 1
                    else:
                        collapse_err += 1

            row = {
                "pruning_level": p_level,
                "scenario": scenario,
                "masking_mode": masking_mode,
                "pruning_method": pruning_method,
                "top1_specific_acc": spec_acc,
                "top1_coordinate_acc": coordinate_acc,
                "top1_super_acc": super_acc,
                "mrr": mrr,
                "semantic_entropy": entropy,
                "coordinate error": coord_err,
                "superordinate error": super_err,
                "domain error": domain_err,
                "domain collapse": collapse_err,
            }

            # Typicality Delta (Sec. 3.4, "typicality effect"): only
            # meaningful once the metadata actually contains both Typical
            # and Atypical exemplars per coordinate class.
            if "typicality" in eval_meta.columns:
                typ = eval_meta["typicality"].astype(str).str.strip().str.casefold()
                if (typ == "atypical").any() and (typ == "typical").any():
                    row.update(
                        compute_typicality_delta(sim_matrix_np, target_indices_np, eval_meta)
                    )

            results.append(row)

        return pd.DataFrame(results)
APPLY_EOF_SRC_JOINT_EVALUATOR_PY

cat > "src/bozeat_experiment.py" << 'APPLY_EOF_SRC_BOZEAT_EXPERIMENT_PY'
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

            title_text = f"'{ret_concept}' {'\u2713' if is_correct else '\u2717'}"
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
APPLY_EOF_SRC_BOZEAT_EXPERIMENT_PY

cat > "run_bozeat_grid.py" << 'APPLY_EOF_RUN_BOZEAT_GRID_PY'
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.bozeat_experiment import run_bozeat_experiment
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED


def main(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")

    print("=" * 60)
    print(" BOZEAT TEXT->IMAGE RETRIEVAL EXPERIMENT (curated 10-class subset) ")
    print("=" * 60)
    print(f"[*] Prompts ({len(CURATED_CLASSES)}): {CURATED_CLASSES}")
    print(f"[*] Pruning Grid ({len(PRUNING_LEVELS_FOCUSED)} stages, 2.5% steps, 0-70%)")

    # restrict_classes here makes BOTH the query prompts and the retrieval
    # candidate pool the curated 10-class subset -- this is what makes the
    # experiment fast (re-encoding ~30-50 images per level instead of the
    # full balanced dataset).
    evaluator = JointSpaceEvaluator(restrict_classes=CURATED_CLASSES, balance_taxonomically=True)

    csv_path, grid_path = run_bozeat_experiment(
        evaluator,
        target_prompts=CURATED_CLASSES,
        pruning_levels=PRUNING_LEVELS_FOCUSED,
        output_dir=output_dir,
    )

    print(f"\n[+] Done.\n    Results CSV: {csv_path}\n    Image grid: {grid_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bozeat retrieval experiment.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
APPLY_EOF_RUN_BOZEAT_GRID_PY

cat > "run_pipeline.py" << 'APPLY_EOF_RUN_PIPELINE_PY'
import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_analysis_plots import plot_category_breakdown_suite
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED


def _plot_scenario_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Joint (both encoders pruned) vs. Vision-Only (only the vision
    encoder pruned, text stays pristine) -- the two scenarios currently in
    scope. Text-only was dropped from the active comparison per the
    project's current focus, though JointSpaceEvaluator.run_eval still
    accepts scenario="text_only" if it's ever needed again.
    """
    scenario_labels = {
        "joint": "Joint (both encoders pruned)",
        "vision_only": "Vision-Only (vision encoder pruned, text pristine)",
    }
    colors = {"joint": "#2ca02c", "vision_only": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    for ax, col, title in [
        (axes[0], "top1_specific_acc", "Top-1 Specific Accuracy"),
        (axes[1], "semantic_entropy", "Semantic Vector Entropy (bits)"),
    ]:
        for scenario, label in scenario_labels.items():
            sub = df[df["scenario"] == scenario].sort_values("pruning_level")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(
                sub["pruning_level"] * 100, sub[col],
                color=colors[scenario], marker="o", markersize=3,
                linewidth=1.8, label=label,
            )
        ax.set_xlabel("Pruning Level (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Joint vs. Vision-Only Pruning Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "scenario_comparison_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Scenario comparison plot saved to: {save_path}")


def run_joint_pipeline(
    output_dir: str | None = None,
    curated_classes: list[str] | None = None,
    pruning_levels: list[float] | None = None,
) -> str:
    """Core evaluation: curated 10-class subset, joint + vision_only
    scenarios, 2.5% pruning increments from 0% to 70%.
    """
    if curated_classes is None:
        curated_classes = CURATED_CLASSES
    if pruning_levels is None:
        pruning_levels = PRUNING_LEVELS_FOCUSED

    print("=" * 60)
    print(" CORE EVALUATION PIPELINE (curated 10-class subset) ")
    print("=" * 60)
    print(f"[*] Classes ({len(curated_classes)}): {curated_classes}")
    print(f"[*] Pruning Grid ({len(pruning_levels)} stages, 2.5% steps, 0-70%)")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    evaluator = JointSpaceEvaluator(
        restrict_classes=curated_classes,
        balance_taxonomically=True,
    )

    print("\n[*] Running JOINT scenario (both encoders pruned)...")
    joint_df = evaluator.run_eval(
        pruning_levels=pruning_levels, target_area="full", scenario="joint"
    )

    print("\n[*] Running VISION-ONLY scenario (vision encoder pruned, text pristine)...")
    vision_only_df = evaluator.run_eval(
        pruning_levels=pruning_levels, target_area="full", scenario="vision_only"
    )

    csv_path = os.path.join(output_dir, "joint_space_metrics.csv")
    joint_df.to_csv(csv_path, index=False)
    print(f"\n[+] Joint-scenario metrics saved to:\n    {csv_path}")

    scenario_comparison_df = pd.concat([joint_df, vision_only_df], ignore_index=True)
    scenario_csv = os.path.join(output_dir, "scenario_comparison.csv")
    scenario_comparison_df.to_csv(scenario_csv, index=False)
    print(f"[+] Scenario comparison metrics saved to:\n    {scenario_csv}")
    _plot_scenario_comparison(scenario_comparison_df, output_dir)

    # THE cross-category plot (accuracy + 4-tier clinical error heatmap),
    # computed from the joint scenario.
    plot_category_breakdown_suite(joint_df, output_dir=output_dir)

    print("\n[+] Core evaluation pipeline complete.")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the core curated-subset CLIP pruning evaluation."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Custom directory to save output results and plots"
    )
    args = parser.parse_args()

    run_joint_pipeline(output_dir=args.output_dir)
APPLY_EOF_RUN_PIPELINE_PY

cat > "run_full_pipeline.py" << 'APPLY_EOF_RUN_FULL_PIPELINE_PY'
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Root-level entry point imports
try:
    from run_bozeat_grid import main as run_bozeat_grid_main
    from run_indexing import run_indexing
    from run_pipeline import run_joint_pipeline
    from run_tsne import main as run_tsne_main
except ImportError as e:
    print(f"[!] Import error detected: {e}")
    print("[!] Ensure all run_*.py scripts are present in the project root.")
    sys.exit(1)


def prepare_directories(base_dir: str) -> dict[str, str]:
    """Ensures all output and data directories exist prior to pipeline execution."""
    dirs = {
        "data_images": os.path.join(base_dir, "data", "images"),
        "data_processed": os.path.join(base_dir, "data", "processed"),
        "results_base": os.path.join(base_dir, "data", "results"),
        "results_bozeat": os.path.join(base_dir, "data", "results", "bozeat_experiment"),
        "logs": os.path.join(base_dir, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def execute_master_pipeline(
    skip_indexing: bool = False,
    output_dir: str | None = None,
) -> None:
    """Streamlined 4-stage pipeline, scoped to the curated 10-class subset
    (src/curated_config.py) throughout:

      1. Visual memory indexing (once, over the full downloaded dataset --
         cheap and only needs re-running when new images are added).
      2. Core evaluation: joint vs. vision-only scenario comparison, the
         cross-category clinical error breakdown, 2.5% pruning increments
         from 0-70%.
      3. Bozeat text->image retrieval experiment.
      4. One consolidated tSNE grid.

    Previously a 9-stage pipeline evaluating the full ~7,200-image
    balanced dataset at every stage; this is the redesigned, focused
    version -- see src/curated_config.py for the rationale.
    """
    start_time = time.time()
    dirs = prepare_directories(PROJECT_ROOT)
    results_dir = output_dir if output_dir else dirs["results_base"]

    print("=" * 70)
    print("  SEMANTIC DEMENTIA CLIP ATROPHY SIMULATION -- CURATED PIPELINE  ")
    print("=" * 70)
    print(f"[*] Root Directory        : {PROJECT_ROOT}")
    print(f"[*] Results Output Path   : {results_dir}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Offline Visual Memory Bank Indexing
    # -------------------------------------------------------------------------
    if not skip_indexing:
        print("\n[STAGE 1/4] Building Visual Memory Index...")
        try:
            indexed_path = run_indexing(
                raw_csv=os.path.join(PROJECT_ROOT, "tests", "metadata_raw.csv"),
                img_dir=dirs["data_images"],
                output_dir=dirs["data_processed"],
            )
            print(f"[+] Stage 1 Complete. Saved tensor bank to: {indexed_path}")
        except Exception as err:
            print(f"[!] Stage 1 Failed: {err}")
            sys.exit(1)
    else:
        print("\n[STAGE 1/4] Skipping Visual Memory Indexing (--skip_indexing set).")

    stage_failures = []

    # -------------------------------------------------------------------------
    # STAGE 2: Core Evaluation (joint vs. vision-only, cross-category plot)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/4] Running Core Evaluation (curated 10-class subset)...")
    try:
        csv_metrics = run_joint_pipeline(output_dir=results_dir)
        print(f"[+] Stage 2 Complete. Metrics exported to: {csv_metrics}")
    except Exception as err:
        print(f"[!] Stage 2 Failed: {err}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: Bozeat Text->Image Retrieval Experiment
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/4] Running Bozeat Retrieval Experiment...")
    try:
        run_bozeat_grid_main(output_dir=dirs["results_bozeat"])
        print("[+] Stage 3 Complete.")
    except Exception as err:
        print(f"[!] Stage 3 Failed: {err}")
        stage_failures.append(("Stage 3 (Bozeat Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 4: Curated t-SNE Grid
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/4] Generating Curated t-SNE Grid...")
    try:
        run_tsne_main(output_dir=results_dir)
        print("[+] Stage 4 Complete.")
    except Exception as err:
        print(f"[!] Stage 4 Failed: {err}")
        stage_failures.append(("Stage 4 (t-SNE Grid)", err))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    if stage_failures:
        print(f" [!] MASTER PIPELINE FINISHED WITH {len(stage_failures)} FAILED STAGE(S) IN {elapsed:.2f}s")
        for stage_name, err in stage_failures:
            print(f"     - {stage_name}: {err}")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f" [+] MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for the curated Semantic Dementia CLIP Pipeline."
    )
    parser.add_argument(
        "--skip_indexing",
        action="store_true",
        help="Skip image preprocessing and feature indexing (Stage 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output path for results",
    )
    args = parser.parse_args()

    execute_master_pipeline(
        skip_indexing=args.skip_indexing,
        output_dir=args.output_dir,
    )
APPLY_EOF_RUN_FULL_PIPELINE_PY

cat > "src/generate_analysis_plots.py" << 'APPLY_EOF_SRC_GENERATE_ANALYSIS_PLOTS_PY'
# NOTE: this file previously also contained plot_hierarchical_breakdown_suite
# (redundant with plot_category_breakdown_suite's own fine-to-coarse framing),
# plot_signal_noise_distribution_shift, and plot_concept_retrieval_heatmap --
# all removed as out-of-scope extras for the current focused pipeline
# (curated 10-class subset; cross-category breakdown + Bozeat + one tSNE
# grid + joint-vs-vision-only scenario comparison). Only the one function
# actually still in use remains here.

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
    """THE cross-category plot: Top-1 accuracy trajectory + a proportion-based
    clinical error-taxonomy heatmap (Coordinate/Superordinate/Domain
    Error/Domain Collapse) across pruning levels.

    NOTE: previously plotted a "Vision CKA" line (now removed -- CKA
    computation was cut from run_eval as out-of-scope for the current
    focused pipeline) and a decorative, non-empirical "Expected Theory
    Bound" dashed curve (removed -- it wasn't derived from anything, just
    a quadratic decay placeholder). The heatmap previously showed raw
    error COUNTS with annotation labels that visually overlapped once
    there were more than a handful of pruning-level columns; it now shows
    PROPORTIONS (bounded 0-1, comparable across runs with different
    dataset sizes) with annotation font scaled down and made optional for
    wide grids.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Top-1 Specific Accuracy
    acc_col = next(
        (c for c in ["top1_specific_acc", "i2t_top1", "Correct"] if c in df.columns),
        None,
    )
    if acc_col == "Correct":
        total = df[["Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse"]].sum(axis=1)
        acc_series = df["Correct"] / total
    elif acc_col is not None:
        acc_series = df[acc_col]
    else:
        acc_series = None

    if acc_series is not None:
        ax1.plot(
            prune_pcts, acc_series, marker="o", color="#1f77b4",
            linewidth=2.5, label="Top-1 Specific Accuracy",
        )

    ax1.set_xlabel("Pruning Level (%)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Top-1 Accuracy vs. Pruning Level", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True)
    ax1.set_ylim(-0.02, 1.02)

    # Panel 2: Error Taxonomy Heatmap (proportions, not raw counts)
    err_cols = [
        c for c in [
            "coordinate error", "superordinate error", "domain error", "domain collapse",
        ] if c in df.columns
    ]

    if err_cols:
        err_sum = df[err_cols].sum(axis=1)
        # total_samples derived from: err_sum = total * (1 - accuracy)
        if acc_series is not None:
            total_samples = err_sum / (1 - acc_series).replace(0, np.nan)
        else:
            total_samples = err_sum.replace(0, np.nan)
        heatmap_data = (df[err_cols].div(total_samples, axis=0)).T
        heatmap_data.columns = [f"{p:.1f}%" for p in prune_pcts]
        heatmap_data.index = [c.replace("_", " ").title() for c in err_cols]

        n_cols = len(heatmap_data.columns)
        show_annot = n_cols <= 20  # avoid unreadable overlap on wide grids
        sns.heatmap(
            heatmap_data,
            annot=show_annot,
            fmt=".2f",
            annot_kws={"size": 7} if show_annot else None,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            ax=ax2,
            cbar=True,
            cbar_kws={"label": "Proportion of outcomes"},
        )
        ax2.set_title("Clinical Error Taxonomy (proportion of outcomes)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Pruning Level (%)", fontsize=11)
        ax2.tick_params(axis="x", labelsize=7, rotation=90)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cross_category_breakdown.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved cross-category breakdown plot to: {save_path}")
APPLY_EOF_SRC_GENERATE_ANALYSIS_PLOTS_PY

cat > "README.md" << 'APPLY_EOF_README_MD'
# Simulating Semantic Dementia Through Progressive Pruning of the CLIP Embedding Space

BSc Honours research codebase, University of the Witwatersrand.

## Current scope (streamlined)

This pipeline was redesigned around a **curated 10-class subset** rather
than the full 77-class taxonomy, after the full-dataset version proved too
slow for practical iteration. See `src/curated_config.py` for the single
source of truth.

**Curated classes** (8 animals, one per coordinate/basic-level group for
maximum taxonomic spread, + 2 plants):

| Class | Coordinate Group | Typicality |
|---|---|---|
| German Shepherd | Domestic Dogs | Typical |
| Tabby Cat | Felines | Typical |
| Chimpanzee | Primates | Typical |
| King Penguin | Birds | Atypical (flightless) |
| American Alligator | Reptiles | Typical |
| Bullfrog | Amphibians | Typical |
| Goldfish | Fish | Typical |
| Monarch Butterfly | Insects | Typical |
| Lemon | Citrus (Plants) | Typical |
| Banana | Fruits (Plants) | Typical |

Balances to 100 samples/class = **1,000 total images** (limited by the 5
classes downloaded in the earliest data-collection pass) -- roughly a 7x
reduction from the previous ~7,200-image balanced dataset, which was the
dominant compute cost throughout the pipeline.

**Pruning schedule:** 2.5% increments, 0% to 70% inclusive (29 levels) --
`src/curated_config.py::PRUNING_LEVELS_FOCUSED`. Concentrates resolution
in the region where real pruning sweeps showed the interesting
hierarchical transitions happening; drops the 70-90% tail, which prior
results showed was already flat at floor performance.

**Scenarios:** Joint (both encoders pruned) vs. Vision-Only (vision
encoder pruned, text stays pristine). Text-only was dropped from the
active comparison -- `JointSpaceEvaluator.run_eval()` still accepts
`scenario="text_only"` if it's ever needed again.

**Metrics kept:** Top-1 specific/coordinate/superordinate accuracy, MRR,
Shannon entropy, the 4-tier clinical error breakdown (Coordinate
Error/Superordinate Error/Domain Error/Domain Collapse), typicality delta.
CKA, neighborhood preservation, the Bfloor noise-floor reference, and the
top-10 depth breakdown were cut as out of scope for the current focus --
`compute_cka`/`compute_neighborhood_preservation`/
`compute_top10_breakdown_depth` remain in `src/metrics.py` if needed
again.

## Pipeline (`run_full_pipeline.py`, 4 stages)

1. **Visual Memory Indexing** (`run_indexing.py`) -- encodes the full
   downloaded dataset through pristine CLIP once. Only needs re-running
   when new images are added.
2. **Core Evaluation** (`run_pipeline.py`) -- Joint vs. Vision-Only
   scenario comparison, the cross-category clinical error breakdown plot,
   over the curated subset and focused pruning schedule.
3. **Bozeat Experiment** (`run_bozeat_grid.py` /
   `src/bozeat_experiment.py`) -- text->image retrieval ("a duck with four
   legs" analogue) using all 10 curated classes as BOTH the query prompts
   and the retrieval candidate pool (this restriction is what makes it
   fast -- previously it searched the full balanced dataset for every
   query). Produces a full-resolution retrieval-accuracy/confidence curve
   across all 29 levels, plus a readable qualitative image grid at a
   representative subset of levels.
4. **Curated t-SNE Grid** (`run_tsne.py` /
   `src/generate_tsne_curated.py`) -- one consolidated tSNE visualization,
   points colored by specific class (10 distinct colors), panels at 6
   representative pruning stages (0/15/30/45/60/70%).

Stage 1 (indexing) and Stage 2 (core evaluation) are fatal if they fail --
nothing downstream can run without their output. Stages 3-4 (Bozeat,
t-SNE) fail independently without halting the run; the final summary
reports which, if any, failed.

## What was removed (previously a 9-stage pipeline)

- **Text-only scenario** -- dropped from the active comparison.
- **Five duplicate t-SNE implementations** consolidated into one:
  `run_tsne_5stages.py`+`generate_tsne_subcategories.py`,
  `run_tSNE_nerr.py`+`gen_tSNE_subcat_nerr.py`,
  `run_tsne_no_errors.py`+`generate_tsne_subcat_no_error.py`,
  `generate_tsne.py`, and `extended_hierarchical_eval.py`'s
  `generate_hierarchical_hsv_tsne` -- all deleted.
- **RQ2 depth-zone grid, static-vs-iterative masking comparison, pruning-
  strategy comparison** (`run_depth_zone_grid.py`,
  `run_masking_mode_comparison.py`, `run_pruning_strategy_comparison.py`)
  -- deleted; extra scope from an earlier, broader phase of this project
  not part of the current focused set of questions.
- **`TestingHarness.compare_pruning_locations()`** -- no longer wired into
  the master pipeline (redundant with Stage 2's joint/vision-only
  comparison); `src/testing_harness.py` itself is left in place and still
  runnable standalone if wanted.
- **CKA, neighborhood preservation, Bfloor noise floor, top-10 depth
  breakdown** -- cut from the metrics computed per pruning level (see
  above; still implemented in `src/metrics.py`, just unused).
  `src/baseline_conditions.py` (Bfloor's implementation) was deleted
  outright since nothing else used it.
- **`plot_hierarchical_breakdown_suite`, `plot_signal_noise_distribution_
  shift`, `plot_concept_retrieval_heatmap`** -- deleted from
  `src/generate_analysis_plots.py`; only `plot_category_breakdown_suite`
  (the cross-category plot) remains.

## Known limitations

- The curated subset is deliberately small and hand-picked for spread and
  recognizability, not randomly sampled -- results describe these 10
  classes specifically, not a statistically representative sample of the
  full 77-class taxonomy. The full-taxonomy scripts/data (`data/taxonomy_
  full.json`, the other 67 downloadable classes) are still present if a
  broader run is wanted later; nothing about this redesign prevents
  scaling back up, it just isn't the default anymore.
- 2.5%-to-70% pruning schedule does not cover the 70-90% "floor" region at
  all -- if a result needs to characterize full collapse rather than the
  transition zone, that range needs to be added back explicitly.
APPLY_EOF_README_MD

echo "[+] Done. Verifying syntax..."
for f in "src/curated_config.py" "src/generate_tsne_curated.py" "run_tsne.py" "src/joint_evaluator.py" "src/bozeat_experiment.py" "run_bozeat_grid.py" "run_pipeline.py" "run_full_pipeline.py" "src/generate_analysis_plots.py"; do
  python3 -m py_compile "$f" && echo "  OK: $f"
done
echo "[+] All files updated and syntax-checked successfully."
