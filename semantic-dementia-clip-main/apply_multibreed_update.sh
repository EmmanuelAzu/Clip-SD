#!/usr/bin/env bash
# Applies the multi-breed-taxonomy / hierarchy-graph update to the repo.
# Run this from the repo root on the cluster: bash apply_multibreed_update.sh
set -euo pipefail

echo "[*] Writing new/updated files..."
mkdir -p src

cat > "src/curated_config.py" << 'APPLY_EOF_SRC_CURATED_CONFIG_PY'
"""Single source of truth for the curated evaluation subset, color scheme,
and pruning schedule, shared across every script in the pipeline.

REDESIGNED from the previous version, which picked exactly ONE specific
breed per coordinate group. That design could not test the project's
central hypothesis at all: with only one breed per group, "coordinate
error" (confusing two breeds within the same group, e.g. German Shepherd
vs Golden Retriever) was structurally impossible -- every wrong answer had
to jump straight to a different species. The whole point of this research
(RQ1: fine-grained distinctions collapse before broad categories) requires
multiple breeds/species per coordinate group so that within-group collapse
is observable at all, and observable as happening BEFORE between-group
collapse.

CURATED_TAXONOMY: 9 coordinate groups, each with 2-6 specific
breeds/species (33 specific classes total), weighted heavily toward
Animals/Plants (94%) with one small Non-Living group (Kitchen Objects) for
contrast. Every entry verified against data/taxonomy_full.json and
confirmed already downloaded (excludes the 5 classes -- Hen, Turtle,
Daisy, Cucumber, Lawnmower -- that don't exist in Tiny-ImageNet-200).
"""

import numpy as np

CURATED_TAXONOMY: dict[str, list[str]] = {
    "Domestic Dogs": [
        "German Shepherd", "Golden Retriever", "Labrador Retriever",
        "Chihuahua", "Standard Poodle", "Yorkshire Terrier",
    ],
    "Felines": ["Tabby Cat", "Egyptian Cat", "Persian Cat", "Cougar", "Lion"],
    "Insects": ["Monarch Butterfly", "Sulphur Butterfly", "Ladybug", "Dragonfly", "Mantis"],
    "Birds": ["Albatross", "Goose", "Black Stork", "King Penguin"],
    "Primates": ["Chimpanzee", "Orangutan", "Baboon"],
    "Amphibians": ["Bullfrog", "Tailed Frog", "European Fire Salamander"],
    "Citrus": ["Lemon", "Orange"],
    "Fruits": ["Banana", "Pomegranate", "Acorn"],
    "Kitchen Objects": ["Frying Pan", "Teapot"],
}

# Flat list, for code that just needs "is this class in scope" rather than
# the group structure (e.g. metadata filtering).
CURATED_CLASSES: list[str] = [c for group in CURATED_TAXONOMY.values() for c in group]

# Fixed 5 images per specific class, regardless of how many are actually
# available (some classes have 100-200 downloaded) -- this is intentional:
# the point is to study collapse ORDER (within-group before between-group)
# on a fast, small, evenly-weighted sample, not to maximize statistical
# power per class.
IMAGES_PER_CLASS = 5

# One base sequential colormap per coordinate group ("hue family"), with
# each specific breed/species within that group assigned a distinct shade
# from the family via COORDINATE_COLOR_SHADES below. This makes visual
# proximity in a plot legend map onto taxonomic proximity: two shades of
# blue are both dogs; blue vs. orange are different coordinate groups
# entirely. Families chosen to be maximally distinguishable from each
# other at a glance.
COORDINATE_CMAP_FAMILIES: dict[str, str] = {
    "Domestic Dogs": "Blues",
    "Felines": "Oranges",
    "Insects": "RdPu",
    "Birds": "Greens",
    "Primates": "Purples",
    "Amphibians": "YlOrBr",
    "Citrus": "YlGn",
    "Fruits": "PuRd",
    "Kitchen Objects": "Greys",
}


def build_class_colors() -> dict[str, tuple]:
    """Returns {specific_class_name: RGBA color}, shaded within each
    coordinate group's hue family. Shades are sampled from 0.45-0.95 of
    the colormap (skipping the near-white low end, which is hard to see
    against a white plot background).
    """
    import matplotlib.pyplot as plt

    colors = {}
    for group, members in CURATED_TAXONOMY.items():
        cmap = plt.get_cmap(COORDINATE_CMAP_FAMILIES[group])
        n = len(members)
        shades = np.linspace(0.45, 0.95, n) if n > 1 else [0.75]
        for member, shade in zip(members, shades):
            colors[member] = cmap(shade)
    return colors


# 2.5% increments, 0% to 75% inclusive (31 levels) -- extended from the
# previous 0-70% range per explicit request to include the full trajectory
# up to 75%.
PRUNING_LEVELS_FOCUSED = [round(x, 3) for x in np.arange(0.00, 0.751, 0.025).tolist()]
APPLY_EOF_SRC_CURATED_CONFIG_PY

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
        fixed_samples_per_class=None,
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
            if fixed_samples_per_class is not None:
                # Fixed cap per class regardless of how many are actually
                # available (e.g. 5 images/breed even if 100+ exist) --
                # used by the curated multi-breed-per-group design so
                # every specific class contributes equally, rather than
                # auto-balancing to whatever the smallest available class
                # happens to have.
                samples_per_class = fixed_samples_per_class
            elif target_n is not None and target_n > 0:
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

            # Top-5 / Top-10 accuracy: is the true class within the K
            # highest-similarity candidates, not just the single best?
            n_candidates = sim_matrix.shape[1]
            topk_preds = torch.argsort(sim_matrix, dim=1, descending=True)
            top5_k = min(5, n_candidates)
            top10_k = min(10, n_candidates)
            top5_correct = (topk_preds[:, :top5_k] == target_indices.unsqueeze(1)).any(dim=1)
            top10_correct = (topk_preds[:, :top10_k] == target_indices.unsqueeze(1)).any(dim=1)
            top5_acc = top5_correct.float().mean().item()
            top10_acc = top10_correct.float().mean().item()

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
                "top5_specific_acc": top5_acc,
                "top10_specific_acc": top10_acc,
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


def plot_accuracy_curve(df, output_dir=None):
    """Top-1/Top-5/Top-10 specific accuracy, all three on one plot, vs.
    pruning level. Split out from the old combined
    plot_category_breakdown_suite (which crammed an accuracy line and an
    error-taxonomy heatmap into one untidy two-panel figure) into its own
    standalone plot per explicit request.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    series_specs = [
        ("top1_specific_acc", "Top-1 Accuracy", "#1f77b4", "o"),
        ("top5_specific_acc", "Top-5 Accuracy", "#ff7f0e", "s"),
        ("top10_specific_acc", "Top-10 Accuracy", "#2ca02c", "^"),
    ]
    plotted_any = False
    for col, label, color, marker in series_specs:
        if col in df.columns:
            ax.plot(
                prune_pcts, df[col], marker=marker, markersize=4,
                color=color, linewidth=2.2, label=label,
            )
            plotted_any = True

    if not plotted_any and "i2t_top1" in df.columns:
        ax.plot(prune_pcts, df["i2t_top1"], marker="o", color="#1f77b4",
                 linewidth=2.5, label="Top-1 Accuracy")

    ax.set_xlabel("Pruning Level (%)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Top-1 / Top-5 / Top-10 Accuracy vs. Pruning Level", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved accuracy curve to: {save_path}")
    return save_path


def plot_error_taxonomy_heatmap(df, output_dir=None):
    """The clinical error-taxonomy heatmap (Coordinate/Superordinate/
    Domain Error/Domain Collapse), on its own, standalone -- split out of
    the old combined plot per explicit request. Proportions, not raw
    counts (bounded 0-1, comparable across runs with different dataset
    sizes); annotation labels shown only when the grid is narrow enough
    to stay readable.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    prune_pcts = [p * 100 for p in df[p_col]]

    acc_col = next((c for c in ["top1_specific_acc", "i2t_top1"] if c in df.columns), None)
    acc_series = df[acc_col] if acc_col else None

    err_cols = [
        c for c in ["coordinate error", "superordinate error", "domain error", "domain collapse"]
        if c in df.columns
    ]
    if not err_cols:
        print("[!] No error-taxonomy columns found -- skipping heatmap.")
        return None

    err_sum = df[err_cols].sum(axis=1)
    if acc_series is not None:
        total_samples = err_sum / (1 - acc_series).replace(0, np.nan)
    else:
        total_samples = err_sum.replace(0, np.nan)
    heatmap_data = (df[err_cols].div(total_samples, axis=0)).T
    heatmap_data.columns = [f"{p:.1f}%" for p in prune_pcts]
    heatmap_data.index = [c.replace("_", " ").title() for c in err_cols]

    n_cols = len(heatmap_data.columns)
    show_annot = n_cols <= 20
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.4), 5), dpi=200)
    sns.heatmap(
        heatmap_data,
        annot=show_annot,
        fmt=".2f",
        annot_kws={"size": 7} if show_annot else None,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        ax=ax,
        cbar=True,
        cbar_kws={"label": "Proportion of outcomes"},
    )
    ax.set_title("Clinical Error Taxonomy (proportion of outcomes)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pruning Level (%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=90)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_taxonomy_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved error taxonomy heatmap to: {save_path}")
    return save_path
APPLY_EOF_SRC_GENERATE_ANALYSIS_PLOTS_PY

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
    display_prompts: list[str] | None = None,
    display_pruning_levels: list[float] | None = None,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Runs Bozeat-style retrieval for `target_prompts` at every level in
    `pruning_levels`, using `harness.valid_metadata` as BOTH the source of
    query prompts and the retrieval candidate pool -- pass an evaluator
    already restricted to the curated subset (JointSpaceEvaluator(
    restrict_classes=CURATED_CLASSES)) for this to be fast and meaningful.

    display_prompts: the full retrieval CURVE covers every prompt in
    target_prompts (cheap regardless of count, since the candidate pool is
    small), but the qualitative image GRID needs a small row count to stay
    readable -- pass a representative subset here (e.g. one breed per
    coordinate group) to control the grid's size independently of how many
    prompts are actually being evaluated. Defaults to all of
    target_prompts if not given.

    Returns (csv_path, image_grid_path).
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")
    os.makedirs(output_dir, exist_ok=True)

    if display_prompts is None:
        display_prompts = target_prompts

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
    grid_path = _plot_image_grid(results_df, display_prompts, display_pruning_levels, output_dir)

    return csv_path, grid_path


def _plot_retrieval_curve(results_df: pd.DataFrame, target_prompts: list[str], output_dir: str) -> str:
    """Full-resolution (every pruning level) per-class retrieval accuracy
    and mean confidence, plus the aggregate across all classes. Colors
    use the same coordinate-group hue-family scheme as the tSNE plot
    (src/curated_config.py::build_class_colors) so a class's color is
    consistent across every figure in the pipeline.
    """
    from src.curated_config import build_class_colors
    class_colors = build_class_colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

    agg_acc = results_df.groupby("pruning_level")["correct"].mean()
    agg_conf = results_df.groupby("pruning_level")["top1_similarity"].mean()

    for prompt in target_prompts:
        color = class_colors.get(prompt, "#888888")
        sub = results_df[results_df["prompt"] == prompt].sort_values("pruning_level")
        ax1.plot(sub["pruning_level"] * 100, sub["correct"].astype(float),
                  color=color, alpha=0.6, linewidth=1.1)
        ax2.plot(sub["pruning_level"] * 100, sub["top1_similarity"],
                  color=color, alpha=0.6, linewidth=1.1, label=prompt)

    ax1.plot(agg_acc.index * 100, agg_acc.values, color="black", linewidth=3, label="Mean (all classes)")
    ax2.plot(agg_conf.index * 100, agg_conf.values, color="black", linewidth=3, linestyle="--")

    ax1.set_xlabel("Pruning Level (%)"); ax1.set_ylabel("Retrieval Correct (1/0)")
    ax1.set_title("Bozeat Retrieval Accuracy per Class", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Pruning Level (%)"); ax2.set_ylabel("Top-1 Cosine Similarity")
    ax2.set_title("Retrieval Confidence per Class", fontsize=12, fontweight="bold")
    # Always include the legend (per explicit request), even with many
    # classes -- small font, multi-column, so it stays usable rather than
    # dropped for tidiness.
    ax2.legend(fontsize=5.5, ncol=3, loc="upper right"); ax2.grid(alpha=0.3)

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
from src.curated_config import (
    CURATED_CLASSES,
    CURATED_TAXONOMY,
    PRUNING_LEVELS_FOCUSED,
    IMAGES_PER_CLASS,
)

# The full retrieval CURVE covers all 33 curated classes (cheap regardless
# of count, since the candidate pool is small either way). The qualitative
# image GRID needs a small row count to stay readable, so it uses one
# representative breed/species per coordinate group (9 rows) by default.
DEFAULT_DISPLAY_PROMPTS = [members[0] for members in CURATED_TAXONOMY.values()]


def main(output_dir=None, display_prompts=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", "bozeat_experiment")
    if display_prompts is None:
        display_prompts = DEFAULT_DISPLAY_PROMPTS

    print("=" * 60)
    print(" BOZEAT TEXT->IMAGE RETRIEVAL EXPERIMENT (curated multi-breed subset) ")
    print("=" * 60)
    print(f"[*] Full prompt set ({len(CURATED_CLASSES)} classes, {len(CURATED_TAXONOMY)} coordinate groups)")
    print(f"[*] Image grid display prompts (1/group): {display_prompts}")
    print(f"[*] Pruning Grid ({len(PRUNING_LEVELS_FOCUSED)} stages, 2.5% steps, 0-75%)")

    # restrict_classes here makes BOTH the query prompts and the retrieval
    # candidate pool the curated multi-breed subset -- this is what makes
    # the experiment fast. fixed_samples_per_class=5 keeps every specific
    # class equally weighted regardless of how many images are actually
    # available, per the within-group-collapse study design.
    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    csv_path, grid_path = run_bozeat_experiment(
        evaluator,
        target_prompts=CURATED_CLASSES,
        pruning_levels=PRUNING_LEVELS_FOCUSED,
        display_prompts=display_prompts,
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

cat > "src/generate_tsne_curated.py" << 'APPLY_EOF_SRC_GENERATE_TSNE_CURATED_PY'
"""The one tSNE visualization for this project.

Shows EVERY pruning level in the schedule (not a subsampled set), colored
by coordinate-group hue family with a distinct shade per specific
breed/species within that group (src/curated_config.py::build_class_colors)
-- so a viewer can tell at a glance which points belong to the same
species (near-identical shade), the same broader group (same hue family),
or a different group entirely (different hue), without needing to
cross-reference a class-by-class legend.
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

from src.curated_config import CURATED_TAXONOMY, CURATED_CLASSES, build_class_colors


def generate_curated_tsne_grid(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    output_dir: str | None = None,
    n_cols: int = 6,
) -> str:
    """Generates one tSNE grid: one panel per pruning level (default:
    every level in the schedule), points colored by specific class within
    a coordinate-group hue family, fit globally across all stacked stages
    so positions are directly comparable panel to panel. Legend is always
    included, per explicit request, organized by coordinate group.
    """
    if pruning_levels_to_show is None:
        from src.curated_config import PRUNING_LEVELS_FOCUSED
        pruning_levels_to_show = PRUNING_LEVELS_FOCUSED
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    class_colors = build_class_colors()
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
    n_cols = min(n_cols, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows))
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
                pts[:, 0], pts[:, 1], s=22, color=class_colors[cls],
                alpha=0.85, edgecolor="white", linewidth=0.3,
            )

        ax.set_title(f"{p_level * 100:.1f}%", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    for ax in axes_flat[n_stages:]:
        ax.axis("off")

    # Legend organized by coordinate group -- shows the hue-family
    # structure directly (one legend "row" per group, shades within it),
    # rather than a flat 33-entry alphabetical list.
    handles = []
    labels_list = []
    for group, members in CURATED_TAXONOMY.items():
        handles.append(mpatches.Patch(color="white", alpha=0))  # spacer/header
        labels_list.append(f"— {group} —")
        for m in members:
            handles.append(mpatches.Patch(color=class_colors[m]))
            labels_list.append(m)

    fig.legend(
        handles, labels_list, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=7, frameon=False, ncol=1, title="Coordinate Group / Species",
        title_fontsize=8,
    )

    fig.suptitle(
        f"Joint Embedding Space t-SNE Across All Pruning Levels ({scenario} scenario)\n"
        "Color = coordinate group (hue family) + specific breed/species (shade)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
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
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS


def main(output_dir=None, scenario="joint"):
    print("=" * 60)
    print(" CURATED t-SNE GRID (multi-breed subset, all pruning levels) ")
    print("=" * 60)
    print(f"[*] Classes ({len(CURATED_CLASSES)}): {CURATED_CLASSES}")
    print(f"[*] Pruning levels: {len(PRUNING_LEVELS_FOCUSED)} (0-75%, 2.5% steps)")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

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
from src.generate_analysis_plots import plot_accuracy_curve, plot_error_taxonomy_heatmap
from src.curated_config import CURATED_CLASSES, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS


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
        fixed_samples_per_class=IMAGES_PER_CLASS,
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

    # THE cross-category plots: (1) Top-1/5/10 accuracy vs pruning, and
    # (2) the 4-tier clinical error-taxonomy heatmap -- now two separate,
    # clean figures rather than one crowded two-panel plot.
    plot_accuracy_curve(joint_df, output_dir=output_dir)
    plot_error_taxonomy_heatmap(joint_df, output_dir=output_dir)

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

cat > "src/generate_hierarchy_graph.py" << 'APPLY_EOF_SRC_GENERATE_HIERARCHY_GRAPH_PY'
"""Hierarchical dendrogram of the embedding space, at a few representative
pruning levels, to directly visualize the SHAPE of collapse -- distinct
from the t-SNE (geometric layout) and the cross-category plot (aggregate
error rates). This answers "do all dog breeds merge with each other
before any of them merges with a cat, and does the whole Felines group
then merge with Primates before Insects?" directly, as a tree, rather
than requiring that structure to be inferred from scatter positions or
error-rate numbers.

Method: at each selected pruning level, compute the per-class mean
embedding (centroid) across that class's images, then run agglomerative
hierarchical clustering (scipy) on the 33 centroids. Leaf labels are
colored by true coordinate group (same hue-family scheme as the tSNE and
Bozeat plots) so a reader can see at a glance whether early merges happen
WITHIN a color family (expected, healthy hierarchical structure) or
ACROSS color families (evidence the model has already lost the
coordinate-level boundary).
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.curated_config import CURATED_TAXONOMY, CURATED_CLASSES, build_class_colors


def _class_to_group(cls: str) -> str:
    for group, members in CURATED_TAXONOMY.items():
        if cls in members:
            return group
    return "Unknown"


def generate_hierarchy_dendrogram(
    evaluator,
    pruning_levels_to_show: list[float] | None = None,
    scenario: str = "joint",
    linkage_method: str = "average",
    output_dir: str | None = None,
) -> str:
    if pruning_levels_to_show is None:
        pruning_levels_to_show = [0.0, 0.25, 0.50, 0.75]
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    class_colors = build_class_colors()

    n_stages = len(pruning_levels_to_show)
    fig, axes = plt.subplots(1, n_stages, figsize=(6.5 * n_stages, 9), dpi=200)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, p_level in zip(axes_flat, pruning_levels_to_show):
        print(f"[*] Hierarchy graph: extracting features at {p_level * 100:.1f}% pruning...")
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

        # Per-class centroid embeddings
        centroids = []
        centroid_labels = []
        for cls in CURATED_CLASSES:
            mask = [lbl == cls for lbl in labels]
            if not any(mask):
                continue
            centroids.append(img_feats_np[mask].mean(axis=0))
            centroid_labels.append(cls)
        centroids = np.vstack(centroids)

        # Cosine distance is the natural choice here since the embeddings
        # are L2-normalized and retrieval itself is cosine-similarity
        # based -- using it for clustering keeps this consistent with
        # what the rest of the pipeline actually measures.
        dist = pdist(centroids, metric="cosine")
        Z = linkage(dist, method=linkage_method)

        dendrogram(
            Z, labels=centroid_labels, ax=ax, orientation="left",
            leaf_font_size=8, color_threshold=0,
            above_threshold_color="#888888",
        )
        ax.set_title(f"{p_level * 100:.1f}% Pruned", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cosine Distance (linkage)", fontsize=9)

        # Recolor leaf tick labels by true coordinate group
        for tick_label in ax.get_ymajorticklabels():
            cls_name = tick_label.get_text()
            if cls_name in class_colors:
                tick_label.set_color(class_colors[cls_name])
                tick_label.set_fontweight("bold")

    fig.suptitle(
        "Hierarchical Clustering of Class Centroids Across Pruning Levels\n"
        "(leaf label color = true coordinate group -- early within-color merges = "
        "healthy hierarchy retained; early cross-color merges = structure already lost)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "hierarchy_dendrogram.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Hierarchy dendrogram saved to: {save_path}")
    return save_path
APPLY_EOF_SRC_GENERATE_HIERARCHY_GRAPH_PY

cat > "run_hierarchy_graph.py" << 'APPLY_EOF_RUN_HIERARCHY_GRAPH_PY'
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.generate_hierarchy_graph import generate_hierarchy_dendrogram
from src.curated_config import CURATED_CLASSES, IMAGES_PER_CLASS


def main(output_dir=None, pruning_levels=None, scenario="joint"):
    print("=" * 60)
    print(" HIERARCHICAL DENDROGRAM (class-centroid clustering) ")
    print("=" * 60)

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")
    if pruning_levels is None:
        pruning_levels = [0.0, 0.25, 0.50, 0.75]

    print(f"[*] Pruning levels shown: {pruning_levels}")

    evaluator = JointSpaceEvaluator(
        restrict_classes=CURATED_CLASSES,
        balance_taxonomically=True,
        fixed_samples_per_class=IMAGES_PER_CLASS,
    )

    save_path = generate_hierarchy_dendrogram(
        evaluator, pruning_levels_to_show=pruning_levels, scenario=scenario, output_dir=output_dir
    )
    print(f"\n[+] Done: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the hierarchical dendrogram visualization.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario)
APPLY_EOF_RUN_HIERARCHY_GRAPH_PY

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
    from run_hierarchy_graph import main as run_hierarchy_graph_main
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
    """Streamlined 5-stage pipeline, scoped to the curated multi-breed
    subset (src/curated_config.py) throughout:

      1. Visual memory indexing (once, over the full downloaded dataset --
         cheap and only needs re-running when new images are added).
      2. Core evaluation: joint vs. vision-only scenario comparison,
         Top-1/5/10 accuracy curve, the clinical error-taxonomy heatmap
         (two separate plots), 2.5% pruning increments from 0-75%.
      3. Bozeat text->image retrieval experiment.
      4. One consolidated t-SNE grid (every pruning level, hue-family
         colors by coordinate group).
      5. Hierarchical dendrogram of class centroids -- shows whether
         within-group merges (e.g. dog breeds with each other) happen
         before between-group merges (dogs with cats), directly testing
         the project's central hypothesis as a tree structure.

    See src/curated_config.py for the full design rationale (33 specific
    breeds/species across 9 coordinate groups, 5 images/class, weighted
    toward Animals/Plants).
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
        print("\n[STAGE 1/5] Building Visual Memory Index...")
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
        print("\n[STAGE 1/5] Skipping Visual Memory Indexing (--skip_indexing set).")

    stage_failures = []

    # -------------------------------------------------------------------------
    # STAGE 2: Core Evaluation (joint vs. vision-only, cross-category plots)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/5] Running Core Evaluation (curated multi-breed subset)...")
    try:
        csv_metrics = run_joint_pipeline(output_dir=results_dir)
        print(f"[+] Stage 2 Complete. Metrics exported to: {csv_metrics}")
    except Exception as err:
        print(f"[!] Stage 2 Failed: {err}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: Bozeat Text->Image Retrieval Experiment
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/5] Running Bozeat Retrieval Experiment...")
    try:
        run_bozeat_grid_main(output_dir=dirs["results_bozeat"])
        print("[+] Stage 3 Complete.")
    except Exception as err:
        print(f"[!] Stage 3 Failed: {err}")
        stage_failures.append(("Stage 3 (Bozeat Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 4: Curated t-SNE Grid
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/5] Generating Curated t-SNE Grid...")
    try:
        run_tsne_main(output_dir=results_dir)
        print("[+] Stage 4 Complete.")
    except Exception as err:
        print(f"[!] Stage 4 Failed: {err}")
        stage_failures.append(("Stage 4 (t-SNE Grid)", err))

    # -------------------------------------------------------------------------
    # STAGE 5: Hierarchical Dendrogram
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/5] Generating Hierarchical Dendrogram...")
    try:
        run_hierarchy_graph_main(output_dir=results_dir)
        print("[+] Stage 5 Complete.")
    except Exception as err:
        print(f"[!] Stage 5 Failed: {err}")
        stage_failures.append(("Stage 5 (Hierarchical Dendrogram)", err))

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

echo "[+] Done. Verifying syntax..."
for f in "src/curated_config.py" "src/joint_evaluator.py" "src/generate_analysis_plots.py" "src/bozeat_experiment.py" "run_bozeat_grid.py" "src/generate_tsne_curated.py" "run_tsne.py" "run_pipeline.py" "src/generate_hierarchy_graph.py" "run_hierarchy_graph.py" "run_full_pipeline.py"; do
  python3 -m py_compile "$f" && echo "  OK: $f"
done
echo "[+] All files updated and syntax-checked successfully."
