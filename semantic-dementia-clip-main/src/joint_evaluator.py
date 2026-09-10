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
