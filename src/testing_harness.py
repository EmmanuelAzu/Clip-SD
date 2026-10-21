import os
import sys
import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pruning_engine import CLIPPruningEngine


class TestingHarness:

    def __init__(
        self, metadata_path=None, index_tensor_path=None, image_dir=None
    ):
        """Initializes the multimodal Semantic Dementia evaluation harness aligned with the 4-tier taxonomy."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if metadata_path is None:
            # NOTE: previously defaulted to "data/metadata_processed.csv" and
            # "data/image_index.pt", but run_indexing.py -- the only script
            # that actually produces these files -- writes them to
            # "data/processed/", so this class raised FileNotFoundError out
            # of the box. Fixed to match the real output location.
            metadata_path = os.path.join(
                PROJECT_ROOT, "data", "processed", "metadata_processed.csv"
            )
        if index_tensor_path is None:
            index_tensor_path = os.path.join(
                PROJECT_ROOT, "data", "processed", "image_index.pt"
            )
        if image_dir is None:
            image_dir = os.path.join(PROJECT_ROOT, "data", "images")

        self.metadata_path = metadata_path
        self.index_tensor_path = index_tensor_path
        self.image_dir = image_dir

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"[!] Metadata file not found at {self.metadata_path}. Run build_general_dataset.py first."
            )

        self.metadata = pd.read_csv(self.metadata_path)
        self._validate_metadata_scope(self.metadata)
        self.valid_metadata = self.metadata.copy()

        if os.path.exists(self.index_tensor_path):
            self.baseline_visual_memory = torch.load(
                self.index_tensor_path, map_location=self.device
            )
        else:
            self.baseline_visual_memory = None

        self.base_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device
        )
        self.pruning_engine = CLIPPruningEngine(self.base_model)

        # Single word prompt template
        self.prompt_templates = ["{}"]

        # Tiered taxonomic distance penalty map
        self.distance_costs = {
            "Correct": 0.0,
            "Coordinate Error": 1.0,
            "Superordinate Error": 2.0,
            "Domain Error": 3.0,
            "Domain Collapse": 4.0,
        }

    def _validate_metadata_scope(self, df: pd.DataFrame) -> None:
        """Fails loudly on incomplete taxonomy rows.

        NOTE: this previously also hard-failed on any non-"Living" domain
        row (Directive 1.1's living-only scope restriction). That
        restriction has been lifted: the taxonomy now deliberately spans
        both Living and Non-Living domains (data/taxonomy_full.json) so
        that Domain-tier errors ("Domain Error"/"Domain Collapse") are
        actually reachable/testable, matching the proposal's 4-tier design.
        A null taxonomy field is still rejected, since it would otherwise
        silently fall through every equality check in classify_error() and
        get mis-classified as "Domain Collapse" regardless of the model's
        actual prediction.
        """
        required_cols = [c for c in ["domain", "superordinate", "coordinate", "specific"] if c in df.columns]
        if df[required_cols].isnull().any().any():
            null_cols = [c for c in required_cols if df[c].isnull().any()]
            raise ValueError(
                f"[!] Metadata contains null taxonomy values in column(s): {null_cols}. "
                "A missing coordinate/superordinate/domain would otherwise be silently "
                "mis-classified as a coarser-tier error by classify_error()."
            )

    def classify_error(
        self, target_row: dict, predicted_row: pd.Series
    ) -> str:
        """Determines retrieval error tier across the 4-tier taxonomy.

        Assumes the taxonomy has already been validated as complete (no
        nulls) by _validate_metadata_scope() at load time, so equality
        comparisons here can be trusted directly.
        """
        if target_row["specific"] == predicted_row["specific"]:
            return "Correct"
        elif target_row["coordinate"] == predicted_row["coordinate"]:
            return "Coordinate Error"
        elif target_row["superordinate"] == predicted_row["superordinate"]:
            return "Superordinate Error"
        elif target_row["domain"] == predicted_row["domain"]:
            return "Domain Error"
        else:
            return "Domain Collapse"

    def _reindex_visual_memory(self, model, batch_size=256):
        """Encodes dataset images into memory under the current healthy or atrophied model state."""
        batch_images = []
        memory_chunks = []
        valid_rows = []

        for idx, row in self.metadata.iterrows():
            filename = row.get("filename", "")
            img_path = os.path.join(self.image_dir, filename)

            if (
                not os.path.exists(img_path)
                and "filepath" in row
                and pd.notna(row["filepath"])
            ):
                img_path = str(row["filepath"])

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(self.preprocess(img))
                    valid_rows.append(idx)
                except Exception as e:
                    print(f"[!] Warning: Could not read image {img_path}: {e}")

            if (
                len(batch_images) == batch_size
                or idx == len(self.metadata) - 1
            ):
                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    with torch.no_grad():
                        feats = model.encode_image(batch_tensor)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        memory_chunks.append(feats)
                    batch_images = []

        if memory_chunks:
            self.valid_metadata = self.metadata.iloc[valid_rows].reset_index(
                drop=True
            )
            return torch.cat(memory_chunks, dim=0)
        else:
            raise FileNotFoundError(
                f"[!] No valid image files found in '{self.image_dir}'. Check dataset setup."
            )

    def run_simulation(
        self,
        target_area="full",
        modality_mode="joint",
        max_pruning=0.90,
        step=0.05,
        batch_size=256,
        top_k=10,
    ):
        """Runs progressive atrophy simulation from 0% to 90% at 5% increments.

        Args:
            target_area: "projection" (joint space) or "full" (backbone
              encoders).
            modality_mode: "joint", "vision_only", or "text_only".
            max_pruning: Maximum pruning ratio (0.90).
            step: Step increment (0.05).
            batch_size: Processing batch size.
            top_k: Top-K retrieval limit (default 10).
        """
        print(
            f"\n[*] Running Simulation | Area: {target_area.upper()} | Mode: {modality_mode.upper()} | Top-K: {top_k}"
        )

        test_queries = self.metadata.drop_duplicates(
            subset=["specific"]
        ).to_dict("records")
        num_queries = len(test_queries)

        # 5% increments from 0.00 to max_pruning (0.90)
        steps_count = int(round(max_pruning / step)) + 1
        pruning_levels = [round(i * step, 2) for i in range(steps_count)]

        results = []

        for p in tqdm(pruning_levels, desc=f"Atrophy Loop ({modality_mode})"):
            if modality_mode == "joint":
                amount_spec = p
            elif modality_mode == "vision_only":
                amount_spec = {"text": 0.0, "vision": p}
            elif modality_mode == "text_only":
                amount_spec = {"text": p, "vision": 0.0}
            else:
                amount_spec = p

            atrophied_model = self.pruning_engine.get_pruned_model(
                amount=amount_spec,
                encoder_type="joint",
                pruning_method="random_unstructured",
                target_area=target_area,
            )
            atrophied_model.eval()

            vision_is_pruned = modality_mode in ["joint", "vision_only"]
            if vision_is_pruned or self.baseline_visual_memory is None:
                current_visual_memory = self._reindex_visual_memory(
                    atrophied_model, batch_size=batch_size
                )
            else:
                current_visual_memory = self.baseline_visual_memory

            # Single word prompt evaluation query construction
            all_prompts = [q["specific"].lower() for q in test_queries]
            all_tokens = clip.tokenize(all_prompts).to(self.device)

            encoded_text_chunks = []
            with torch.no_grad():
                for i in range(0, len(all_tokens), batch_size):
                    batch_tokens = all_tokens[i : i + batch_size]
                    feats = atrophied_model.encode_text(batch_tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    encoded_text_chunks.append(feats)

            text_vectors = torch.cat(encoded_text_chunks, dim=0)

            num_images = current_visual_memory.shape[0]
            k_adj = min(top_k, num_images)

            with torch.no_grad():
                sim_matrix = (
                    100.0 * text_vectors @ current_visual_memory.T
                ).softmax(dim=-1)
                topk_indices = (
                    torch.topk(sim_matrix, k=k_adj, dim=-1).indices.cpu().numpy()
                )

            top1_counts = {
                "Correct": 0,
                "Coordinate Error": 0,
                "Superordinate Error": 0,
                "Domain Error": 0,
                "Domain Collapse": 0,
            }
            total_expected_taxonomic_cost = 0.0

            for q_idx, query_item in enumerate(test_queries):
                query_topk = topk_indices[q_idx]

                top1_row = self.valid_metadata.iloc[query_topk[0]]
                top1_err = self.classify_error(query_item, top1_row)
                top1_counts[top1_err] += 1

                q_cost = 0.0
                for n_idx in query_topk:
                    neighbor_row = self.valid_metadata.iloc[n_idx]
                    err = self.classify_error(query_item, neighbor_row)
                    q_cost += self.distance_costs[err]

                total_expected_taxonomic_cost += q_cost / k_adj

            mean_taxonomic_cost = total_expected_taxonomic_cost / num_queries

            results.append(
                {
                    "Pruning_Level": p,
                    "Target_Area": target_area,
                    "Modality_Mode": modality_mode,
                    **top1_counts,
                    "Expected_Taxonomic_Cost": round(mean_taxonomic_cost, 4),
                }
            )

        return pd.DataFrame(results)

    def compare_pruning_locations(
        self, max_pruning=0.90, step=0.05, top_k=10
    ):
        """Runs comparative simulations across joint space, vision-only, text-only, and full backbone decay."""
        scenarios = [
            ("projection", "joint", "Joint Space Projection (Primary)"),
            ("projection", "vision_only", "Vision Projection Only"),
            ("projection", "text_only", "Text Projection Only"),
            ("full", "joint", "Full Backbone Encoders"),
        ]

        all_results = []
        for target_area, mode, label in scenarios:
            print(
                f"\n================ Running Scenario: {label} ================"
            )
            df = self.run_simulation(
                target_area=target_area,
                modality_mode=mode,
                max_pruning=max_pruning,
                step=step,
                top_k=top_k,
            )
            all_results.append(df)

        comparison_df = pd.concat(all_results, ignore_index=True)
        return comparison_df


if __name__ == "__main__":
    harness = TestingHarness()

    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Primary Joint Hub Simulation (5% increments, Top-10)
    # target_area="full" so this headline result uses the same definition of
    # "pruning level p" (whole encoder backbone) as run_pipeline.py's
    # run_eval() and every plotting helper -- projection-only pruning is
    # still available as one explicit arm of compare_pruning_locations()
    # below, where it is a deliberate ablation rather than the headline run.
    df_joint = harness.run_simulation(
        target_area="full",
        modality_mode="joint",
        max_pruning=0.90,
        step=0.05,
        top_k=10,
    )
    joint_csv = os.path.join(results_dir, "joint_hub_sd_simulation.csv")
    df_joint.to_csv(joint_csv, index=False)
    print(f"\n[+] Saved primary joint hub simulation to {joint_csv}")

    # 2. Multi-Location Comparative Analysis
    df_comparison = harness.compare_pruning_locations(
        max_pruning=0.90, step=0.05, top_k=10
    )
    comp_csv = os.path.join(results_dir, "location_comparison_simulation.csv")
    df_comparison.to_csv(comp_csv, index=False)
    print(
        f"[+] Saved comparative pruning location simulation to {comp_csv}"
    )