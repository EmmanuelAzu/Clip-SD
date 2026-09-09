import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def compute_entropy(sim_matrix: np.ndarray, temperature: float = 100.0) -> float:
    """Computes mean Shannon Vector Entropy H(P) over the retrieval softmax
    distribution (proposal Sec. 3.4.1).

    For each query row, the raw cosine-similarity scores are converted to a
    retrieval probability distribution via softmax (temperature-scaled, to
    match the 100x logit scaling CLIP itself uses at inference), then:

        H(P) = -sum_i p_i * log2(p_i)

    A healthy/confident model produces one dominant probability spike (low
    H); as pruning erodes discriminative power the distribution flattens
    and H rises. Returns the mean H across all query rows in sim_matrix,
    in bits (log base 2), bounded in [0, log2(N)] for N candidates.
    """
    if sim_matrix.size == 0:
        return 0.0
    scaled = temperature * sim_matrix
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(scaled)
    probs = exp / exp.sum(axis=1, keepdims=True)
    eps = 1e-12
    row_entropy = -np.sum(probs * np.log2(probs + eps), axis=1)
    return float(np.mean(row_entropy))


def compute_mrr(sim_matrix: np.ndarray, targets: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) for target retrieval."""
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    ranks = np.where(sorted_indices == targets[:, None])[1] + 1
    return float(np.mean(1.0 / ranks))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between representations X and Y."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    dot_product = np.linalg.norm(Y.T @ X, "fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, "fro")
    norm_y = np.linalg.norm(Y.T @ Y, "fro")
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return float(dot_product / (norm_x * norm_y))


def compute_neighborhood_preservation(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Computes Neighbor Preservation Ratio (NPR) between pristine and pruned representations."""
    if len(X) <= 1:
        return 0.0
    k_adj = min(k + 1, len(X))
    nbrs_X = NearestNeighbors(n_neighbors=k_adj).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    nbrs_Y = NearestNeighbors(n_neighbors=k_adj).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    intersection = [len(set(nx).intersection(set(ny))) for nx, ny in zip(nbrs_X, nbrs_Y)]
    return float(np.mean(intersection) / max(1, k_adj - 1))


def compute_category_breakdown(sim_matrix: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame) -> dict[str, float]:
    """Computes per-category Top-1 accuracy across metadata tiers."""
    cat_col = next((c for c in ["domain", "superordinate", "coordinate"] if c in metadata.columns), None)
    if not cat_col:
        return {}

    preds = np.argmax(sim_matrix, axis=1)
    correct = (preds == targets)

    meta_aligned = metadata.reset_index(drop=True)
    cat_accs = {}
    for cat, group in meta_aligned.groupby(cat_col):
        indices = group.index.values
        if len(indices) > 0 and max(indices) < len(correct):
            cat_accs[str(cat)] = float(correct[indices].mean())
    return cat_accs


def compute_hierarchical_breakdown(sim_matrix: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame) -> dict[str, float | int]:
    """Evaluates Top-1 prediction taxonomy errors across the 4-tier hierarchy.

    Aligns error classification and distance costs with TestingHarness metrics.
    """
    meta_aligned = metadata.reset_index(drop=True)

    required_cols = ["specific", "coordinate", "superordinate", "domain"]
    for col in required_cols:
        if col not in meta_aligned.columns:
            raise ValueError(f"[!] Missing required taxonomy column '{col}' in metadata.")
    if meta_aligned[required_cols].isnull().any().any():
        # A null taxonomy field (NaN != NaN in pandas/NumPy) would otherwise
        # silently fall through every equality check below and get
        # mis-classified as "Domain Collapse" regardless of the model's
        # actual prediction.
        null_cols = [c for c in required_cols if meta_aligned[c].isnull().any()]
        raise ValueError(f"[!] Metadata contains null taxonomy values in column(s): {null_cols}.")

    preds = np.argmax(sim_matrix, axis=1)
    
    # Map index predictions back to taxonomy metadata
    target_rows = meta_aligned.iloc[targets].to_dict("records")
    pred_rows = meta_aligned.iloc[preds].to_dict("records")

    counts = {
        "Correct": 0,
        "Coordinate Error": 0,
        "Superordinate Error": 0,
        "Domain Error": 0,
        "Domain Collapse": 0,
    }

    distance_costs = {
        "Correct": 0.0,
        "Coordinate Error": 1.0,
        "Superordinate Error": 2.0,
        "Domain Error": 3.0,
        "Domain Collapse": 4.0,
    }

    total_cost = 0.0
    total = len(target_rows)

    for target_row, pred_row in zip(target_rows, pred_rows):
        if target_row["specific"] == pred_row["specific"]:
            err = "Correct"
        elif target_row["coordinate"] == pred_row["coordinate"]:
            err = "Coordinate Error"
        elif target_row["superordinate"] == pred_row["superordinate"]:
            err = "Superordinate Error"
        elif target_row["domain"] == pred_row["domain"]:
            err = "Domain Error"
        else:
            err = "Domain Collapse"

        counts[err] += 1
        total_cost += distance_costs[err]

    total_errors = total - counts["Correct"]

    return {
        "top1_specific_acc": float(counts["Correct"] / total) if total > 0 else 0.0,
        "top1_coordinate_acc": float((counts["Correct"] + counts["Coordinate Error"]) / total) if total > 0 else 0.0,
        "top1_super_acc": float((counts["Correct"] + counts["Coordinate Error"] + counts["Superordinate Error"]) / total) if total > 0 else 0.0,
        "top1_domain_acc": float((total - counts["Domain Collapse"]) / total) if total > 0 else 0.0,
        "pct_superordinate_errors": float(counts["Superordinate Error"] / total_errors) if total_errors > 0 else 0.0,
        "pct_domain_collapse_errors": float(counts["Domain Collapse"] / total_errors) if total_errors > 0 else 0.0,
        "Expected_Taxonomic_Cost": round(float(total_cost / total), 4) if total > 0 else 0.0,
        **counts,
    }


def compute_typicality_delta(sim_matrix: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame) -> dict[str, float]:
    """Computes the Clinical Typicality Effect (Directive 3.4).

    Tracks Top-1 accuracy separately for Typical vs Atypical exemplars (per
    the "typicality" taxonomy column, Directive 1.4) and returns
    Delta_typicality = Acc_typical - Acc_atypical at this pruning checkpoint.
    A positive, monotonically increasing curve across pruning levels is the
    paper's named validation criterion for matching human dementia behaviour.

    Requires a "typicality" column with values "Typical"/"Atypical" on
    metadata; raises if it is missing rather than silently skipping the
    computation, since a caller relying on this to validate the hypothesis
    should know immediately if the data can't support it.
    """
    if "typicality" not in metadata.columns:
        raise ValueError(
            "[!] Metadata has no 'typicality' column -- Directive 1.4 requires every "
            "specific entity to be flagged Typical/Atypical before Delta_typicality can "
            "be computed."
        )

    meta_aligned = metadata.reset_index(drop=True)
    preds = np.argmax(sim_matrix, axis=1)
    correct = (preds == targets)

    typicality = meta_aligned["typicality"].astype(str).str.strip().str.casefold()
    typical_mask = (typicality == "typical").values
    atypical_mask = (typicality == "atypical").values

    n_typical = int(typical_mask.sum())
    n_atypical = int(atypical_mask.sum())

    acc_typical = float(correct[typical_mask].mean()) if n_typical > 0 else float("nan")
    acc_atypical = float(correct[atypical_mask].mean()) if n_atypical > 0 else float("nan")

    return {
        "n_typical": n_typical,
        "n_atypical": n_atypical,
        "acc_typical": acc_typical,
        "acc_atypical": acc_atypical,
        "delta_typicality": (
            acc_typical - acc_atypical
            if n_typical > 0 and n_atypical > 0
            else float("nan")
        ),
    }


def compute_top10_breakdown_depth(
    sim_matrix: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame, unique_concepts: list[str]
) -> dict[str, float]:
    """Evaluates taxonomy depth preservation across top-10 retrieval candidates."""
    meta_aligned = metadata.reset_index(drop=True)

    concept_to_coordinate = meta_aligned.drop_duplicates("specific").set_index("specific")["coordinate"].to_dict()
    concept_to_super = meta_aligned.drop_duplicates("specific").set_index("specific")["superordinate"].to_dict()
    concept_to_domain = meta_aligned.drop_duplicates("specific").set_index("specific")["domain"].to_dict()

    top10_k = min(10, sim_matrix.shape[1])
    top10_indices = np.argsort(-sim_matrix, axis=1)[:, :top10_k]

    same_coordinate_counts, same_super_counts, same_domain_counts = [], [], []

    for i, top_k in enumerate(top10_indices):
        target_concept = unique_concepts[targets[i]] if i < len(targets) else meta_aligned.iloc[i]["specific"]
        target_coordinate = concept_to_coordinate.get(target_concept)
        target_super = concept_to_super.get(target_concept)
        target_domain = concept_to_domain.get(target_concept)

        retrieved_concepts = [
            unique_concepts[idx] if idx < len(unique_concepts) else meta_aligned.iloc[idx]["specific"]
            for idx in top_k
        ]
        retrieved_coordinates = [concept_to_coordinate.get(c) for c in retrieved_concepts]
        retrieved_supers = [concept_to_super.get(c) for c in retrieved_concepts]
        retrieved_domains = [concept_to_domain.get(c) for c in retrieved_concepts]

        same_coordinate_counts.append(sum(1 for c in retrieved_coordinates if c == target_coordinate) / float(top10_k))
        same_super_counts.append(sum(1 for s in retrieved_supers if s == target_super) / float(top10_k))
        same_domain_counts.append(sum(1 for d in retrieved_domains if d == target_domain) / float(top10_k))

    return {
        "top10_coordinate_ratio": float(np.mean(same_coordinate_counts)),
        "top10_superordinate_ratio": float(np.mean(same_super_counts)),
        "top10_domain_ratio": float(np.mean(same_domain_counts)),
    }