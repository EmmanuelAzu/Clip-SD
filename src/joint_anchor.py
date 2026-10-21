"""Joint (text + image) query construction.

Every retrieval mechanism in this pipeline (Bozeat, hierarchical
retrieval) has so far queried with PURE TEXT embeddings -- a single
linguistic vector per concept. That tests only the model's linguistic
pathway into the shared embedding space, not the more theoretically
central claim from Proposal Sec. 2.2: that this shared space functions as
an amodal transmodal hub, integrating modality-specific "spokes" into one
meaning representation (the ATL analogy).

This module builds an alternative query: for each concept, blend its text
embedding with the MEAN of its own images' embeddings, representing a
more "hub-like" integrated meaning rather than a purely linguistic one.
The result has the exact same shape as a pure-text query tensor, so it's
a drop-in replacement anywhere text_feats is currently used as the query
source -- the retrieval mechanics themselves (cosine similarity against
image candidates, top-1 argmax, routing, everything downstream) are
completely unchanged; only what vector represents "the query" changes.
"""

import torch


def build_joint_anchor_queries(
    text_feats: torch.Tensor,
    query_concepts: list[str],
    img_feats: torch.Tensor,
    img_labels: list[str],
    alpha: float = 0.5,
) -> torch.Tensor:
    """Returns a (len(query_concepts), D) tensor of L2-normalized joint
    query vectors, aligned 1:1 with query_concepts (same order, same
    length as text_feats) -- safe to substitute directly for text_feats
    in any downstream similarity computation.

    alpha: weight on the text embedding vs. the mean-image embedding.
    0.5 (default) is an equal, symmetric blend -- neither modality
    dominates. alpha=1.0 recovers pure-text (the original behavior);
    alpha=0.0 would be pure mean-image (an unlabeled visual prototype,
    not meaningfully a "query" on its own, but included for completeness
    if ever wanted).

    If a query concept has no matching images in img_feats/img_labels
    (shouldn't happen in normal use, since queries are drawn from the
    same taxonomy as the candidate pool, but guarded defensively), that
    query falls back to its pure text embedding unchanged rather than
    raising -- a silent NaN from an empty mean would be a much worse
    failure mode than a graceful fallback.
    """
    if text_feats.shape[0] != len(query_concepts):
        raise ValueError(
            f"[!] text_feats has {text_feats.shape[0]} rows but query_concepts "
            f"has {len(query_concepts)} entries -- these must be aligned 1:1."
        )

    device = text_feats.device
    anchors = []
    for i, concept in enumerate(query_concepts):
        mask = torch.tensor([l == concept for l in img_labels], device=img_feats.device)
        if mask.any():
            mean_img = img_feats[mask].mean(dim=0)
            mean_img = mean_img / mean_img.norm()
            blended = alpha * text_feats[i] + (1.0 - alpha) * mean_img.to(device)
            blended = blended / blended.norm()
        else:
            blended = text_feats[i]
        anchors.append(blended)

    return torch.stack(anchors)
