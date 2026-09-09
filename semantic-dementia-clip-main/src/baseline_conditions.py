"""Implements the Random Noise Baseline (Bfloor) described in the research
proposal (Sec. 3.3.3): "The query vectors are completely randomized via
Gaussian noise injection prior to computing the matrix multiplications,
simulating a state of total, end-stage semantic collapse. This baseline
establishes the mathematical floor or the random guessing rate."

This was previously named as a required control condition but had no
implementation anywhere in the codebase -- unlike Bhealthy/Btext/Bvision
(which are all *weight*-pruning conditions), Bfloor is a *query-level*
perturbation: it never touches the model's weights at all. It answers a
different question than weight pruning does: "what does chance performance
look like on this exact dataset/taxonomy?" -- the floor every pruning curve
should be compared against.
"""

import torch

BFLOOR_SEED = 4242


def gaussian_noise_floor_features(reference_features: torch.Tensor, seed: int = BFLOOR_SEED) -> torch.Tensor:
    """Replaces query features with L2-normalized Gaussian noise of the
    same shape/device/dtype as `reference_features` (Bfloor, Sec. 3.3.3).

    This is NOT a weight-pruning strategy -- no model is touched. It
    generates a mathematical floor reference: if a query vector carries no
    semantic information whatsoever, this is the retrieval performance you
    would see by chance alone on this specific dataset/taxonomy.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        reference_features.shape, generator=generator, dtype=reference_features.dtype
    )
    noise = noise.to(reference_features.device)
    noise = noise / noise.norm(dim=-1, keepdim=True)
    return noise
