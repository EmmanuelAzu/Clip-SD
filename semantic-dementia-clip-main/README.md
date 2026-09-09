# Simulating Semantic Dementia Through Progressive Pruning of the CLIP Embedding Space

This repository contains the open-source clinical simulation framework developed for Emmanuel Azubuike's BSc Honours research at the University of the Witwatersrand.

## Abstract

Traditional machine learning treats cognitive decline as discrete, categorical steps. This project bridges neuropsychology and computational clinical modeling by simulating continuous artificial neuro-degeneration. Using the OpenAI CLIP (ViT-B/32) architecture coupled with an exact k-NN retrieval framework, we model progressive cortical atrophy through targeted weight pruning to evaluate the rate and trajectory of conceptual erosion against a strict 4-tier clinical taxonomy (Domain → Superordinate → Coordinate/Basic Level → Specific).

## Dataset

Images are drawn from **Tiny-ImageNet-200** (Stanford's CS231n mirror of a 200-class, single-object-cropped ImageNet subset) — not THINGS or BOSS. Both are well-known, purpose-built stimulus sets for this kind of work, but neither has a verified programmatic download path this codebase could safely integrate; Tiny-ImageNet is already proven working here and its images are single-object crops, satisfying the same "isolated subject, no ambient background confound" requirement (Sec. 3.3.1 / Risk 1) that motivated the original THINGS/BOSS choice.

The taxonomy (`data/taxonomy_full.json`) spans **77 classes across both Living and Non-Living domains**, grouped into 23 basic-level ("coordinate") categories — e.g. "Domestic Dogs" spans German Shepherd / Golden Retriever / Labrador Retriever / Chihuahua / Standard Poodle / Yorkshire Terrier as sibling specific classes, directly matching the "Golden Retriever vs. Siberian Husky"-style example used to motivate RQ1. Every class is also tagged `Typical`/`Atypical` within its coordinate group (e.g. King Penguin is the atypical member of "Birds" alongside the typical Albatross/Goose/Stork), which is what makes the Typicality Delta metric computable at all.

**Current on-disk coverage:** 10 of the 77 classes (5 Living, 5 Non-Living — 1,000 images) are already downloaded to `data/images/`. The remaining 67 require running `python src/dataset_downloader.py` once, on a machine with network access, against the Tiny-ImageNet-200 archive — the script is idempotent and incremental, so re-running it after a partial download only fetches what's still missing. Even with just the current 10 classes, both domains and two genuine Typical/Atypical coordinate-group contrasts (Kitchen Objects: Frying Pan vs. Teapot; Living Room Objects: Chair vs. Table) are already available.

## Pruning Methodology

`CLIPPruningEngine` supports:
- **Target area**: `"projection"` (only the final `text_projection`/`visual.proj` layers that map into the shared embedding space) or `"full"` (the entire transformer backbone — every attention and MLP weight, at every layer). **The main pipeline uses `"full"`** — pruning reaches the whole encoder, not just the joint-space projection heads.
- **Depth zone** (`"full"` only): `early`/`middle`/`deep`/`global`, for RQ2's layer-depth comparison.
- **Pruning method**: `l1_unstructured` (deterministic magnitude pruning — the "zero-weight" default), `random_unstructured` (undirected random pruning), or `structured_channel`/`structured_head` (whole-channel/whole-attention-head removal).
- **Masking mode**: `"static"` (default — every pruning level is computed fresh from the pristine model) or `"iterative"` (damage compounds on top of the previous level). For magnitude/norm-ranked methods (`l1_unstructured`, `structured_*`) these two modes are **mathematically identical** — no retraining happens between levels, so weight rank-ordering never changes and the same global threshold recovers the same result either way. They only genuinely diverge for `random_unstructured`, where iterative mode explicitly guarantees previously-zeroed weights stay zeroed as pruning progresses. See `run_masking_mode_comparison.py`.

## Evaluation Scenarios

Three scenarios (Methodology Sec. 3.3.4) are supported end-to-end via `JointSpaceEvaluator.run_eval(scenario=...)`:
- `"joint"` — both encoders pruned together (Combined Transmodal Atrophy / Semantic Dementia Hub Failure proxy).
- `"text_only"` — only the text encoder is pruned (Progressive Aphasia proxy).
- `"vision_only"` — only the vision encoder is pruned (Visual Object Agnosia proxy).

## Metrics

**Technical:** Top-1/coordinate/superordinate accuracy, MRR, CKA (representational similarity vs. pristine), Neighborhood Preservation Ratio, and **Shannon Vector Entropy** of the retrieval softmax distribution (Sec. 3.4.1 — rising entropy signals eroding confidence even where Top-1 hasn't flipped yet).

**Clinical/taxonomic:** retrieval failures are classified into a 4-tier error hierarchy — **Coordinate Error, Superordinate Error, Domain Error, Domain Collapse** (four tiers, not three — an intermediate "Domain Error" tier sits between Superordinate Error and full cross-domain Domain Collapse) — plus a per-pruning-level Typicality Delta (Δ = Acc_typical − Acc_atypical) once both Typical and Atypical exemplars exist for a coordinate class.

**Baselines:** a Gaussian noise-injection floor (`Bfloor`, Sec. 3.3.3) establishes chance-level performance on the exact taxonomy/dataset in use, computed once per sweep and plotted as a reference line alongside every pruning trajectory.

## Pipeline (`run_full_pipeline.py`, 9 stages)

1. **Visual Memory Indexing** — encodes the raw dataset through the pristine model (standalone sanity-check snapshot; not reused by the pruning sweeps below, which always re-encode fresh at each pruning level since the vision encoder itself is what's being damaged).
2. **Core Atrophy Sweep** — 19-stage (0–90%, 5% steps) joint/text-only/vision-only scenario comparison, full metrics suite, Bfloor reference.
3. **Subcategory t-SNE Trajectories**
4. **Concept Dispersion t-SNE Grids**
5. **Bozeat Visual Retrieval Grid** — text→image "draw from prompt" style benchmark.
6. **Modality/Location-Isolated Comparison** — lighter/faster joint vs. vision-only vs. text-only vs. projection-only ablation (`TestingHarness`).
7. **RQ2 Depth-Zone Grid** — early/middle/deep/global pruning, run and compared separately.
8. **Static vs. Iterative Masking Comparison**
9. **Pruning Strategy Comparison** — magnitude vs. random weight pruning, against the Bfloor noise floor.

Each of stages 3–9 fails independently without halting the run; the final summary reports which (if any) failed.

## Known Limitations

- Full fine-grained coverage (77 classes) requires a one-time network-dependent download step (`src/dataset_downloader.py`) not run automatically by the master pipeline.
- Tiny-ImageNet-200's 64×64px training crops are lower resolution and less curated for clean backgrounds than purpose-built neuropsych stimulus sets (THINGS/BOSS) — acceptable for this pilot's scope but worth flagging explicitly rather than implying THINGS/BOSS was used.
- With only 10/77 classes currently downloaded, statistics are based on a modest sample; this expands automatically (no code changes needed) as more classes are downloaded.
