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
