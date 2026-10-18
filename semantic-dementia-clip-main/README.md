# Simulating Semantic Dementia Through Progressive Pruning of the CLIP Embedding Space

Honours research code, School of Computer Science and Applied Mathematics, University of the Witwatersrand.

The project prunes the weights of CLIP (ViT-B/32) in small steps, without retraining, and measures whether
fine-grained category knowledge is lost before broad category knowledge, as in semantic dementia.

## What the code does

- **Model:** OpenAI CLIP ViT-B/32. Class names are used as bare lowercase prompts (for example `lemon`).
- **Data:** 33 classes in 9 coordinate groups, 5 images per class (165 images), taken from Tiny-ImageNet-200.
  The class list, groups and typicality labels are in `src/curated_config.py` and `data/taxonomy_full.json`.
- **Pruning:** L1 unstructured magnitude pruning, global within each encoder, from 0% to 75% in 2.5% steps
  (31 levels). Every level starts again from the intact model (`src/pruning_engine.py`).
- **Scenarios:** `joint` (both encoders pruned) and `vision_only` (only the image encoder pruned).

## The 12-stage pipeline (`run_full_pipeline.py`)

1. Image indexing
2. Core evaluation: Top-1/5/10, entropy, four-tier error taxonomy (`src/joint_evaluator.py`, `src/metrics.py`)
3. Flat (Bozeat-style) text-to-image retrieval (`src/bozeat_experiment.py`)
4. Image t-SNE grid
5. Class-centre dendrograms
6. Cluster-routed hierarchical retrieval (`src/hierarchical_retrieval.py`)
7. Typicality analysis (`src/typicality_analysis.py`)
8. Joint image-and-text t-SNE (`src/generate_tsne_joint.py`)
9. Select-class confusion matrices (`src/select_class_confusion.py`)
10. Flat retrieval with joint text-plus-image queries (`src/joint_anchor.py`)
11. Hierarchical retrieval with joint queries
12. Larger-pool validation (`src/large_pool_validation.py`)

## Running it

```bash
bash setup_env.sh                   # create the environment (cluster)
python run_full_pipeline.py         # full pipeline; add --skip_indexing to reuse the index
sbatch run_overnight.slurm          # the same, as a SLURM job
```

Each run writes a timestamped folder under `data/results/`.

## Final results

`results/run_v3_final/` holds the CSV files and plots from the run reported in the paper
(`run_20260916_113720_v3`). `results/run_v3_final/joint_space_metrics.csv` has the main per-level metrics.

## Notes and known limits

- Pruning is deterministic, so each setting was run once and there are no error bars.
- 26 of the 33 classes are animals, so superordinate accuracy has a floor of 26/33 = 0.788.
- Joint queries use the class's own images from the searched pool, so they are not leakage-free.
- Layer-targeted pruning and the text-only scenario are implemented but were not part of the reported results.
- `apply_*.sh` are earlier patch scripts kept for history. The files in `src/` and the `run_*.py` scripts
  are the current versions.
