# Clip-SD: Simulating Semantic Dementia by Progressively Pruning the CLIP Embedding Space

Code, data and results for the honours research project *Simulating Semantic Dementia by Progressively Pruning the
CLIP Embedding Space* (Emmanuel Azubuike, supervised by Dr Devon Jarvis, School of Computer Science and Applied
Mathematics, University of the Witwatersrand).

The project prunes the weights of OpenAI CLIP (ViT-B/32) in small steps, **without any retraining**, and asks whether
fine-grained category knowledge is lost before broad category knowledge, the order seen in semantic dementia (SD).
The model is not a replica of the disease. It is used to study whether a similar *pattern* of loss appears in a
large vision-language model.

---

## Contents

1. [Quick start](#quick-start)
2. [What the experiments do](#what-the-experiments-do)
3. [Repository layout](#repository-layout)
4. [The 12-stage pipeline](#the-12-stage-pipeline)
5. [Data](#data)
6. [Results](#results)
7. [Reproducing a run](#reproducing-a-run)
8. [Key design choices](#key-design-choices)
9. [Notes and limits](#notes-and-limits)
10. [Citation and licence](#citation-and-licence)

---

## Quick start

```bash
git clone https://github.com/EmmanuelAzu/Clip-SD.git
cd Clip-SD
bash setup_env.sh                    # creates ./venv and installs requirements.txt (written for a SLURM cluster)
# or, on any machine with Python 3.10+:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run_full_pipeline.py          # all 12 stages (needs a GPU for reasonable speed)
python run_full_pipeline.py --skip_indexing   # reuse an existing image index
sbatch run_overnight.slurm           # the same, as a SLURM job
```

Every run writes to its own folder, `data/results/run_YYYYMMDD_HHMMSS/`, together with a `run_manifest.json`
that records the classes, pruning schedule, Python version and git commit that produced it. A run never overwrites
an earlier one.

`requirements.txt` installs `torch`, `torchvision`, `ftfy`, `regex`, `tqdm`, `pandas`, `pillow`, `scikit-learn`,
`seaborn`, `matplotlib`, `numpy` and OpenAI's CLIP (from GitHub). The CLIP weights are downloaded on first use.

---

## What the experiments do

**Model.** OpenAI CLIP ViT-B/32. Each class is represented by a bare, lower-case text prompt (for example `lemon`),
with no sentence template.

**Data.** 33 classes in 9 coordinate groups, with 5 images per class (165 images), taken from Tiny-ImageNet-200
(64x64 pixels). 26 of the 33 classes are animals. The groups and class list are defined in
`src/curated_config.py`:

| Coordinate group | Classes |
|---|---|
| Domestic Dogs | German Shepherd, Golden Retriever, Labrador Retriever, Chihuahua, Standard Poodle, Yorkshire Terrier |
| Felines | Tabby Cat, Egyptian Cat, Persian Cat, Cougar, Lion |
| Insects | Monarch Butterfly, Sulphur Butterfly, Ladybug, Dragonfly, Mantis |
| Birds | Albatross, Goose, Black Stork, King Penguin |
| Primates | Chimpanzee, Orangutan, Baboon |
| Amphibians | Bullfrog, Tailed Frog, European Fire Salamander |
| Citrus | Lemon, Orange |
| Fruits | Banana, Pomegranate, Acorn |
| Kitchen Objects | Frying Pan, Teapot |

**Pruning.** L1 unstructured magnitude pruning, applied **globally within each encoder** (all targeted weights of an
encoder are ranked together and a single threshold is used). The targeted weights are the attention and MLP
matrices of the transformer blocks and the projection layers. Levels run from **0% to 75% in 2.5% steps (31
levels)**. Every level is made from a fresh copy of the intact model, so damage does not compound across levels.
Pruning is deterministic (seed 42), so each setting is run once.

**Scenarios.** `joint` (both the image and the text encoder are pruned) and `vision_only` (only the image encoder).
A `text_only` scenario is implemented but was not part of the reported results.

**Classification and error types.** Each image is classified by cosine similarity to the 33 class prompts.
Errors are placed in a four-tier taxonomy:

| Tier | Meaning (checked in this order) | Share if guessing at random |
|---|---|---|
| Coordinate error | wrong class, same coordinate group (for example one dog breed for another) | 9.8% |
| Superordinate error | different coordinate group, same superordinate category | 53.8% |
| Domain error | different superordinate category, same domain (living or non-living) | 24.6% |
| Domain collapse | the prediction is in the other domain (living versus non-living) | 11.7% |

The cascade is implemented in `src/metrics.py`.

The code also reports Top-1/5/10 accuracy, mean reciprocal rank, and the entropy of the softmax over the class
similarities (temperature 100, the scale CLIP uses at inference; maximum log2(33) = 5.04 bits).

**Retrieval tasks modelled on Bozeat et al. (2003).** Bozeat and colleagues asked patients with SD to draw concrete
objects under different amounts of support. We use two retrieval analogues, searching the pool of 165 images:

- **Text-only query**, an analogue of drawing from the name.
- **Text plus image query**, an analogue of delayed copying: the query is the normalised blend
  `0.5 x text embedding + 0.5 x mean embedding of the class's images` (`src/joint_anchor.py`, `alpha = 0.5`).

Retrieval errors are placed in the same four tiers.

**Cluster-routed retrieval.** An IVF-style two-step search. Each query is first routed to the nearest
coordinate-group centroid, then searched only inside that group. This splits failures into *routing* errors (broad
knowledge lost) and *within-group* errors (fine-grained knowledge lost). Routing accuracy, within-group accuracy,
hierarchical accuracy and flat (brute-force) accuracy are all reported.

**Embedding-space analysis.** t-SNE maps of the image embeddings, and joint maps of images with text prompts, at
each pruning level, plus confusion matrices for selected classes.

---

## Repository layout

```
Clip-SD/
├── README.md
├── requirements.txt
├── setup_env.sh                    # environment setup (SLURM cluster)
├── run_overnight.slurm             # SLURM job that runs the full pipeline
├── run_full_pipeline.py            # master orchestrator: runs all 12 stages
├── run_indexing.py                 # stage 1: build the image index
├── run_pipeline.py                 # stage 2: core evaluation (joint and vision_only)
├── run_bozeat_grid.py              # stages 3 and 10: Bozeat-style retrieval
├── run_tsne.py                     # stage 4: image t-SNE grid
├── run_hierarchy_graph.py          # stage 5: class-centre dendrogram
├── run_hierarchical_retrieval.py   # stages 6 and 11: cluster-routed retrieval
├── run_tsne_joint.py               # stage 8: joint image and text t-SNE
├── run_select_class_confusion.py   # stage 9: confusion matrices for selected classes
├── run_large_pool_validation.py    # stage 12: larger-pool check
├── src/
│   ├── curated_config.py           # the 33 classes, groups, colours, pruning schedule
│   ├── pruning_engine.py           # CLIPPruningEngine (L1, random, structured; static or iterative)
│   ├── dataset_downloader.py       # Tiny-ImageNet-200 download using data/taxonomy_full.json
│   ├── build_general_dataset.py    # extracts and arranges the chosen classes
│   ├── dataset_encoder.py          # encodes images and text with CLIP
│   ├── joint_evaluator.py          # core evaluation loop (accuracy, entropy, error tiers)
│   ├── metrics.py                  # accuracy, entropy, error taxonomy
│   ├── bozeat_experiment.py        # text-to-image retrieval under pruning
│   ├── joint_anchor.py             # text-plus-image query construction
│   ├── hierarchical_retrieval.py   # cluster-routed retrieval and its decomposition
│   ├── large_pool_validation.py    # retrieval against a larger distractor pool
│   ├── typicality_analysis.py      # typical versus atypical image accuracy
│   ├── generate_tsne_curated.py    # image t-SNE
│   ├── generate_tsne_joint.py      # joint t-SNE with text prompts
│   ├── generate_hierarchy_graph.py # dendrogram of class centres
│   ├── select_class_confusion.py   # confusion matrices
│   ├── generate_analysis_plots.py, plot_results.py, plot_clinical_decay.py,
│   │   shared_space_progression.py, testing_harness.py, visualize_dataset.py   # plotting and helper modules
├── data/
│   ├── taxonomy_full.json          # full class taxonomy (wnid -> names and groups)
│   ├── taxonomy_drawable_32.json
│   ├── metadata_processed.csv      # index of the downloaded images
│   ├── images/                     # downloaded Tiny-ImageNet images (72 classes)
│   └── processed/                  # processed images and taxonomy files
├── results/
│   └── run_v3_final/               # the run reported in the paper (see "Results")
├── tests/                          # taxonomy test and encoder check
└── tools/                          # small checks: class coverage in the index, frying-pan class
```

---

## The 12-stage pipeline

`run_full_pipeline.py` runs these stages in order, all on the curated 33-class set.

| # | Stage | Main code |
|---|---|---|
| 1 | Image indexing | `run_indexing.py` |
| 2 | Core evaluation: joint versus vision-only; Top-1/5/10, entropy, four-tier error taxonomy | `run_pipeline.py`, `src/joint_evaluator.py`, `src/metrics.py` |
| 3 | Flat text-to-image retrieval (drawing-from-the-name analogue) | `src/bozeat_experiment.py` |
| 4 | Image t-SNE grid | `src/generate_tsne_curated.py` |
| 5 | Dendrogram of class centres | `src/generate_hierarchy_graph.py` |
| 6 | Cluster-routed retrieval versus flat retrieval | `src/hierarchical_retrieval.py` |
| 7 | Typicality analysis | `src/typicality_analysis.py` |
| 8 | Joint image-and-text t-SNE | `src/generate_tsne_joint.py` |
| 9 | Confusion matrices for selected classes | `src/select_class_confusion.py` |
| 10 | Flat retrieval with text-plus-image queries (delayed-copying analogue) | `src/joint_anchor.py` |
| 11 | Cluster-routed retrieval with text-plus-image queries | `src/joint_anchor.py` |
| 12 | Larger-pool validation (supplementary check) | `src/large_pool_validation.py` |

Stages 10 and 11 write to separate `*_joint_query/` folders, so they are always directly comparable with stages 3 and
6 and never overwrite them. Stage 12 is a supplementary check on a larger distractor pool and is not a main result.

Individual stages can be run on their own, for example:

```bash
python run_bozeat_grid.py --query_mode text            # drawing-from-the-name analogue
python run_bozeat_grid.py --query_mode joint --joint_alpha 0.5   # text plus image
python run_hierarchical_retrieval.py --scenario joint --query_mode text
python run_tsne_joint.py --scenario joint
python run_select_class_confusion.py --classes "Labrador Retriever" "Golden Retriever"
```

All of these accept `--output_dir` to choose where the results go.

---

## Data

- **Source:** Tiny-ImageNet-200 (Le and Yang, 2015), a 200-class subset of ImageNet with 64x64 images.
  The images used here are included in `data/images/` for convenience (72 classes downloaded; the 33 classes in the
  curated set are the ones used in all reported results).
- **Class list and groups:** `src/curated_config.py` (the curated set) and `data/taxonomy_full.json` (the full
  taxonomy, with the broader coordinate groups used by the larger-pool check).
- **Images per class:** a fixed 5 per class for the main experiments.
- **Index:** `run_indexing.py` writes `data/processed/metadata_processed.csv` and an `image_index.pt` tensor.
  `.pt` files are excluded by `.gitignore`, so the index is rebuilt on a new machine (drop `--skip_indexing` on
  the first run).
- `python tools/check_indexing_coverage.py` confirms that all 33 curated classes are present in the index.

---

## Results

`results/run_v3_final/` holds the output of the run reported in the paper (run time stamp
`20260916_113720`; see `run_manifest.json`).

| File or folder | What it holds |
|---|---|
| `joint_space_metrics.csv` | Per-level metrics, joint scenario: Top-1/5/10, coordinate-level and superordinate-level accuracy, MRR, entropy, error counts per tier, typical and atypical accuracy |
| `scenario_comparison.csv` | The same metrics for the joint and vision-only scenarios |
| `accuracy_curve.png`, `error_taxonomy_heatmap.png`, `scenario_comparison_plot.png` | Classification curves and the error-tier heatmap |
| `bozeat_experiment/` | Text-only retrieval (drawing-from-the-name analogue): per-query results (`bozeat_retrieval_results.csv`), accuracy curve and grid of retrieved images |
| `bozeat_experiment_joint_query/` | The same with text-plus-image queries (delayed-copying analogue) |
| `hierarchical_retrieval/` | Cluster-routed retrieval, text-only: per-query results, the routing/within-group summary (`..._decomposition_summary.csv`) and plots |
| `hierarchical_retrieval_joint_query/` | The same with text-plus-image queries |
| `tsne_curated.png`, `tsne_joint.png` | Image t-SNE grid and joint image-and-text t-SNE |
| `select_class_confusion_matrices.png` | Confusion matrices for selected classes |
| `typicality_effect.png`, `hierarchy_dendrogram.png` | Typicality analysis and class-centre dendrogram |
| `large_pool_validation/` | Supplementary larger-pool check |
| `run_manifest.json` | Classes, groups, pruning levels, Python version and settings of the run |

Column notes for the CSV files:

- `pruning_level` is the fraction of targeted weights set to zero (0.0 to 0.75).
- `top1_specific_acc` is exact-class accuracy. `top1_coordinate_acc` counts a prediction as correct if it is in the
  right coordinate group, and `top1_super_acc` if it is in the right superordinate category.
- `coordinate error`, `superordinate error`, `domain error`, `domain collapse` are counts of errors in each tier.
- In the retrieval files, `correct` says whether the top-ranked image is of the queried class and `top1_similarity`
  is its cosine similarity. The image paths in these files are the paths on the cluster where the run was made.
- In the hierarchical summaries, `routing_acc` is the share of queries sent to the correct group,
  `within_group_acc_given_correct_routing` is accuracy inside the correct group, `hierarchical_acc` is their
  product and `flat_acc` is brute-force accuracy over all images.

---

## Reproducing a run

1. Set up the environment as in [Quick start](#quick-start).
2. Make sure `data/images/` holds the images (included in this repository) or rebuild them with
   `src/dataset_downloader.py` and `src/build_general_dataset.py`.
3. Run `python run_full_pipeline.py` (without `--skip_indexing` the first time).
4. Compare `data/results/run_<timestamp>/joint_space_metrics.csv` with
   `results/run_v3_final/joint_space_metrics.csv`. Pruning is deterministic, so the metrics should match on the
   same model weights and images. Small numerical differences can appear across GPU types and library versions.

The reported run was made on a SLURM GPU node (`run_overnight.slurm`, partition `biggpu`, 24-hour limit). Edit the
partition and module lines for your own cluster.

---

## Key design choices

- **No retraining.** Pruning only removes weights. This isolates the effect of damage, as in a lesion study.
- **Global magnitude pruning per encoder.** A single threshold per encoder is used rather than the same sparsity
  in every layer, which would be more destructive at the same nominal level.
- **Fresh copy for every level.** Levels are independent, and runs are deterministic.
- **Bare class names as prompts.** This keeps the task like naming a picture with a single word.
- **Coordinate groups with several members.** Within-group (fine) and between-group (broad) errors can only be
  told apart if a group has more than one class, so each group has 2 to 6.
- **Two query types.** Text-only and text-plus-image queries correspond to two Bozeat support conditions, and
  there is no analogue here of immediate copying.

---

## Notes and limits

- Pruning is deterministic, so each setting was run once and there are no error bars.
- 26 of the 33 classes are animals, so superordinate accuracy has a floor of 26/33 = 0.788 (the value reached by
  always predicting one animal).
- With 5 images per class and 33 queries per retrieval run, each retrieval point rests on a small sample.
  Bootstrapped intervals would need more images per class.
- Text-plus-image queries use the class's own images from the searched pool, so they are not leakage-free.
  A held-out-image version is a natural extension.
- Layer-targeted pruning (early, middle, deep) and the `text_only` scenario are implemented but not part of the
  reported results.
- The retrieval CSV files store absolute image paths from the cluster where the run was made.

---

## Citation and licence

If you use this code, please cite the accompanying paper (details to be added when published) and the data and
model sources:

- Radford et al. (2021). Learning transferable visual models from natural language supervision. ICML.
- Le and Yang (2015). Tiny ImageNet Visual Recognition Challenge. Stanford CS231N.
- Bozeat et al. (2003). A duck with four legs: investigating the structure of conceptual knowledge using picture
  drawing in semantic dementia. Cognitive Neuropsychology 20:27-47.

No licence file is included yet. Until one is added, all rights are reserved by the author.
