import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

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
    from run_hierarchical_retrieval import main as run_hierarchical_retrieval_main
    from src.typicality_analysis import plot_typicality_effect
    from run_tsne_joint import main as run_tsne_joint_main
    from run_select_class_confusion import main as run_select_class_confusion_main
    from run_large_pool_validation import main as run_large_pool_validation_main
    from src.curated_config import CURATED_CLASSES, CURATED_TAXONOMY, PRUNING_LEVELS_FOCUSED, IMAGES_PER_CLASS
except ImportError as e:
    print(f"[!] Import error detected: {e}")
    print("[!] Ensure all run_*.py scripts are present in the project root.")
    sys.exit(1)


def prepare_directories(base_dir: str, run_dir: str) -> dict[str, str]:
    """Ensures all output and data directories exist prior to pipeline execution.

    Every run gets its own timestamped subfolder under data/results/ (see
    _make_run_dir below), so results from different runs/pipeline versions
    never collide or silently overwrite each other -- previously every run
    wrote into the same data/results/ folder, and since output filenames
    changed across pipeline redesigns (e.g. category_breakdown_suite.png
    was later split into accuracy_curve.png + error_taxonomy_heatmap.png),
    old files from earlier pipeline versions just accumulated there
    forever with no way to tell which run produced which file.
    """
    dirs = {
        "data_images": os.path.join(base_dir, "data", "images"),
        "data_processed": os.path.join(base_dir, "data", "processed"),
        "results_base": os.path.join(base_dir, "data", "results"),
        "run_dir": run_dir,
        "results_bozeat": os.path.join(run_dir, "bozeat_experiment"),
        "logs": os.path.join(base_dir, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _make_run_dir(base_dir: str) -> str:
    """Creates data/results/run_<timestamp>/ and returns its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, "data", "results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _get_git_commit(base_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=base_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git repo, or git unavailable)"


def _write_run_manifest(run_dir: str, skip_indexing: bool) -> None:
    """Writes run_manifest.json inside the run folder -- a self-contained
    record of exactly what configuration produced this run's results, so
    you can tell runs apart later without guessing from filenames alone.
    """
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_commit(PROJECT_ROOT),
        "skip_indexing": skip_indexing,
        "curated_classes": CURATED_CLASSES,
        "n_curated_classes": len(CURATED_CLASSES),
        "coordinate_groups": {g: members for g, members in CURATED_TAXONOMY.items()},
        "images_per_class": IMAGES_PER_CLASS,
        "pruning_levels": PRUNING_LEVELS_FOCUSED,
        "python_version": sys.version,
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[*] Run manifest written to: {manifest_path}")


def _update_latest_pointer(base_dir: str, run_dir: str) -> None:
    """Updates data/results/LATEST_RUN.txt to point at this run, and
    (best-effort) refreshes a data/results/latest symlink too, so you
    always have a stable path to "the current run" without needing to
    know its timestamp. The plain-text pointer is the reliable one --
    symlinks can behave inconsistently once copied over scp/Windows.
    """
    results_base = os.path.join(base_dir, "data", "results")
    run_name = os.path.basename(run_dir)

    with open(os.path.join(results_base, "LATEST_RUN.txt"), "w") as f:
        f.write(run_name + "\n")

    latest_link = os.path.join(results_base, "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(run_name, latest_link)
    except OSError as e:
        print(f"[!] Could not create 'latest' symlink (non-fatal): {e}")


def execute_master_pipeline(
    skip_indexing: bool = False,
    output_dir: str | None = None,
) -> None:
    """Streamlined 12-stage pipeline, scoped to the curated multi-breed
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
      6. Hierarchical (cluster-routed) retrieval vs. flat search --
         routes each query to a coordinate-group centroid first, then
         searches only within that group, decomposing failure into
         routing (broad-category) errors vs. within-group (fine-grained)
         errors, and comparing against the existing flat/brute-force
         Bozeat baseline.
      7. Typicality effect analysis (RQ3) -- plots acc_typical,
         acc_atypical, and delta_typicality (already computed by Stage 2,
         previously never visualized) against the paper's stated
         validation criterion: delta should be positive and widen as
         pruning increases.
      8. Joint (image + text) t-SNE -- a genuinely joint embedding-space
         visualization (Stage 4's t-SNE is image-only, despite its old
         title's claim otherwise, now corrected) showing where each
         class's text anchor sits relative to its own image cluster, and
         whether that anchor drifts away as pruning increases.
      9. Select-class confusion matrices, 0-35% pruning -- a simpler,
         literal NxN confusion matrix (true vs. predicted class) for a
         handful of familiar classes, complementing Stage 2's taxonomy-
         tier error breakdown with an easier-to-read view.
      10-11. Bozeat and hierarchical retrieval, re-run with JOINT
         (text+mean-image blended) queries instead of pure text
         (src/joint_anchor.py) -- tests Proposal Sec. 2.2's transmodal-hub
         framing directly: does grounding the query in both modalities,
         not just language, change retrieval robustness under pruning?
         Written to separate *_joint_query/ subfolders so these are
         always directly comparable against Stages 3 and 6's pure-text
         results, never overwriting them.
      12. Large pool validation -- re-runs Bozeat and hierarchical
         retrieval against all ~72 downloaded classes (not just the
         curated 33), at 6 checkpoints (0-50%), testing whether the
         collapse patterns from Stages 3 and 6 hold up against a bigger,
         more realistic distractor set. Note: hierarchical routing here
         uses the FULL taxonomy's own, broader coordinate groups (e.g.
         "Ungulates", "Arachnids"), not the curated 9-group scheme, since
         the latter doesn't cover the +44 non-curated classes -- see
         src/large_pool_validation.py's docstring for the full rationale.

    Every run is written to its own timestamped folder
    (data/results/run_YYYYMMDD_HHMMSS/) with a run_manifest.json recording
    exactly what configuration (classes, pruning schedule, git commit)
    produced it -- pass --output_dir explicitly to override this and reuse
    a specific folder instead.

    See src/curated_config.py for the full design rationale (33 specific
    breeds/species across 9 coordinate groups, 5 images/class, weighted
    toward Animals/Plants).
    """
    start_time = time.time()
    run_dir = output_dir if output_dir else _make_run_dir(PROJECT_ROOT)
    dirs = prepare_directories(PROJECT_ROOT, run_dir)
    results_dir = run_dir

    print("=" * 70)
    print("  SEMANTIC DEMENTIA CLIP ATROPHY SIMULATION -- CURATED PIPELINE  ")
    print("=" * 70)
    print(f"[*] Root Directory        : {PROJECT_ROOT}")
    print(f"[*] Results Output Path   : {results_dir}")
    print("=" * 70)

    _write_run_manifest(run_dir, skip_indexing)

    # -------------------------------------------------------------------------
    # STAGE 1: Offline Visual Memory Bank Indexing
    # -------------------------------------------------------------------------
    if not skip_indexing:
        print("\n[STAGE 1/12] Building Visual Memory Index...")
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
        print("\n[STAGE 1/12] Skipping Visual Memory Indexing (--skip_indexing set).")

    stage_failures = []

    # -------------------------------------------------------------------------
    # STAGE 2: Core Evaluation (joint vs. vision-only, cross-category plots)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/12] Running Core Evaluation (curated multi-breed subset)...")
    try:
        csv_metrics = run_joint_pipeline(output_dir=results_dir)
        print(f"[+] Stage 2 Complete. Metrics exported to: {csv_metrics}")
    except Exception as err:
        print(f"[!] Stage 2 Failed: {err}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: Bozeat Text->Image Retrieval Experiment
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/12] Running Bozeat Retrieval Experiment...")
    try:
        run_bozeat_grid_main(output_dir=dirs["results_bozeat"])
        print("[+] Stage 3 Complete.")
    except Exception as err:
        print(f"[!] Stage 3 Failed: {err}")
        stage_failures.append(("Stage 3 (Bozeat Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 4: Curated t-SNE Grid
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/12] Generating Curated t-SNE Grid...")
    try:
        run_tsne_main(output_dir=results_dir)
        print("[+] Stage 4 Complete.")
    except Exception as err:
        print(f"[!] Stage 4 Failed: {err}")
        stage_failures.append(("Stage 4 (t-SNE Grid)", err))

    # -------------------------------------------------------------------------
    # STAGE 5: Hierarchical Dendrogram
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/12] Generating Hierarchical Dendrogram...")
    try:
        run_hierarchy_graph_main(output_dir=results_dir)
        print("[+] Stage 5 Complete.")
    except Exception as err:
        print(f"[!] Stage 5 Failed: {err}")
        stage_failures.append(("Stage 5 (Hierarchical Dendrogram)", err))

    # -------------------------------------------------------------------------
    # STAGE 6: Hierarchical (Cluster-Routed) Retrieval vs. Flat Search
    # -------------------------------------------------------------------------
    print("\n[STAGE 6/12] Running Hierarchical Cluster-Routed Retrieval Experiment...")
    try:
        run_hierarchical_retrieval_main(output_dir=os.path.join(run_dir, "hierarchical_retrieval"))
        print("[+] Stage 6 Complete.")
    except Exception as err:
        print(f"[!] Stage 6 Failed: {err}")
        stage_failures.append(("Stage 6 (Hierarchical Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 7: Typicality Effect Analysis (RQ3 -- previously computed, never examined)
    # -------------------------------------------------------------------------
    print("\n[STAGE 7/12] Generating Typicality Effect Analysis...")
    try:
        if csv_metrics is not None:
            plot_typicality_effect(csv_metrics, output_dir=results_dir)
            print("[+] Stage 7 Complete.")
        else:
            print("[!] Stage 7 Skipped: Stage 2's metrics CSV is unavailable (Stage 2 failed or was skipped).")
    except Exception as err:
        print(f"[!] Stage 7 Failed: {err}")
        stage_failures.append(("Stage 7 (Typicality Effect)", err))

    # -------------------------------------------------------------------------
    # STAGE 8: Joint (Image + Text) t-SNE
    # -------------------------------------------------------------------------
    print("\n[STAGE 8/12] Generating Joint (Image + Text) t-SNE Grid...")
    try:
        run_tsne_joint_main(output_dir=results_dir)
        print("[+] Stage 8 Complete.")
    except Exception as err:
        print(f"[!] Stage 8 Failed: {err}")
        stage_failures.append(("Stage 8 (Joint t-SNE)", err))

    # -------------------------------------------------------------------------
    # STAGE 9: Select-Class Confusion Matrices (0-35% pruning)
    # -------------------------------------------------------------------------
    print("\n[STAGE 9/12] Generating Select-Class Confusion Matrices...")
    try:
        run_select_class_confusion_main(output_dir=results_dir)
        print("[+] Stage 9 Complete.")
    except Exception as err:
        print(f"[!] Stage 9 Failed: {err}")
        stage_failures.append(("Stage 9 (Select-Class Confusion)", err))

    # -------------------------------------------------------------------------
    # STAGE 10: Bozeat Retrieval -- Joint (Text+Image) Query Mode
    # -------------------------------------------------------------------------
    print("\n[STAGE 10/12] Running Bozeat Retrieval with Joint (Text+Image) Queries...")
    try:
        run_bozeat_grid_main(output_dir=os.path.join(run_dir, "bozeat_experiment_joint_query"), query_mode="joint")
        print("[+] Stage 10 Complete.")
    except Exception as err:
        print(f"[!] Stage 10 Failed: {err}")
        stage_failures.append(("Stage 10 (Bozeat Joint Query)", err))

    # -------------------------------------------------------------------------
    # STAGE 11: Hierarchical Retrieval -- Joint (Text+Image) Query Mode
    # -------------------------------------------------------------------------
    print("\n[STAGE 11/12] Running Hierarchical Retrieval with Joint (Text+Image) Queries...")
    try:
        run_hierarchical_retrieval_main(
            output_dir=os.path.join(run_dir, "hierarchical_retrieval_joint_query"), query_mode="joint"
        )
        print("[+] Stage 11 Complete.")
    except Exception as err:
        print(f"[!] Stage 11 Failed: {err}")
        stage_failures.append(("Stage 11 (Hierarchical Retrieval Joint Query)", err))

    # -------------------------------------------------------------------------
    # STAGE 12: Large Pool Validation (all ~72 downloaded classes, not just
    # the curated 33) -- tests whether the collapse patterns seen in Stages
    # 3 and 6 hold up against a bigger, more realistic distractor set.
    # -------------------------------------------------------------------------
    print("\n[STAGE 12/12] Running Large Pool Validation...")
    try:
        run_large_pool_validation_main(output_dir=os.path.join(run_dir, "large_pool_validation"))
        print("[+] Stage 12 Complete.")
    except Exception as err:
        print(f"[!] Stage 12 Failed: {err}")
        stage_failures.append(("Stage 12 (Large Pool Validation)", err))

    elapsed = time.time() - start_time
    _update_latest_pointer(PROJECT_ROOT, run_dir)

    print("\n" + "=" * 70)
    if stage_failures:
        print(f" [!] MASTER PIPELINE FINISHED WITH {len(stage_failures)} FAILED STAGE(S) IN {elapsed:.2f}s")
        for stage_name, err in stage_failures:
            print(f"     - {stage_name}: {err}")
        print(f" [*] Results (partial): {run_dir}")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f" [+] MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
        print(f" [*] Results: {run_dir}")
        print(f" [*] Also available at: data/results/latest -> {os.path.basename(run_dir)}")
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
