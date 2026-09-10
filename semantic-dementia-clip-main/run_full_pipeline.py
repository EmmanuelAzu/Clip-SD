import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Root-level entry point imports
try:
    from run_bozeat_grid import main as run_bozeat_grid_main
    from run_indexing import run_indexing
    from run_pipeline import run_joint_pipeline
    from run_tsne import main as run_tsne_main
except ImportError as e:
    print(f"[!] Import error detected: {e}")
    print("[!] Ensure all run_*.py scripts are present in the project root.")
    sys.exit(1)


def prepare_directories(base_dir: str) -> dict[str, str]:
    """Ensures all output and data directories exist prior to pipeline execution."""
    dirs = {
        "data_images": os.path.join(base_dir, "data", "images"),
        "data_processed": os.path.join(base_dir, "data", "processed"),
        "results_base": os.path.join(base_dir, "data", "results"),
        "results_bozeat": os.path.join(base_dir, "data", "results", "bozeat_experiment"),
        "logs": os.path.join(base_dir, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def execute_master_pipeline(
    skip_indexing: bool = False,
    output_dir: str | None = None,
) -> None:
    """Streamlined 4-stage pipeline, scoped to the curated 10-class subset
    (src/curated_config.py) throughout:

      1. Visual memory indexing (once, over the full downloaded dataset --
         cheap and only needs re-running when new images are added).
      2. Core evaluation: joint vs. vision-only scenario comparison, the
         cross-category clinical error breakdown, 2.5% pruning increments
         from 0-70%.
      3. Bozeat text->image retrieval experiment.
      4. One consolidated tSNE grid.

    Previously a 9-stage pipeline evaluating the full ~7,200-image
    balanced dataset at every stage; this is the redesigned, focused
    version -- see src/curated_config.py for the rationale.
    """
    start_time = time.time()
    dirs = prepare_directories(PROJECT_ROOT)
    results_dir = output_dir if output_dir else dirs["results_base"]

    print("=" * 70)
    print("  SEMANTIC DEMENTIA CLIP ATROPHY SIMULATION -- CURATED PIPELINE  ")
    print("=" * 70)
    print(f"[*] Root Directory        : {PROJECT_ROOT}")
    print(f"[*] Results Output Path   : {results_dir}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Offline Visual Memory Bank Indexing
    # -------------------------------------------------------------------------
    if not skip_indexing:
        print("\n[STAGE 1/4] Building Visual Memory Index...")
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
        print("\n[STAGE 1/4] Skipping Visual Memory Indexing (--skip_indexing set).")

    stage_failures = []

    # -------------------------------------------------------------------------
    # STAGE 2: Core Evaluation (joint vs. vision-only, cross-category plot)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/4] Running Core Evaluation (curated 10-class subset)...")
    try:
        csv_metrics = run_joint_pipeline(output_dir=results_dir)
        print(f"[+] Stage 2 Complete. Metrics exported to: {csv_metrics}")
    except Exception as err:
        print(f"[!] Stage 2 Failed: {err}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: Bozeat Text->Image Retrieval Experiment
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/4] Running Bozeat Retrieval Experiment...")
    try:
        run_bozeat_grid_main(output_dir=dirs["results_bozeat"])
        print("[+] Stage 3 Complete.")
    except Exception as err:
        print(f"[!] Stage 3 Failed: {err}")
        stage_failures.append(("Stage 3 (Bozeat Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 4: Curated t-SNE Grid
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/4] Generating Curated t-SNE Grid...")
    try:
        run_tsne_main(output_dir=results_dir)
        print("[+] Stage 4 Complete.")
    except Exception as err:
        print(f"[!] Stage 4 Failed: {err}")
        stage_failures.append(("Stage 4 (t-SNE Grid)", err))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    if stage_failures:
        print(f" [!] MASTER PIPELINE FINISHED WITH {len(stage_failures)} FAILED STAGE(S) IN {elapsed:.2f}s")
        for stage_name, err in stage_failures:
            print(f"     - {stage_name}: {err}")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f" [+] MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
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
