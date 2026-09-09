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
    from run_tsne_5stages import main as run_tsne_5stages_main
    from run_tSNE_nerr import main as run_tsne_nerr_main
    from run_tsne_no_errors import main as run_tsne_no_errors_main
    from run_depth_zone_grid import main as run_depth_zone_grid_main
    from run_masking_mode_comparison import main as run_masking_mode_comparison_main
    from run_pruning_strategy_comparison import main as run_pruning_strategy_comparison_main
    from src.testing_harness import TestingHarness
except ImportError as e:
    print(f"[!] Import error detected: {e}")
    print("[!] Ensure all run_*.py scripts are present in the project root.")
    sys.exit(1)


def prepare_directories(base_dir: str) -> dict[str, str]:
    """Ensures all output and data directories exist prior to pipeline execution."""
    dirs = {
        "data_raw": os.path.join(base_dir, "data", "raw"),
        "data_processed": os.path.join(base_dir, "data", "processed"),
        "results_base": os.path.join(base_dir, "data", "results"),
        "results_tsne": os.path.join(base_dir, "data", "results", "tsne"),
        "results_confusion": os.path.join(
            base_dir, "data", "results", "confusion_matrices"
        ),
        "results_hierarchical": os.path.join(
            base_dir, "data", "results", "hierarchical_confusion"
        ),
        "results_bozeat": os.path.join(
            base_dir, "data", "results", "bozeat_experiment"
        ),
        "results_location_comparison": os.path.join(
            base_dir, "data", "results", "location_comparison"
        ),
        "results_depth_zone": os.path.join(
            base_dir, "data", "results", "depth_zone_grid"
        ),
        "results_masking_mode": os.path.join(
            base_dir, "data", "results", "masking_mode_comparison"
        ),
        "results_pruning_strategy": os.path.join(
            base_dir, "data", "results", "pruning_strategy_comparison"
        ),
        "logs": os.path.join(base_dir, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def execute_master_pipeline(
    sample_frac: float = 1.0,
    target_n: int | None = None,
    balance_taxonomically: bool = True,
    skip_indexing: bool = False,
    output_dir: str | None = None,
) -> None:
    """Executes the full end-to-end multimodal semantic dementia simulation pipeline."""
    start_time = time.time()
    dirs = prepare_directories(PROJECT_ROOT)
    results_dir = output_dir if output_dir else dirs["results_base"]

    print("=" * 70)
    print("  MULTIMODAL SEMANTIC DEMENTIA CLIP ATROPHY SIMULATION PIPELINE  ")
    print("=" * 70)
    print(f"[*] Root Directory        : {PROJECT_ROOT}")
    print(f"[*] Results Output Path   : {results_dir}")
    print(f"[*] Sample Fraction       : {sample_frac}")
    print(f"[*] Target Sample Count   : {target_n}")
    print(f"[*] Taxonomic Class Balancing: {balance_taxonomically}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Offline Visual Memory Bank Indexing
    # -------------------------------------------------------------------------
    if not skip_indexing:
        print("\n[STAGE 1/9] Building Visual Memory Index...")
        try:
            indexed_path = run_indexing(
                raw_csv=os.path.join(
                    PROJECT_ROOT, "tests", "metadata_raw.csv"
                ),
                img_dir=dirs["data_raw"],
                output_dir=dirs["data_processed"],
            )
            print(f"[+] Stage 1 Complete. Saved tensor bank to: {indexed_path}")
        except Exception as err:
            print(f"[!] Stage 1 Failed: {err}")
            sys.exit(1)
    else:
        print("\n[STAGE 1/9] Skipping Visual Memory Indexing (--skip_indexing set).")

    # -------------------------------------------------------------------------
    # STAGE 2: 19-Stage Quantitative Evaluation & Metric Suite
    # (also runs the joint/text_only/vision_only Three-Scenario Comparison,
    # Sec. 3.3.4 Stage 7.5, and the Bfloor Gaussian noise-floor reference,
    # Sec. 3.3.3 -- both now implemented inside run_joint_pipeline().)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/9] Running Multimodal Atrophy Pipeline (0% to 90%)...")
    try:
        csv_metrics = run_joint_pipeline(
            sample_frac=sample_frac,
            target_n=target_n,
            balance_taxonomically=balance_taxonomically,
            output_dir=results_dir,
        )
        print(f"[+] Stage 2 Complete. Metrics exported to: {csv_metrics}")
    except Exception as err:
        print(f"[!] Stage 2 Failed: {err}")
        sys.exit(1)

    # Tracks per-stage outcomes so the final banner can never claim success
    # while a stage actually failed (previously each late stage's exception
    # was swallowed and printed only as a warning, so the pipeline reported
    # "COMPLETED SUCCESSFULLY" even when, e.g., Stage 5 crashed outright).
    stage_failures = []

    # -------------------------------------------------------------------------
    # STAGE 3: Subcategory Keyed t-SNE Trajectories (5-stage / Keyed Grid)
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/9] Generating Subcategory t-SNE Trajectory Grids...")
    try:
        run_tsne_5stages_main()
        print("[+] Stage 3 Complete.")
    except Exception as err:
        print(f"[!] Stage 3 Failed: {err}")
        stage_failures.append(("Stage 3 (Subcategory t-SNE)", err))

    # -------------------------------------------------------------------------
    # STAGE 4: Concept Dispersion & Topological Error t-SNE Grids
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/9] Generating Concept Dispersion & Error t-SNE Grids...")
    try:
        print("  --> Generating Domain-Grouped Concept Dispersion...")
        run_tsne_nerr_main()
        print("  --> Generating Clean Concept Dispersion Grid...")
        run_tsne_no_errors_main()
        print("[+] Stage 4 Complete.")
    except Exception as err:
        print(f"[!] Stage 4 Failed: {err}")
        stage_failures.append(("Stage 4 (Concept Dispersion t-SNE)", err))

    # -------------------------------------------------------------------------
    # STAGE 5: Bozeat Visual Retrieval Grid Benchmark
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/9] Executing Bozeat Visual Retrieval Experiment...")
    try:
        run_bozeat_grid_main()
        print("[+] Stage 5 Complete.")
    except Exception as err:
        print(f"[!] Stage 5 Failed: {err}")
        stage_failures.append(("Stage 5 (Bozeat Visual Retrieval)", err))

    # -------------------------------------------------------------------------
    # STAGE 6: Modality-Isolated Location Comparison (fast/simple version)
    # -- Joint Projection / Vision-Only Projection / Text-Only Projection /
    # Full Backbone, via TestingHarness.compare_pruning_locations(). This is
    # the lightweight companion to Stage 2's richer scenario comparison
    # (which uses the full CKA/NPR/entropy/typicality metrics suite);
    # this one is faster and additionally covers the "projection-only"
    # ablation arm.
    # -------------------------------------------------------------------------
    print("\n[STAGE 6/9] Running Modality/Location-Isolated Pruning Comparison...")
    try:
        harness = TestingHarness()
        comparison_df = harness.compare_pruning_locations(max_pruning=0.90, step=0.05, top_k=10)
        comp_csv = os.path.join(dirs["results_location_comparison"], "location_comparison_simulation.csv")
        comparison_df.to_csv(comp_csv, index=False)
        print(f"[+] Stage 6 Complete. Saved to: {comp_csv}")
    except Exception as err:
        print(f"[!] Stage 6 Failed: {err}")
        stage_failures.append(("Stage 6 (Location Comparison)", err))

    # -------------------------------------------------------------------------
    # STAGE 7: RQ2 -- Depth-Zone Targeted Pruning Grid (early/middle/deep/global)
    # -------------------------------------------------------------------------
    print("\n[STAGE 7/9] Running RQ2 Depth-Zone Pruning Grid...")
    try:
        run_depth_zone_grid_main(output_dir=dirs["results_depth_zone"])
        print("[+] Stage 7 Complete.")
    except Exception as err:
        print(f"[!] Stage 7 Failed: {err}")
        stage_failures.append(("Stage 7 (Depth-Zone Grid)", err))

    # -------------------------------------------------------------------------
    # STAGE 8: Static vs. Iterative (Compounding) Masking Comparison
    # -------------------------------------------------------------------------
    print("\n[STAGE 8/9] Running Static vs. Iterative Masking Comparison...")
    try:
        run_masking_mode_comparison_main(output_dir=dirs["results_masking_mode"])
        print("[+] Stage 8 Complete.")
    except Exception as err:
        print(f"[!] Stage 8 Failed: {err}")
        stage_failures.append(("Stage 8 (Masking Mode Comparison)", err))

    # -------------------------------------------------------------------------
    # STAGE 9: Pruning Strategy Comparison (Magnitude vs. Random vs. Bfloor)
    # -------------------------------------------------------------------------
    print("\n[STAGE 9/9] Running Pruning Strategy Comparison (Magnitude vs. Random vs. Noise Floor)...")
    try:
        run_pruning_strategy_comparison_main(output_dir=dirs["results_pruning_strategy"])
        print("[+] Stage 9 Complete.")
    except Exception as err:
        print(f"[!] Stage 9 Failed: {err}")
        stage_failures.append(("Stage 9 (Pruning Strategy Comparison)", err))

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
        description="Master Orchestrator for Multimodal Semantic Dementia CLIP Pipeline."
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--target_n",
        type=int,
        default=None,
        help="Explicit total sample count override",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Disable taxonomic class balancing",
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
        sample_frac=args.sample_frac,
        target_n=args.target_n,
        balance_taxonomically=not args.no_balance,
        skip_indexing=args.skip_indexing,
        output_dir=args.output_dir,
    )