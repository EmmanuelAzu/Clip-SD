import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.large_pool_validation import run_large_pool_validation


def main(output_dir=None, scenario="joint"):
    print("=" * 60)
    print(" LARGE POOL VALIDATION (all downloaded classes as candidates) ")
    print("=" * 60)
    results = run_large_pool_validation(output_dir=output_dir, scenario=scenario)
    print("\n[+] Done:")
    for k, v in results.items():
        print(f"    {k}: {v}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Bozeat + hierarchical retrieval against the full ~72-class pool.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="joint", choices=["joint", "vision_only", "text_only"])
    args = parser.parse_args()
    main(output_dir=args.output_dir, scenario=args.scenario)
