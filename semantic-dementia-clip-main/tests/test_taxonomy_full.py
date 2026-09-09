import json
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# NOTE: previously validated data/processed/everyday_taxonomy.csv/.json --
# a 28-class Living+Non-Living taxonomy that was built once (by
# build_general_dataset.py, in an earlier form) but never actually wired
# into the live pipeline (run_full_pipeline.py never reads it), leaving
# this test checking an orphaned artifact nothing else in the codebase
# depends on. data/taxonomy_full.json has since become the single,
# actually-consumed source of truth for class definitions (loaded by
# src/dataset_downloader.py and src/build_general_dataset.py), so this
# test now validates that file instead, where a broken taxonomy will
# actually be caught before it breaks the downloader/indexer.
TAXONOMY_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "taxonomy_full.json")

REQUIRED_FIELDS = {"domain", "superordinate", "coordinate", "specific", "typicality"}
VALID_DOMAINS = {"Living", "Non-Living"}
VALID_TYPICALITY = {"Typical", "Atypical"}


def test_taxonomy_full_schema():
    assert os.path.exists(TAXONOMY_JSON_PATH), f"[!] Missing taxonomy file: {TAXONOMY_JSON_PATH}"

    with open(TAXONOMY_JSON_PATH, "r") as f:
        taxonomy = json.load(f)

    assert len(taxonomy) > 0, "[!] Taxonomy is empty!"

    df = pd.DataFrame.from_dict(taxonomy, orient="index")
    df.index.name = "wnid"
    df = df.reset_index()

    missing_fields = REQUIRED_FIELDS - set(df.columns)
    assert not missing_fields, f"[!] Missing required fields: {missing_fields}"
    assert df.isnull().sum().sum() == 0, "[!] Found unexpected null values in taxonomy."

    bad_domains = set(df["domain"].unique()) - VALID_DOMAINS
    assert not bad_domains, f"[!] Unexpected domain value(s): {bad_domains}"

    bad_typicality = set(df["typicality"].unique()) - VALID_TYPICALITY
    assert not bad_typicality, f"[!] Unexpected typicality value(s): {bad_typicality}"

    # Every wnid must be unique (dict keys guarantee this) and every
    # (coordinate, specific) pair must be unique too, or two different
    # classes would collide under the same specific-level label.
    dupes = df[df.duplicated(subset=["specific"], keep=False)]
    assert dupes.empty, f"[!] Duplicate 'specific' labels found:\n{dupes}"

    # For the Typicality Delta metric (metrics.compute_typicality_delta) to
    # be computable at all, at least one coordinate group needs both a
    # Typical and an Atypical member.
    has_contrast = (
        df.groupby("coordinate")["typicality"].nunique().max() >= 2
    )
    assert has_contrast, (
        "[!] No coordinate group has both a Typical and an Atypical member -- "
        "the Typicality Delta metric could never produce a non-NaN result."
    )

    print("=" * 60)
    print("[OK] ALL TAXONOMY SCHEMA ASSERTIONS PASSED")
    print("=" * 60)
    print(f"Total Classes: {len(df)}")
    print("\nClass Counts by Domain:")
    print(df["domain"].value_counts().to_string())
    print("\nClass Counts by Superordinate:")
    print(df["superordinate"].value_counts().to_string())
    print(f"\nCoordinate (basic-level) groups: {df['coordinate'].nunique()}")
    print("\nTypicality Counts:")
    print(df["typicality"].value_counts().to_string())


if __name__ == "__main__":
    test_taxonomy_full_schema()
