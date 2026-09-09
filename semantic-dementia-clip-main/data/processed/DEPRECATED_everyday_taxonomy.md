# Deprecated: everyday_taxonomy.csv / everyday_taxonomy.json

These two files were a 28-class, Living+Non-Living taxonomy built by an
earlier version of `build_general_dataset.py`, but nothing in the live
pipeline (`run_full_pipeline.py`) ever actually read them -- only
`tests/test_everday_taxonomy.py` referenced them, purely to check they
existed and were well-formed.

They have been superseded by `data/taxonomy_full.json` (77 classes, both
domains, 23 basic-level coordinate groups), which is now the single source
of truth loaded by both `src/dataset_downloader.py` and
`src/build_general_dataset.py`. The schema test has been repointed at
`data/taxonomy_full.json` (see `tests/test_taxonomy_full.py`).

These two files are kept here only as a historical/legacy artifact. Do not
build new functionality against them.
