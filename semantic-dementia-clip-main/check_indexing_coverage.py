import pandas as pd
from src.curated_config import CURATED_CLASSES

df = pd.read_csv("data/processed/metadata_processed.csv")
present = set(df["specific"].unique())
missing = set(CURATED_CLASSES) - present
counts = df[df["specific"].isin(CURATED_CLASSES)]["specific"].value_counts()

print(f"Total classes indexed: {df['specific'].nunique()}")
print(f"Curated classes present: {len(CURATED_CLASSES) - len(missing)}/{len(CURATED_CLASSES)}")
if missing:
    print("MISSING classes:", missing)
else:
    print("NONE MISSING -- all 33 curated classes are indexed.")
print()
print("Per-class counts (need >=5 each):")
print(counts.sort_values())
