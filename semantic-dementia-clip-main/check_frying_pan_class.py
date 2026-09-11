"""Checks whether the images labeled 'Frying Pan' in the dataset actually
look like frying pans, or whether the wnid mapping in
data/taxonomy_full.json is wrong.
"""
import pandas as pd
import os

df = pd.read_csv("data/processed/metadata_processed.csv")
frying_pan_rows = df[df["specific"] == "Frying Pan"]

print(f"Found {len(frying_pan_rows)} rows labeled 'Frying Pan'")
print(f"wnid used: {frying_pan_rows['wnid'].unique()}")
print()
print("First 5 file paths -- open a few of these directly to check by eye:")
for _, row in frying_pan_rows.head(5).iterrows():
    path = row.get("filepath", row.get("filename", "?"))
    print(" ", path)

print()
print("For comparison, Teapot (same coordinate group, should look different"
      " from Frying Pan but also be a real kitchen object):")
teapot_rows = df[df["specific"] == "Teapot"]
print(f"wnid used: {teapot_rows['wnid'].unique()}")
for _, row in teapot_rows.head(3).iterrows():
    path = row.get("filepath", row.get("filename", "?"))
    print(" ", path)
