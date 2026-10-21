import io
import json
import os
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Single source of truth for class definitions -- loaded from
# data/taxonomy_full.json rather than a second hardcoded copy of the
# taxonomy dict (this file and dataset_downloader.py previously each
# maintained their own independent copy, which is exactly the kind of
# duplication that let them silently drift apart; both now load the same
# file). See dataset_downloader.py for the full provenance note. The old
# "Directive 1.1" living-only restriction has been lifted -- this taxonomy
# spans both Living and Non-Living domains.
from src.dataset_downloader import load_taxonomy, TAXONOMY_PATH

TINY_IMAGENET_DRAWABLE_TAXONOMY = load_taxonomy(TAXONOMY_PATH)


def extract_dataset(zip_path, extract_dir, max_images_per_class=1000):
    os.makedirs(extract_dir, exist_ok=True)
    raw_images_dir = os.path.join(extract_dir, "raw")
    os.makedirs(raw_images_dir, exist_ok=True)

    extracted_records = []

    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()

        found_classes = [
            wnid
            for wnid in TINY_IMAGENET_DRAWABLE_TAXONOMY
            if any(
                f"/train/{wnid}/" in f or f"/{wnid}/" in f for f in all_files
            )
        ]
        print(
            f"[*] Found {len(found_classes)} / {len(TINY_IMAGENET_DRAWABLE_TAXONOMY)} target classes in zip archive."
        )

        for wnid in tqdm(found_classes, desc="Extracting Images"):
            tax_info = TINY_IMAGENET_DRAWABLE_TAXONOMY[wnid]
            class_files = [
                f
                for f in all_files
                if f"/{wnid}/images/" in f and f.endswith(".JPEG")
            ][:max_images_per_class]

            for idx, file_path in enumerate(class_files):
                img_data = z.read(file_path)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

                clean_name = tax_info["specific"].lower().replace(" ", "_")
                local_filename = f"{clean_name}_{wnid}_{idx}.jpg"
                save_path = os.path.join(raw_images_dir, local_filename)
                img.save(save_path)

                # Single word prompt template
                prompt_text = tax_info["specific"].lower()

                extracted_records.append(
                    {
                        "filepath": save_path,
                        "filename": local_filename,
                        "wnid": wnid,
                        "specific": tax_info["specific"],
                        "coordinate": tax_info["coordinate"],
                        "superordinate": tax_info["superordinate"],
                        "domain": tax_info["domain"],
                        "typicality": tax_info["typicality"],
                        "prompt_text": prompt_text,
                    }
                )

    return extracted_records


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        fallback_zip = os.path.join(data_dir, "raw", "tiny-imagenet-200.zip")
        if os.path.exists(fallback_zip):
            zip_path = fallback_zip
        else:
            raise FileNotFoundError(
                f"[!] Could not locate tiny-imagenet-200.zip at {zip_path}"
            )

    print(f"[*] Extracting dataset from: {zip_path}")
    records = extract_dataset(zip_path, data_dir, max_images_per_class=100)

    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "processed", "metadata_processed.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # NOTE: previously re-wrote data/taxonomy_drawable_32.json here on every
    # run -- redundant now that the taxonomy is loaded from (and stays in)
    # data/taxonomy_full.json, and misleading besides, since that filename
    # implies a fixed 32-class scope this taxonomy no longer has.

    print(
        f"\n[+] SUCCESS: Extracted {len(records)} images across {df['wnid'].nunique()} curated classes."
    )
    print(f"[+] Metadata saved to: {csv_path}")


if __name__ == "__main__":
    main()