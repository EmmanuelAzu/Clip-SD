import json
import os
import shutil
import urllib.request
import zipfile
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Single source of truth for class definitions. Previously this module (and
# build_general_dataset.py) each hardcoded their own separate copy of the
# taxonomy dict, which is exactly the kind of duplication that let them
# drift apart. Now both load from data/taxonomy_full.json.
#
# This taxonomy spans BOTH Living and Non-Living domains (the old
# "Directive 1.1" living-only restriction has been lifted -- see
# run_indexing.py) and is fine-grained: e.g. "Domestic Dogs" spans German
# Shepherd/Golden Retriever/Labrador Retriever/Chihuahua/Standard
# Poodle/Yorkshire Terrier as sibling specific classes, directly matching
# the "Golden Retriever vs Siberian Husky"-style RQ1 example from the
# proposal. Every wnid here is either already confirmed to resolve (5
# Living + 5 Non-Living classes already downloaded to data/images/) or was
# extracted directly from filenames already present in
# tests/metadata_raw.csv (itself the product of a prior successful
# Tiny-ImageNet-200 download), so none of these are guessed/unverified.
TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "taxonomy_full.json")


def load_taxonomy(path: str = TAXONOMY_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


class TinyImageNetCurationPipeline:
    """Downloads/curates images for every class in data/taxonomy_full.json.

    Idempotent and incremental: images already present in data/images/
    (whether from a prior run of this pipeline or manually placed) are
    picked up and included in the regenerated metadata without being
    re-copied; only genuinely missing classes trigger a fresh download.

    NOTE (network): actually calling download_dataset() requires internet
    access to fetch the ~120MB archive from Stanford's CS231n mirror. This
    class only prepares/curates data -- it does not run automatically as
    part of run_full_pipeline.py, since the master pipeline should not
    silently attempt a large network download. Run this script manually
    once, on a machine with network access, before running
    run_full_pipeline.py:

        python src/dataset_downloader.py

    Output: a regenerated tests/metadata_raw.csv covering every class that
    now has at least one image on disk (both previously-existing and
    newly-downloaded), which run_indexing.py (Stage 1) then consumes.
    """

    def __init__(self, workspace_dir: str = None, images_per_class: int = 200):
        self.workspace_dir = workspace_dir or os.path.join(PROJECT_ROOT, "data")
        self.zip_path = os.path.join(self.workspace_dir, "tiny-imagenet-200.zip")
        self.extract_dir = os.path.join(self.workspace_dir, "tiny-imagenet-200")
        self.raw_output_dir = os.path.join(self.workspace_dir, "images")
        # Canonical raw-metadata output location that run_indexing.py
        # actually looks for (previously this wrote to
        # "data/metadata_processed.csv" directly -- a THIRD, different path
        # from the "data/processed/metadata_processed.csv" that
        # run_indexing.py itself produces, meaning the two scripts'
        # outputs never actually lined up. Fixed to write raw per-image
        # metadata to tests/metadata_raw.csv, which is what run_indexing.py
        # is actually configured to read as its input.)
        self.metadata_path = os.path.join(PROJECT_ROOT, "tests", "metadata_raw.csv")
        self.taxonomy = load_taxonomy()
        self.images_per_class = images_per_class

        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.raw_output_dir, exist_ok=True)

    def download_dataset(self) -> None:
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        if not os.path.exists(self.zip_path):
            print("[*] Downloading Tiny ImageNet archive (~120MB)...")
            with (
                urllib.request.urlopen(url) as response,
                open(self.zip_path, "wb") as out_file,
            ):
                meta = response.info()
                content_length = meta.get("Content-Length")
                file_size = int(content_length) if content_length else None
                chunk_size = 1024 * 1024

                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    while True:
                        buffer = response.read(chunk_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        pbar.update(len(buffer))
            print("[+] Download complete.")
        else:
            print("[*] Found cached Tiny ImageNet archive. Skipping download.")

    def _scan_existing_images(self) -> list[dict]:
        """Finds images already present in data/images/ matching a known
        taxonomy class (filename pattern: "{specific_lower}_{wnid}_...").
        Lets a re-run pick up classes downloaded by a previous invocation,
        or manually placed images, without re-downloading anything.
        """
        records = []
        if not os.path.isdir(self.raw_output_dir):
            return records

        wnid_to_meta = self.taxonomy
        existing_files = os.listdir(self.raw_output_dir)
        for wnid, meta_info in wnid_to_meta.items():
            clean_name = meta_info["specific"].lower().replace(" ", "_")
            # Match by wnid substring (every filename contains "_{wnid}_"
            # regardless of what "specific" label was used at download
            # time) rather than by name prefix -- this way renaming a class
            # in the taxonomy (e.g. the generic "Cat" -> the taxonomically
            # correct "Tabby Cat" once egyptian_cat/persian_cat siblings
            # were added) doesn't orphan already-downloaded files whose
            # filenames were written under the old label.
            needle = f"_{wnid}_"
            matches = [f for f in existing_files if needle in f]
            for fname in matches:
                records.append(
                    {
                        "filename": fname,
                        "filepath": os.path.join("data", "images", fname),
                        "wnid": wnid,
                        "domain": meta_info["domain"],
                        "superordinate": meta_info["superordinate"],
                        "coordinate": meta_info["coordinate"],
                        "specific": meta_info["specific"],
                        "typicality": meta_info["typicality"],
                        "prompt_text": clean_name,
                    }
                )
        return records

    def extract_and_filter(self) -> str:
        existing_records = self._scan_existing_images()
        already_have = {r["wnid"] for r in existing_records}
        print(
            f"[*] Found {len(existing_records)} already-downloaded image(s) covering "
            f"{len(already_have)}/{len(self.taxonomy)} taxonomy classes."
        )

        missing_wnids = [w for w in self.taxonomy if w not in already_have]
        new_records = []

        if missing_wnids:
            if not os.path.exists(self.extract_dir):
                if not os.path.exists(self.zip_path):
                    print(
                        f"[!] {len(missing_wnids)} class(es) still need downloading, but "
                        f"the Tiny ImageNet archive isn't present. Run "
                        f"download_dataset() first (requires network access)."
                    )
                    missing_wnids = []
                else:
                    print("[*] Extracting zip file...")
                    with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                        zip_ref.extractall(self.workspace_dir)
                    print("[+] Extraction complete.")

            train_dir = os.path.join(self.extract_dir, "train")
            for wnid in tqdm(missing_wnids, desc="Curating missing classes"):
                meta_info = self.taxonomy[wnid]
                class_img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.exists(class_img_dir):
                    print(f"[!] Warning: Tiny-ImageNet directory for class {wnid} "
                          f"('{meta_info['specific']}') not found -- skipping.")
                    continue

                images = sorted(os.listdir(class_img_dir))[: self.images_per_class]
                clean_name = meta_info["specific"].lower().replace(" ", "_")
                for img_name in images:
                    src_img_path = os.path.join(class_img_dir, img_name)
                    new_filename = f"{clean_name}_{wnid}_{img_name}"
                    dest_img_path = os.path.join(self.raw_output_dir, new_filename)
                    shutil.copy(src_img_path, dest_img_path)
                    new_records.append(
                        {
                            "filename": new_filename,
                            "filepath": os.path.join("data", "images", new_filename),
                            "wnid": wnid,
                            "domain": meta_info["domain"],
                            "superordinate": meta_info["superordinate"],
                            "coordinate": meta_info["coordinate"],
                            "specific": meta_info["specific"],
                            "typicality": meta_info["typicality"],
                            "prompt_text": clean_name,
                        }
                    )

        all_records = existing_records + new_records
        df = pd.DataFrame(all_records)
        df.to_csv(self.metadata_path, index=False)
        print(f"\n[+] Curated {len(df)} total image(s) across "
              f"{df['wnid'].nunique() if len(df) else 0}/{len(self.taxonomy)} taxonomy classes.")
        print(f"[+] Raw metadata written to: {self.metadata_path}")
        print("[+] Next step: python run_indexing.py  (or run_full_pipeline.py, Stage 1)")

        if os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir)

        return self.metadata_path


if __name__ == "__main__":
    pipeline = TinyImageNetCurationPipeline()
    pipeline.download_dataset()
    pipeline.extract_and_filter()
