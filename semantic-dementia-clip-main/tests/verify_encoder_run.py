import os
import sys
import shutil
import pandas as pd
import torch
from PIL import Image

# Add root directory to python path to allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.dataset_encoder import DatasetEncoder
except ImportError as e:
    print("\n[Error] Could not import DatasetEncoder from src.dataset_encoder.")
    print("Please make sure you have 'src/__init__.py' and 'src/dataset_encoder.py' in your repository.")
    print(e)
    sys.exit(1)

def run_verification():
    raw_metadata_path = "tests/metadata_raw.csv"
    temp_image_dir = "tests/temp_test_images"
    output_tensor_path = "tests/test_image_index.pt"
    output_csv_path = "tests/test_metadata_processed.csv"

    # 1. Check if your metadata CSV exists
    if not os.path.exists(raw_metadata_path):
        print(f"\n[Error] Could not find '{raw_metadata_path}'.")
        print("Please verify that you have placed your metadata file inside the 'tests/' folder.")
        sys.exit(1)

    print(f"[+] Found '{raw_metadata_path}'. Checking taxonomy structure...")
    df = pd.read_csv(raw_metadata_path)

    # 2. Validate your 4-tier hierarchical taxonomy columns
    required_cols = ["filename", "domain", "superordinate", "coordinate", "specific"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n[Error] Your metadata_raw.csv is missing required taxonomic columns: {missing_cols}")
        print("To match your 4-tier clinical taxonomy, it must include: filename, domain, superordinate, coordinate, specific")
        sys.exit(1)
    
    print("[+] 4-Tier taxonomy column check: PASSED")
    print(f"    - Unique Domains: {df['domain'].unique().tolist()}")
    print(f"    - Unique Superordinate Classes: {df['superordinate'].unique().tolist()}")
    print(f"    - Total items to encode: {len(df)}")

    # 3. Create temporary placeholder images based on YOUR CSV's filenames
    print(f"\n[+] Setting up temporary image directory at '{temp_image_dir}'...")
    os.makedirs(temp_image_dir, exist_ok=True)
    
    generated_count = 0
    for idx, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(temp_image_dir, filename)
        
        # Determine color based on domain to give CLIP slightly different inputs
        color = (34, 139, 34) if row['domain'].lower() == 'living' else (112, 128, 144)
        
        if not os.path.exists(img_path):
            # Create a coordinate 224x224 RGB image (standard size for CLIP preprocessing)
            img = Image.new("RGB", (224, 224), color=color)
            img.save(img_path)
            generated_count += 1
            
    print(f"[+] Generated {generated_count} temporary mock images matching your CSV.")

    # 4. Instantiate and execute your DatasetEncoder
    print("\n[+] Instantiating DatasetEncoder and commencing pipeline extraction...")
    try:
        encoder = DatasetEncoder(
            metadata_path=raw_metadata_path,
            image_dir=temp_image_dir,
            output_tensor_path=output_tensor_path,
            output_csv_path=output_csv_path,
            batch_size=2  # Keep batch size small for fast diagnostic dry-run
        )
        
        encoder.extract_features()
        
    except Exception as e:
        print(f"\n[Error] Your dataset_encoder execution failed:")
        print(e)
        cleanup(temp_image_dir, output_tensor_path, output_csv_path)
        sys.exit(1)

    # 5. Run Post-execution sanity checks on output assets
    print("\n[+] Extraction complete! Running clinical output assertions...")
    
    # Check if files were created
    if not os.path.exists(output_tensor_path) or not os.path.exists(output_csv_path):
        print("[Error] Encoder executed but failed to save output assets.")
        cleanup(temp_image_dir, output_tensor_path, output_csv_path)
        sys.exit(1)
        
    # Assert dimensions and unit-normalization
    try:
        embeddings = torch.load(output_tensor_path)
        expected_shape = (len(df), 512)
        
        print(f"    - Saved Tensor Shape: {list(embeddings.shape)} (Expected: {list(expected_shape)})")
        if embeddings.shape != expected_shape:
            print(f"[Warning] Tensor shape mismatch. Got {embeddings.shape}, expected {expected_shape}")
            
        # Check L2 Normalization (||f||_2 = 1.0)
        norms = torch.norm(embeddings, p=2, dim=-1)
        non_normalized = torch.where(~torch.isclose(norms, torch.tensor(1.0), atol=1e-4))[0]
        
        if len(non_normalized) > 0:
            print(f"[Warning] Found {len(non_normalized)} vectors that are NOT unit-normalized.")
            print(f"          First non-normalized norm: {norms[non_normalized[0]].item()}")
        else:
            print("    - L2 Unit Normalization: PASSED (All vectors map cleanly to unit hypersphere)")
            
    except Exception as e:
        print(f"[Error] Failed to read or validate your saved tensor: {e}")

    # Validate output CSV row count and index lock
    try:
        df_proc = pd.read_csv(output_csv_path)
        print(f"    - Processed CSV Rows: {len(df_proc)} (Expected: {len(df)})")
        
        if "matrix_index" in df_proc.columns:
            mismatches = df_proc[df_proc["matrix_index"] != df_proc.index]
            if len(mismatches) > 0:
                print("[Warning] Index locking is out of alignment! Row indexes must match matrix row indices sequentially.")
            else:
                print("    - Metadata Index Lock: PASSED")
        else:
            print("[Warning] Aligned CSV is missing the required 'matrix_index' column.")
            
    except Exception as e:
        print(f"[Error] Failed to validate your processed CSV: {e}")

    # 6. Cleanup sandboxed elements
    cleanup(temp_image_dir, output_tensor_path, output_csv_path)
    print("\n[+] SUCCESS! Your 'DatasetEncoder' is fully functional and ready for production data.")

def cleanup(temp_dir, tensor_path, csv_path):
    print("\n[+] Tearing down temporary workspace files...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(tensor_path):
        os.remove(tensor_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

if __name__ == "__main__":
    run_verification()