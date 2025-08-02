#!/usr/bin/env python3
"""
Simple direct merge script. Run this from your project root.
Save as: merge_simple.py
"""

# First, import the encodedSample class BEFORE loading any pickles
from utils.dataclass import encodedSample
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import gc


def merge_orientation(input_dir: Path, output_dir: Path, orientation: str):
    """Merge batches for a single orientation."""

    print(f"\nProcessing {orientation}...")
    batches_dir = input_dir / orientation / "batches"

    if not batches_dir.exists():
        print(f"  Batches directory not found: {batches_dir}")
        return

    batch_files = sorted(batches_dir.glob("dataset_batch_*.pkl"))
    print(f"  Found {len(batch_files)} batch files")

    if not batch_files:
        return

    # Load all batches
    full_dataset = []
    for batch_file in tqdm(batch_files, desc=f"  Loading {orientation}"):
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                full_dataset.extend(batch_data)
        except Exception as e:
            print(f"\n  Error with {batch_file.name}: {e}")

    print(f"  Loaded {len(full_dataset)} samples")

    if full_dataset:
        # Save merged dataset
        output_file = output_dir / f"dataset_{orientation}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(full_dataset, f)
        print(f"  Saved to {output_file}")

        # Copy other files
        for filename in [f"label_dict_{orientation}.json", f"dataset_stats_{orientation}.json"]:
            src = input_dir / orientation / filename
            if src.exists():
                import shutil
                shutil.copy(src, output_dir / filename)
                print(f"  Copied {filename}")


def main():
    input_dir = Path("/media/e210/portable_hdd/d_bodypart_final_cleaned")
    output_dir = Path("/media/e210/portable_hdd/d_bodypart_final_cleaned")

    output_dir.mkdir(exist_ok=True)

    for orientation in ["TRA", "COR", "SAG"]:
        merge_orientation(input_dir, output_dir, orientation)

    print("\nDone!")


if __name__ == "__main__":
    main()