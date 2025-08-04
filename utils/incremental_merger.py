#!/usr/bin/env python3
"""
Incremental batch merger that creates a single dataset file without overloading RAM.
Save as: utils/incremental_merger.py
"""

import pickle
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import the encodedSample class BEFORE loading any pickles
from utils.dataclass import encodedSample


class IncrementalBatchMerger:
    """Merge batches incrementally to avoid RAM overflow."""

    def __init__(self, dataset_dir: Path, orientation: str, chunk_size: int = 10000):
        """
        Initialize the merger.

        Args:
            dataset_dir: Base directory containing the dataset
            orientation: Orientation (TRA, COR, SAG)
            chunk_size: Number of samples to process at once (adjust based on RAM)
        """
        self.dataset_dir = dataset_dir
        self.orientation = orientation
        self.chunk_size = chunk_size
        self.batch_dir = dataset_dir / orientation / "batches"

    def check_or_create_merged_dataset(self) -> Tuple[Path, Dict[str, List[int]]]:
        """
        Check if merged dataset exists, if not create it incrementally.
        Returns path to dataset and class indices mapping.
        """
        merged_file = self.dataset_dir / f"dataset_{self.orientation}.pkl"
        class_indices_file = self.dataset_dir / f"class_indices_{self.orientation}.pkl"

        if merged_file.exists() and class_indices_file.exists():
            print(f"Merged dataset already exists: {merged_file}")
            with open(class_indices_file, 'rb') as f:
                class_indices = pickle.load(f)
            return merged_file, class_indices

        print(f"Creating merged dataset for {self.orientation}...")

        # First pass: collect class information without loading full data
        class_indices = self._collect_class_indices()

        # Second pass: merge batches incrementally
        self._merge_batches_incrementally(merged_file, class_indices_file, class_indices)

        return merged_file, class_indices

    def _collect_class_indices(self) -> Dict[str, List[int]]:
        """First pass: scan all batches to collect class indices."""
        batch_files = sorted(self.batch_dir.glob("dataset_batch_*.pkl"))

        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.batch_dir}")

        class_indices = defaultdict(list)
        global_idx = 0

        print(f"Scanning {len(batch_files)} batch files to collect class information...")

        for batch_file in tqdm(batch_files, desc="Scanning batches"):
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)

                for local_idx, sample in enumerate(batch_data):
                    # Convert label to string for consistency
                    label_str = str(sample.label)
                    class_indices[label_str].append(global_idx)
                    global_idx += 1

                # Free memory immediately
                del batch_data
                gc.collect()

            except Exception as e:
                print(f"Error processing {batch_file}: {e}")
                continue

        # Convert to regular dict and sort by class name
        class_indices = dict(sorted(class_indices.items()))

        print(f"\nClass distribution:")
        print(f"Total classes: {len(class_indices)}")
        print(f"Total samples: {global_idx}")

        # Show sample distribution
        for class_name, indices in list(class_indices.items())[:5]:
            print(f"  {class_name}: {len(indices)} samples")
        if len(class_indices) > 5:
            print(f"  ... and {len(class_indices) - 5} more classes")

        return class_indices

    def _merge_batches_incrementally(self, output_file: Path, indices_file: Path,
                                     class_indices: Dict[str, List[int]]):
        """Second pass: merge batches into single file incrementally."""
        batch_files = sorted(self.batch_dir.glob("dataset_batch_*.pkl"))

        # Create temporary file for safe writing
        temp_file = output_file.with_suffix('.tmp')

        # Initialize empty list in file
        with open(temp_file, 'wb') as f:
            pickle.dump([], f, protocol=pickle.HIGHEST_PROTOCOL)

        # Process in chunks
        all_samples = []
        total_processed = 0

        print(f"\nMerging {len(batch_files)} batch files incrementally...")

        for batch_idx, batch_file in enumerate(tqdm(batch_files, desc="Merging batches")):
            try:
                # Load batch
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)

                # Add to current chunk
                all_samples.extend(batch_data)
                total_processed += len(batch_data)

                # Free batch data
                del batch_data

                # Write chunk when reaching chunk size or last batch
                if len(all_samples) >= self.chunk_size or batch_idx == len(batch_files) - 1:
                    self._append_to_file(temp_file, all_samples)

                    print(f"  Processed {total_processed} samples so far...")

                    # Clear current chunk
                    all_samples = []
                    gc.collect()

            except Exception as e:
                print(f"Error processing {batch_file}: {e}")
                continue

        print(f"Total samples merged: {total_processed}")

        # Finalize the file
        self._finalize_merged_file(temp_file, output_file)

        # Save class indices
        with open(indices_file, 'wb') as f:
            pickle.dump(class_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nMerged dataset saved to: {output_file}")
        print(f"Class indices saved to: {indices_file}")

    def _append_to_file(self, file_path: Path, new_samples: List):
        """Append samples to existing pickle file."""
        # Load existing data
        with open(file_path, 'rb') as f:
            existing = pickle.load(f)

        # Append new samples
        existing.extend(new_samples)

        # Save back
        with open(file_path, 'wb') as f:
            pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Free memory
        del existing
        gc.collect()

    def _finalize_merged_file(self, temp_file: Path, final_file: Path):
        """Move temporary file to final location."""
        if final_file.exists():
            print(f"Removing existing file: {final_file}")
            final_file.unlink()

        temp_file.rename(final_file)
        print(f"Finalized merged file: {final_file}")


def merge_all_orientations(dataset_dir: Path, orientations: List[str] = ["TRA", "COR", "SAG"],
                           chunk_size: int = 10000):
    """
    Merge batches for all orientations.

    Args:
        dataset_dir: Base dataset directory
        orientations: List of orientations to process
        chunk_size: Chunk size for incremental processing
    """
    for orientation in orientations:
        print(f"\n{'=' * 60}")
        print(f"Processing orientation: {orientation}")
        print(f"{'=' * 60}")

        try:
            merger = IncrementalBatchMerger(dataset_dir, orientation, chunk_size)
            merged_file, class_indices = merger.check_or_create_merged_dataset()

            # Verify the merged file
            print(f"\nVerifying merged file...")
            with open(merged_file, 'rb') as f:
                dataset = pickle.load(f)
                print(f"Merged dataset contains {len(dataset)} samples")

                # Show first few samples
                print("\nFirst 3 samples:")
                for i, sample in enumerate(dataset[:3]):
                    print(f"  Sample {i}: label={sample.label}, "
                          f"img_shape={sample.img.shape if hasattr(sample, 'img') else 'N/A'}")

                del dataset
                gc.collect()

        except Exception as e:
            print(f"Error processing {orientation}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Merge dataset batches incrementally")
    parser.add_argument("--dataset-dir", type=str,
                        default="/media/e210/portable_hdd/d_bodypart_final_cleaned",
                        help="Base dataset directory")
    parser.add_argument("--orientations", nargs="+", default=["TRA", "COR", "SAG"],
                        help="Orientations to process")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of samples to process at once")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        exit(1)

    merge_all_orientations(dataset_dir, args.orientations, args.chunk_size)