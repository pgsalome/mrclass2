#!/usr/bin/env python3
"""
Script to compare a working combined dataset with batch files to understand the difference.
Save this as: scripts/preprocessing/investigate.py
"""

import pickle
import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the encodedSample class
from utils.dataclass import encodedSample


def analyze_dataset(dataset_path: str, dataset_name: str = "Dataset"):
    """Analyze a dataset and return statistics."""

    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*60}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Total samples: {len(dataset)}")

    # Get sequence lengths
    lengths = [len(sample.input_ids) for sample in dataset]
    length_counts = Counter(lengths)

    # Get sample structure
    sample = dataset[0]

    stats = {
        'total_samples': len(dataset),
        'unique_lengths': len(length_counts),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'length_distribution': length_counts,
        'sample_type': type(sample).__name__,
        'has_orientation': hasattr(sample, 'orientation'),
        'input_ids_type': type(sample.input_ids),
        'first_5_lengths': lengths[:5]
    }

    # Print statistics
    print(f"Sequence length stats:")
    print(f"  - Unique lengths: {stats['unique_lengths']}")
    print(f"  - Min length: {stats['min_length']}")
    print(f"  - Max length: {stats['max_length']}")
    print(f"  - Average: {stats['avg_length']:.1f}")
    print(f"  - Std dev: {stats['std_length']:.1f}")

    print(f"\nSample structure:")
    print(f"  - Type: {stats['sample_type']}")
    print(f"  - Has orientation field: {stats['has_orientation']}")
    print(f"  - input_ids type: {stats['input_ids_type']}")

    # Show length distribution
    print(f"\nLength distribution (top 10):")
    for length, count in length_counts.most_common(10):
        print(f"  Length {length}: {count} samples ({count/len(dataset)*100:.1f}%)")

    # Show sample content
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"\nFirst sample:")
    print(f"  - Length: {len(sample.input_ids)}")
    print(f"  - Text: '{tokenizer.decode(sample.input_ids[:50])}...'")
    print(f"  - Label: {sample.label}")
    print(f"  - Numerical attrs: {sample.numerical_attributes}")

    return stats


def analyze_batches(batch_dir: str):
    """Analyze all batches in a directory."""

    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("dataset_batch_*.pkl"))

    if not batch_files:
        print(f"No batch files found in {batch_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Analyzing batches from: {batch_dir}")
    print(f"Found {len(batch_files)} batch files")
    print(f"{'='*60}")

    all_lengths = []
    all_samples = []
    batch_stats = []

    # Analyze each batch
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)

            lengths = [len(sample.input_ids) for sample in batch_data]
            all_lengths.extend(lengths)
            all_samples.extend(batch_data[:2])  # Keep first 2 samples from each batch for inspection

            batch_stats.append({
                'name': batch_file.name,
                'num_samples': len(batch_data),
                'unique_lengths': len(set(lengths)),
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0
            })

        except Exception as e:
            print(f"Error loading {batch_file}: {e}")

    # Overall statistics
    length_counts = Counter(all_lengths)

    stats = {
        'total_samples': len(all_lengths),
        'unique_lengths': len(length_counts),
        'min_length': min(all_lengths) if all_lengths else 0,
        'max_length': max(all_lengths) if all_lengths else 0,
        'avg_length': np.mean(all_lengths) if all_lengths else 0,
        'std_length': np.std(all_lengths) if all_lengths else 0,
        'length_distribution': length_counts,
        'sample_type': type(all_samples[0]).__name__ if all_samples else 'Unknown',
        'has_orientation': hasattr(all_samples[0], 'orientation') if all_samples else False,
        'input_ids_type': type(all_samples[0].input_ids) if all_samples else None,
        'batch_stats': batch_stats
    }

    print(f"\nOverall batch statistics:")
    print(f"  - Total samples: {stats['total_samples']}")
    print(f"  - Unique lengths: {stats['unique_lengths']}")
    print(f"  - Min length: {stats['min_length']}")
    print(f"  - Max length: {stats['max_length']}")
    print(f"  - Average: {stats['avg_length']:.1f}")
    print(f"  - Std dev: {stats['std_length']:.1f}")

    # Show problematic batches
    variable_batches = [b for b in batch_stats if b['unique_lengths'] > 1]
    if variable_batches:
        print(f"\nBatches with variable lengths: {len(variable_batches)}")
        for batch in variable_batches[:5]:
            print(f"  - {batch['name']}: {batch['unique_lengths']} unique lengths, range [{batch['min_length']}-{batch['max_length']}]")

    return stats


def compare_datasets(working_path: str, batch_dir: str):
    """Compare working dataset with batched dataset."""

    print("="*80)
    print("DATASET COMPARISON")
    print("="*80)

    # Analyze working dataset
    working_stats = analyze_dataset(working_path, "Working Combined Dataset")

    # Analyze batches
    batch_stats = analyze_batches(batch_dir)

    if not batch_stats:
        return

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    print("\nKey differences:")

    # Length uniformity
    print(f"\n1. Sequence length uniformity:")
    print(f"   Working dataset: {'UNIFORM' if working_stats['unique_lengths'] == 1 else 'VARIABLE'} ({working_stats['unique_lengths']} unique lengths)")
    print(f"   Batched dataset: {'UNIFORM' if batch_stats['unique_lengths'] == 1 else 'VARIABLE'} ({batch_stats['unique_lengths']} unique lengths)")

    # Length ranges
    print(f"\n2. Length ranges:")
    print(f"   Working: [{working_stats['min_length']} - {working_stats['max_length']}]")
    print(f"   Batched: [{batch_stats['min_length']} - {batch_stats['max_length']}]")

    # Data types
    print(f"\n3. Data types:")
    print(f"   Working input_ids type: {working_stats['input_ids_type']}")
    print(f"   Batched input_ids type: {batch_stats['input_ids_type']}")

    # Sample structure
    print(f"\n4. Sample structure:")
    print(f"   Working has 'orientation' field: {working_stats['has_orientation']}")
    print(f"   Batched has 'orientation' field: {batch_stats['has_orientation']}")

    # If working dataset is uniform but batched is not, show the target length
    if working_stats['unique_lengths'] == 1 and batch_stats['unique_lengths'] > 1:
        target_length = working_stats['min_length']  # Since all are same
        print(f"\n⚠️  ISSUE DETECTED:")
        print(f"   Working dataset has uniform length: {target_length}")
        print(f"   Batched dataset has variable lengths!")
        print(f"\n   SOLUTION: Pad all sequences in batches to length {target_length}")

        # Show which batches need fixing
        if 'batch_stats' in batch_stats:
            need_fixing = [b for b in batch_stats['batch_stats'] if b['unique_lengths'] > 1 or b['max_length'] != target_length]
            print(f"\n   Batches that need fixing: {len(need_fixing)}/{len(batch_stats['batch_stats'])}")





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare working dataset with batch files")
    parser.add_argument("--working", type=str, required=True,
                        help="Path to working combined dataset")
    parser.add_argument("--batch_dir", type=str, required=True,
                        help="Path to batch directory")

    args = parser.parse_args()

    # Compare datasets
    compare_datasets(args.working, args.batch_dir)