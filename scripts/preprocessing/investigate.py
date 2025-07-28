#!/usr/bin/env python3
"""
Script to investigate the actual text content and tokenization in your dataset.
This will help us understand why sequences have different lengths despite padding="longest".
"""

import pickle
import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import numpy as np


def investigate_dataset(dataset_path: str, num_samples: int = 10):
    """
    Investigate the dataset to understand sequence length variations.

    Args:
        dataset_path: Path to the dataset pickle file
        num_samples: Number of shortest/longest samples to examine
    """

    print("=" * 80)
    print("DATASET INVESTIGATION")
    print("=" * 80)

    # Load dataset
    print("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Get sequence lengths
    lengths = [(i, len(sample.input_ids), len(sample.attention_mask))
               for i, sample in enumerate(dataset)]

    # Check if input_ids and attention_mask lengths match
    mismatched = [(i, id_len, mask_len) for i, id_len, mask_len in lengths if id_len != mask_len]
    if mismatched:
        print(f"\nWARNING: Found {len(mismatched)} samples where input_ids and attention_mask lengths don't match!")
        for i, id_len, mask_len in mismatched[:5]:
            print(f"  Sample {i}: input_ids={id_len}, attention_mask={mask_len}")

    # Sort by input_ids length
    lengths_sorted = sorted(lengths, key=lambda x: x[1])

    # Get unique lengths
    input_id_lengths = [x[1] for x in lengths]
    length_counts = Counter(input_id_lengths)

    print(f"\nSequence Length Distribution:")
    print(f"Unique lengths: {len(length_counts)}")
    print(f"Min length: {min(input_id_lengths)}")
    print(f"Max length: {max(input_id_lengths)}")
    print(f"Average length: {np.mean(input_id_lengths):.1f}")

    # Show most common lengths
    print(f"\nMost common lengths:")
    for length, count in length_counts.most_common(10):
        print(f"  Length {length}: {count} samples ({count / len(dataset) * 100:.1f}%)")

    # Initialize tokenizer to decode sequences
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print("\n" + "=" * 80)
    print(f"EXAMINING {num_samples} SHORTEST SAMPLES")
    print("=" * 80)

    # Examine shortest samples
    shortest_samples = lengths_sorted[:num_samples]
    for rank, (idx, length, mask_len) in enumerate(shortest_samples):
        sample = dataset[idx]
        print(f"\n[SHORTEST #{rank + 1}] Sample {idx} - Length: {length}")
        print("-" * 50)

        # Decode the tokens to see the actual text
        decoded_text = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
        print(f"Decoded text:")
        print(f"'{decoded_text}'")

        # Show token IDs
        print(f"\nToken IDs: {sample.input_ids[:20]}...")  # First 20 tokens
        print(f"Attention mask: {sample.attention_mask[:20]}...")  # First 20 tokens

        # Check for padding tokens
        pad_token_id = tokenizer.pad_token_id or 0
        num_pad_tokens = sample.input_ids.count(pad_token_id)
        print(f"Number of padding tokens (ID {pad_token_id}): {num_pad_tokens}")

        # Show where padding starts
        try:
            first_pad_idx = sample.input_ids.index(pad_token_id)
            print(f"Padding starts at index: {first_pad_idx}")
        except ValueError:
            print("No padding tokens found")

        # Show the label
        print(f"Label: {sample.label}")
        print(f"Numerical attributes: {sample.numerical_attributes}")

    print("\n" + "=" * 80)
    print(f"EXAMINING {num_samples} LONGEST SAMPLES")
    print("=" * 80)

    # Examine longest samples
    longest_samples = lengths_sorted[-num_samples:]
    for rank, (idx, length, mask_len) in enumerate(longest_samples):
        sample = dataset[idx]
        print(f"\n[LONGEST #{rank + 1}] Sample {idx} - Length: {length}")
        print("-" * 50)

        # Decode the tokens to see the actual text
        decoded_text = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
        print(f"Decoded text:")
        print(f"'{decoded_text}'")

        # Show token IDs
        print(f"\nToken IDs: {sample.input_ids[:20]}...")  # First 20 tokens
        print(f"Attention mask: {sample.attention_mask[:20]}...")  # First 20 tokens

        # Check for padding tokens
        pad_token_id = tokenizer.pad_token_id or 0
        num_pad_tokens = sample.input_ids.count(pad_token_id)
        print(f"Number of padding tokens (ID {pad_token_id}): {num_pad_tokens}")

        # Show where padding starts
        try:
            first_pad_idx = sample.input_ids.index(pad_token_id)
            print(f"Padding starts at index: {first_pad_idx}")
        except ValueError:
            print("No padding tokens found")

        # Show the label
        print(f"Label: {sample.label}")
        print(f"Numerical attributes: {sample.numerical_attributes}")

    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES TO CHECK")
    print("=" * 80)

    # Check if this could be a batched processing issue
    if len(length_counts) > 1:
        print("❌ FOUND VARIABLE LENGTHS - This shouldn't happen with padding='longest'")
        print("\nPossible causes:")
        print("1. Dataset was processed in batches with different max lengths")
        print("2. Dataset was concatenated from multiple preprocessing runs")
        print("3. Post-processing modified the sequences")
        print("4. There's a bug in the preprocessing tokenization")

        # Check if lengths follow a pattern that suggests batch processing
        unique_lengths = sorted(length_counts.keys())
        print(f"\nUnique lengths: {unique_lengths}")

        # Check if sequences cluster around certain lengths
        length_gaps = [unique_lengths[i + 1] - unique_lengths[i] for i in range(len(unique_lengths) - 1)]
        print(f"Gaps between lengths: {length_gaps}")

    else:
        print("✅ All sequences have the same length - this is expected")

    # Check data types
    sample = dataset[0]
    print(f"\nData types in sample:")
    print(f"  input_ids: {type(sample.input_ids)} (length: {len(sample.input_ids)})")
    print(f"  attention_mask: {type(sample.attention_mask)} (length: {len(sample.attention_mask)})")
    print(f"  numerical_attributes: {type(sample.numerical_attributes)}")
    print(f"  label: {type(sample.label)}")


def compare_mri_ct_samples(mri_path: str, ct_path: str):
    """Compare a few samples from MRI and CT datasets to see the differences."""

    print("\n" + "=" * 80)
    print("COMPARING MRI vs CT SAMPLES")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load both datasets
    with open(mri_path, 'rb') as f:
        mri_dataset = pickle.load(f)

    with open(ct_path, 'rb') as f:
        ct_dataset = pickle.load(f)

    print(f"MRI dataset: {len(mri_dataset)} samples")
    print(f"CT dataset: {len(ct_dataset)} samples")

    # Compare first 3 samples from each
    for i in range(min(3, len(mri_dataset), len(ct_dataset))):
        print(f"\n{'-' * 60}")
        print(f"SAMPLE {i + 1} COMPARISON")
        print(f"{'-' * 60}")

        mri_sample = mri_dataset[i]
        ct_sample = ct_dataset[i]

        print(f"\nMRI Sample {i}:")
        print(f"  Length: {len(mri_sample.input_ids)}")
        mri_text = tokenizer.decode(mri_sample.input_ids, skip_special_tokens=True)
        print(f"  Text: '{mri_text[:200]}...'")

        print(f"\nCT Sample {i}:")
        print(f"  Length: {len(ct_sample.input_ids)}")
        ct_text = tokenizer.decode(ct_sample.input_ids, skip_special_tokens=True)
        print(f"  Text: '{ct_text[:200]}...'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Investigate dataset sequence lengths")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset pickle file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of shortest/longest samples to examine")
    parser.add_argument("--compare_with", type=str,
                        help="Path to another dataset to compare with")

    args = parser.parse_args()

    investigate_dataset(args.dataset_path, args.num_samples)

    if args.compare_with:
        compare_mri_ct_samples(args.compare_with, args.dataset_path)