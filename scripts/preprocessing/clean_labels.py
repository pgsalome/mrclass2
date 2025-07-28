#!/usr/bin/env python3
"""
Script to fix variable-length sequences in an existing dataset.
This converts your existing dataset to fixed-length sequences without full reprocessing.
"""

import pickle
import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass

import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt


@dataclass
class encodedSample:
    """Class for holding encoded DICOM data after preprocessing."""
    img: any  # np.ndarray
    input_ids: List[int]
    attention_mask: List[int]
    numerical_attributes: List[float]
    label: int


def read_json_config(config_path):
    """
    Read JSON configuration file

    Args:
        config_path: Path to the JSON config file

    Returns:
        Dict containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert config dict to an object for dot access if desired
    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    return DotDict(config)


def normalize_class_name(label):
    """
    Normalize class names to handle variations in naming conventions.
    This function applies general, algorithmic normalization rules,
    not specific manual remappings like those handled earlier.

    Args:
        label: Original class label string (after manual fixes and pattern filtering).

    Returns:
        Normalized class label string.
    """
    # Make a copy of the original for comparison
    original = label

    # Handle all 4D sequences by removing the 4D prefix
    if label.startswith("4D") and not label.startswith("4DCCT") and not label.startswith("4DCT"):
        # Extract the base and modifiers
        parts = label.split("-")
        base = parts[0]  # 4DXXX

        # Remove the 4D prefix
        new_base = base[2:]  # Convert 4DMRA to MRA, 4DPER to PER, etc.

        # Keep the rest of the string intact
        if len(parts) > 1:
            return f"{new_base}-{'-'.join(parts[1:])}"
        else:
            return new_base

    # Handle FS and SE order standardization
    if "-FS-SE" in label:
        label = label.replace("-FS-SE", "-SE-FS")

    # Handle standardization of modifiers order
    modifiers = ["FS", "GE", "SE", "IRSE", "IRGE", "VFA", "MIP", "CE"]

    # First, extract the base type (before first hyphen)
    if "-" in label:
        base_type = label.split("-")[0]
        rest = label[len(base_type) + 1:]

        # Check for numbered modalities and preserve them
        if re.match(r"^[A-Z]+\d+", base_type):
            # It's something like T1, T2, CT1, etc.
            pass
        else:
            # Normalize the base type to uppercase
            base_type = base_type.upper()

        # Handle DWI b-values using glioma-specific clinically relevant groupings
        if base_type == "DWI" and rest.startswith("b") and not any(x in rest for x in ["ADC", "FA", "TRACE", "EADC"]):
            # Split into b-value and modifiers
            rest_parts = rest.split("-")
            b_value_str = rest_parts[0][1:]  # Remove the 'b' prefix
            modifiers_part = rest_parts[1:] if len(rest_parts) > 1 else []

            try:
                # Try to convert to float
                b_value = float(b_value_str)

                # Group into clinically relevant ranges for glioma imaging
                if b_value == 0:
                    new_b_value = "b0"  # Keep b=0 as its own category
                elif 1 <= b_value <= 75:
                    new_b_value = "b1-75"  # Very low b-values (includes b=50)
                elif 76 <= b_value <= 150:
                    new_b_value = "b76-150"  # Low b-values (includes b=100)
                elif 151 <= b_value <= 500:
                    new_b_value = "b151-500"  # Low-intermediate b-values
                elif 501 <= b_value <= 850:
                    new_b_value = "b501-850"  # Intermediate b-values
                elif 851 <= b_value <= 1050:
                    new_b_value = "b851-1050"  # Standard clinical b-values (includes b=1000)
                elif 1051 <= b_value <= 1450:
                    new_b_value = "b1051-1450"  # High b-values (includes b=1200)
                elif 1451 <= b_value <= 1950:
                    new_b_value = "b1451-1950"  # Very high b-values
                elif b_value > 1950:
                    new_b_value = "b1951plus"  # Ultra-high b-values
                else:
                    # If negative or otherwise unusual, keep the original
                    return original

                # Reconstruct the name with the new b-value group and the original modifiers
                if modifiers_part:
                    # Extract all modifiers
                    found_modifiers = []
                    remaining_parts = []

                    for part in modifiers_part:
                        if part in modifiers:
                            found_modifiers.append(part)
                        else:
                            remaining_parts.append(part)

                    # Sort modifiers to ensure consistent ordering
                    found_modifiers.sort()

                    # Rebuild with normalized format
                    if found_modifiers and remaining_parts:
                        return f"DWI-{new_b_value}-{'-'.join(remaining_parts)}-{'-'.join(found_modifiers)}"
                    elif found_modifiers:
                        return f"DWI-{new_b_value}-{'-'.join(found_modifiers)}"
                    elif remaining_parts:
                        return f"DWI-{new_b_value}-{'-'.join(remaining_parts)}"
                    else:
                        return f"DWI-{new_b_value}"
                else:
                    return f"DWI-{new_b_value}"
            except ValueError:
                # If conversion fails, keep the original
                return original

        # Extract all modifiers
        found_modifiers = []
        remaining_parts = []

        for part in rest.split("-"):
            if part in modifiers:
                found_modifiers.append(part)
            else:
                remaining_parts.append(part)

        # Sort modifiers to ensure consistent ordering
        found_modifiers.sort()

        # Rebuild the label
        if found_modifiers and remaining_parts:
            label = f"{base_type}-{'-'.join(remaining_parts)}-{'-'.join(found_modifiers)}"
        elif found_modifiers:
            label = f"{base_type}-{'-'.join(found_modifiers)}"
        elif remaining_parts:
            label = f"{base_type}-{'-'.join(remaining_parts)}"
        else:
            label = base_type

    return label


def process_dicom_dataset(dataset_path, label_dict_path, output_dir,
                          filter_patterns=None,
                          min_class_size=20,
                          min_samples=50,
                          do_bootstrap=True):
    """
    Complete pipeline to process DICOM dataset: filter, normalize, filter small classes, and bootstrap

    Args:
        dataset_path: Path to the original dataset pickle file
        label_dict_path: Path to the original label dictionary json file
        output_dir: Directory to save processed dataset
        filter_patterns: List of patterns to filter out classes (e.g., ['4DDWI', '4DCT'])
        min_class_size: Minimum class size to keep (classes with fewer samples will be removed)
        min_samples: Target minimum samples per class after bootstrapping
        do_bootstrap: Whether to perform bootstrapping or not
    """
    if filter_patterns is None:
        filter_patterns = ['4DDWI', '4DCT', '4DPT', '4DCDWI', '4DFMRI', '4DPER']

    print(f"Loading dataset from {dataset_path}")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"Reading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Load the label dictionary
    print(f"Reading label dictionary from: {label_dict_path}")
    with open(label_dict_path, 'r') as f:
        original_label_dict = json.load(f)

    # Create inverse mapping from numeric label to class name for the *initial* dataset
    inv_original_label_dict = {v: k for k, v in original_label_dict.items()}

    # Count samples per class before any processing (true original raw counts)
    if hasattr(dataset[0], 'label'):
        original_class_counts_raw = Counter([inv_original_label_dict[sample.label] for sample in dataset])
    else:
        original_class_counts_raw = Counter([inv_original_label_dict[sample['label']] for sample in dataset])

    # --- NEW STEP: Apply specific label merges and deletions ---
    print("\n" + "=" * 60)
    print("APPLYING SPECIFIC LABEL MERGES AND DELETIONS")
    print("=" * 60)

    manual_remapping_rules = {
        "PLV-EG": "PLV-LEG",  # Merge PLV-EG into PLV-LEG
        "SPINE": "WS",  # Rename SPINE to WS
        "ANPLS": "LS"  # Rename ANPLS to LS
    }
    labels_to_remove_completely = {"HNCANP", "NS"}

    temp_processed_dataset = []
    temp_processed_label_strings = []  # Stores string labels after manual fix
    deleted_sample_count = 0
    remapped_summary = Counter()

    print(f"Processing {len(dataset)} samples for manual label adjustments...")
    for sample in tqdm(dataset, desc="Applying manual label adjustments"):
        # Get the current label string from the original loaded data
        current_numeric_label = sample.label if hasattr(sample, 'label') else sample['label']
        current_label_str = inv_original_label_dict.get(current_numeric_label, "UNKNOWN_LABEL")

        # Check if the label should be removed
        if current_label_str in labels_to_remove_completely:
            deleted_sample_count += 1
            continue  # Skip this sample

        # Apply remapping rules
        if current_label_str in manual_remapping_rules:
            new_label_str = manual_remapping_rules[current_label_str]
            if new_label_str != current_label_str:  # Only count if actually remapped
                remapped_summary[f"'{current_label_str}' -> '{new_label_str}'"] += 1
            current_label_str = new_label_str  # Update to the remapped name

        # Append the sample and its (potentially new) label string
        temp_processed_dataset.append(sample)
        temp_processed_label_strings.append(current_label_str)

    print(f"Total samples before manual adjustments: {len(dataset)}")
    print(f"Samples deleted due to explicit removal: {deleted_sample_count}")
    print("Specific label remappings performed:")
    if remapped_summary:
        for k, v in remapped_summary.items():
            print(f"  - {k}: {v} samples")
    else:
        print("  - No specific remappings were applied.")

    # Reconstruct the label dictionary based on the labels remaining after manual adjustments
    # This also assigns new, contiguous numeric IDs
    unique_labels_after_manual_fix = sorted(list(set(temp_processed_label_strings)))
    current_label_dict = {name: i for i, name in enumerate(unique_labels_after_manual_fix)}
    current_inv_label_dict = {v: k for k, v in current_label_dict.items()}

    # Update numeric labels in the dataset based on the new dictionary
    dataset_after_manual_fix = []
    for i, sample in enumerate(tqdm(temp_processed_dataset, desc="Updating numeric labels after manual fix")):
        new_numeric_label = current_label_dict[temp_processed_label_strings[i]]
        if hasattr(sample, 'label'):
            sample.label = new_numeric_label
        else:
            sample['label'] = new_numeric_label
        dataset_after_manual_fix.append(sample)

    print(f"Total samples after manual adjustments: {len(dataset_after_manual_fix)}")
    print(f"Total unique classes after manual adjustments: {len(current_label_dict)}")

    # Count samples per class *after* manual fixes (this is the new "original" for subsequent steps)
    if hasattr(dataset_after_manual_fix[0], 'label'):
        initial_class_counts_for_pipeline = Counter(
            [current_inv_label_dict[sample.label] for sample in dataset_after_manual_fix])
    else:
        initial_class_counts_for_pipeline = Counter(
            [current_inv_label_dict[sample['label']] for sample in dataset_after_manual_fix])

    # Build `final_mapping_from_raw_to_normalized`:
    # This will map original_raw_name -> final_normalized_name (after STEP 2 and handling deletions)
    final_mapping_from_raw_to_normalized = {}
    for raw_name, raw_id in original_label_dict.items():
        current_name_temp = raw_name

        # Apply manual fixes (deletion/remapping)
        if current_name_temp in labels_to_remove_completely:
            final_mapping_from_raw_to_normalized[raw_name] = "_DELETED_MANUAL_"
            continue

        if current_name_temp in manual_remapping_rules:
            current_name_temp = manual_remapping_rules[current_name_temp]

        # Apply pattern filtering (check if the *remapped* name would be filtered)
        is_pattern_filtered_after_manual = False
        for pattern in filter_patterns:
            if pattern in current_name_temp:
                is_pattern_filtered_after_manual = True
                break
        if is_pattern_filtered_after_manual:
            final_mapping_from_raw_to_normalized[raw_name] = "_DELETED_PATTERN_"
            continue

        # Apply general normalization (normalize_class_name)
        final_normalized_name = normalize_class_name(current_name_temp)
        final_mapping_from_raw_to_normalized[raw_name] = final_normalized_name

    # STEP 1: FILTER OUT SPECIFIED PATTERNS (now operating on the manually fixed dataset)
    print("\n" + "=" * 60)
    print("STEP 1: FILTERING OUT SPECIFIED PATTERNS")
    print("=" * 60)

    pattern_filtered_classes = []
    # Use current_label_dict for pattern matching against *current* label names
    for pattern in filter_patterns:
        matching_classes = [cls for cls in current_label_dict.keys() if pattern in cls]
        if matching_classes:
            pattern_filtered_classes.extend(matching_classes)
            print(f"Found {len(matching_classes)} classes matching pattern '{pattern}'")

    pattern_filtered_classes = list(set(pattern_filtered_classes))

    print(f"Filtering out {len(pattern_filtered_classes)} classes matching specified patterns:")
    if pattern_filtered_classes:
        for cls in pattern_filtered_classes[:10]:
            print(f"  - {cls}")
        if len(pattern_filtered_classes) > 10:
            print(f"  - ... and {len(pattern_filtered_classes) - 10} more")
    else:
        print("  - No classes matched specified patterns for filtering.")

    filtered_dataset = []
    # Use current_inv_label_dict for lookup
    for sample in dataset_after_manual_fix:
        if hasattr(sample, 'label'):
            if current_inv_label_dict[sample.label] not in pattern_filtered_classes:
                filtered_dataset.append(sample)
        else:
            if current_inv_label_dict[sample['label']] not in pattern_filtered_classes:
                filtered_dataset.append(sample)

    print(f"Removed {len(dataset_after_manual_fix) - len(filtered_dataset)} samples from pattern-filtered classes")

    # STEP 2: NORMALIZE CLASS NAMES (general normalization)
    print("\n" + "=" * 60)
    print("STEP 2: NORMALIZING CLASS NAMES (General Rules)")
    print("=" * 60)

    if not filtered_dataset:
        print("No samples left after pattern filtering to normalize.")
        normalized_dataset = []
        normalized_counts = Counter()
        normalized_label_dict = {}
        normalization_mapping = {}
        after_pattern_filtering_counts = Counter()
    else:
        if hasattr(filtered_dataset[0], 'label'):
            remaining_class_names = set([current_inv_label_dict[sample.label] for sample in filtered_dataset])
        else:
            remaining_class_names = set([current_inv_label_dict[sample['label']] for sample in filtered_dataset])

        normalized_labels = {}  # Maps class name (after manual & pattern filter) -> class name (after general norm)
        for original_name_after_patt_filter in remaining_class_names:
            normalized_name_step2 = normalize_class_name(original_name_after_patt_filter)
            normalized_labels[original_name_after_patt_filter] = normalized_name_step2

        # Count class instances after pattern filtering but before general normalization
        after_pattern_filtering_counts = Counter()
        # Count class instances after general normalization
        normalized_counts = Counter()

        for sample in filtered_dataset:
            if hasattr(sample, 'label'):
                original_class_name_for_step2 = current_inv_label_dict[sample.label]
                normalized_class_name_step2 = normalized_labels[original_class_name_for_step2]

                after_pattern_filtering_counts[original_class_name_for_step2] += 1
                normalized_counts[normalized_class_name_step2] += 1
            else:
                original_class_name_for_step2 = current_inv_label_dict[sample['label']]
                normalized_class_name_step2 = normalized_labels[original_class_name_for_step2]

                after_pattern_filtering_counts[original_class_name_for_step2] += 1
                normalized_counts[normalized_class_name_step2] += 1

        # Create new label dictionary with normalized names (from step 2)
        normalized_label_dict = {name: i for i, name in enumerate(sorted(normalized_counts.keys()))}

        # Create mapping from old (current_label_dict, i.e. after manual fix) labels to new normalized labels
        current_to_normalized_id = {}
        for original_class_name_for_step2 in remaining_class_names:
            original_id = current_label_dict[original_class_name_for_step2]
            normalized_class_name_step2 = normalized_labels[original_class_name_for_step2]
            new_id = normalized_label_dict[normalized_class_name_step2]
            current_to_normalized_id[original_id] = new_id

        # Update dataset with normalized label IDs
        normalized_dataset = []
        for sample in tqdm(filtered_dataset, desc="Updating dataset with normalized labels"):
            if hasattr(sample, 'label'):
                sample.label = current_to_normalized_id[sample.label]
                normalized_dataset.append(sample)
            else:
                sample['label'] = current_to_normalized_id[sample['label']]
                normalized_dataset.append(sample)

        print(f"Classes after pattern filtering (before general norm): {len(after_pattern_filtering_counts)}")
        print(f"Classes after general normalization: {len(normalized_counts)}")
        print(
            f"Reduced by: {len(after_pattern_filtering_counts) - len(normalized_counts)} classes through general normalization")

        normalization_mapping = {}  # This maps labels *after pattern filtering* to *after general normalization*
        for original_name, norm_name in normalized_labels.items():
            if norm_name not in normalization_mapping:
                normalization_mapping[norm_name] = []
            normalization_mapping[norm_name].append(original_name)

        print("\nExamples of general normalization:")
        merged_examples = [(norm, originals) for norm, originals in normalization_mapping.items() if len(originals) > 1]
        merged_examples.sort(key=lambda x: len(x[1]), reverse=True)
        if merged_examples:
            for normalized, originals in merged_examples[:10]:
                print(f"  - {normalized}: merged from {len(originals)} classes")
                orig_examples = originals[:3]
                if len(originals) > 3:
                    orig_examples.append("...")
                print(f"    Examples: {orig_examples}")
        else:
            print("  - No classes were merged through general normalization.")

    # STEP 3: FILTER OUT SMALL CLASSES
    print(f"\n" + "=" * 60)
    print(f"STEP 3: FILTERING OUT CLASSES WITH FEWER THAN {min_class_size} SAMPLES")
    print("=" * 60)

    inv_normalized_dict = {v: k for k, v in normalized_label_dict.items()}

    small_classes = [cls for cls, count in normalized_counts.items() if count < min_class_size]

    print(f"Filtering out {len(small_classes)} small classes:")
    if small_classes:
        for cls in small_classes[:20]:
            print(f"  - {cls} ({normalized_counts[cls]} samples)")
        if len(small_classes) > 20:
            print(f"  - ... and {len(small_classes) - 20} more")
    else:
        print("  - No classes below minimum size found.")

    if not normalized_dataset:
        print("No samples left after general normalization to size filter.")
        size_filtered_dataset = []
        size_filtered_counts = Counter()
        filtered_label_dict = {}
    else:
        if hasattr(normalized_dataset[0], 'label'):
            size_filtered_dataset = [sample for sample in normalized_dataset
                                     if inv_normalized_dict[sample.label] not in small_classes]
        else:
            size_filtered_dataset = [sample for sample in normalized_dataset
                                     if inv_normalized_dict[sample['label']] not in small_classes]

        print(f"Removed {len(normalized_dataset) - len(size_filtered_dataset)} samples from small classes")

        if not size_filtered_dataset:
            print("No samples left after size filtering.")
            size_filtered_counts = Counter()
            filtered_label_dict = {}
        else:
            if hasattr(size_filtered_dataset[0], 'label'):
                size_filtered_counts = Counter([inv_normalized_dict[sample.label] for sample in size_filtered_dataset])
            else:
                size_filtered_counts = Counter(
                    [inv_normalized_dict[sample['label']] for sample in size_filtered_dataset])

            # Create new label dictionary without small classes, re-indexing for contiguity
            filtered_label_dict = {name: i for i, name in enumerate(sorted(size_filtered_counts.keys()))}

            # Update dataset with filtered label IDs to the new contiguous IDs
            normalized_to_filtered_id = {}
            for class_name in size_filtered_counts.keys():
                old_id = normalized_label_dict[class_name]
                new_id = filtered_label_dict[class_name]
                normalized_to_filtered_id[old_id] = new_id

            final_label_indexed_dataset = []  # Renamed for clarity on stage
            for sample in tqdm(size_filtered_dataset, desc="Re-indexing labels after size filtering"):
                if hasattr(sample, 'label'):
                    sample.label = normalized_to_filtered_id[sample.label]
                    final_label_indexed_dataset.append(sample)
                else:
                    sample['label'] = normalized_to_filtered_id[sample['label']]
                    final_label_indexed_dataset.append(sample)

            size_filtered_dataset = final_label_indexed_dataset  # Update the dataset reference

    # The `normalized_label_dict` now holds the final label mapping before bootstrapping
    # which is `filtered_label_dict` after size filtering.
    current_final_label_dict = filtered_label_dict

    # STEP 4: BOOTSTRAP UNDERREPRESENTED CLASSES
    print(f"\n" + "=" * 60)
    print(f"STEP 4: BOOTSTRAPPING CLASSES WITH FEWER THAN {min_samples} SAMPLES")
    print("=" * 60)

    if not size_filtered_dataset:
        print("No samples left after size filtering to bootstrap.")
        final_dataset = []
        final_counts = Counter()
    else:
        inv_current_final_dict = {v: k for k, v in current_final_label_dict.items()}

        classes_to_bootstrap = [cls for cls, count in size_filtered_counts.items() if count < min_samples]

        print(f"Bootstrapping {len(classes_to_bootstrap)} classes with fewer than {min_samples} samples:")
        if classes_to_bootstrap:
            for cls in classes_to_bootstrap[:20]:
                print(f"  - {cls} ({size_filtered_counts[cls]} samples)")
            if len(classes_to_bootstrap) > 20:
                print(f"  - ... and {len(classes_to_bootstrap) - 20} more")
        else:
            print("  - All classes meet minimum sample requirement, no bootstrapping needed.")

        class_indices = {}
        for i, sample in enumerate(size_filtered_dataset):
            if hasattr(sample, 'label'):
                class_name = inv_current_final_dict[sample.label]
            else:
                class_name = inv_current_final_dict[sample['label']]
            if class_name not in class_indices:
                class_indices[class_name] = []
            class_indices[class_name].append(i)

        bootstrapped_samples = []

        if do_bootstrap:
            for class_name in tqdm(classes_to_bootstrap, desc="Bootstrapping classes"):
                count = size_filtered_counts[class_name]
                if count >= min_samples:  # Should already be filtered out by `classes_to_bootstrap` but good for safety
                    continue
                indices = class_indices[class_name]
                samples_needed = min_samples - count
                for _ in range(samples_needed):
                    idx = random.choice(indices)
                    original_sample = size_filtered_dataset[idx]
                    if hasattr(original_sample, 'label'):
                        class_attrs = original_sample.__dict__.copy()
                        new_sample = type(original_sample)(**class_attrs)
                    else:
                        new_sample = original_sample.copy()

                    if hasattr(new_sample, 'numerical_attributes') and new_sample.numerical_attributes is not None:
                        noise = np.random.normal(0, 0.01, len(new_sample.numerical_attributes))
                        new_sample.numerical_attributes = [max(0, float(val) + n) for val, n in
                                                           zip(new_sample.numerical_attributes, noise)]
                    elif 'numerical_attributes' in new_sample and new_sample['numerical_attributes'] is not None:
                        noise = np.random.normal(0, 0.01, len(new_sample['numerical_attributes']))
                        new_sample['numerical_attributes'] = [max(0, float(val) + n) for val, n in
                                                              zip(new_sample['numerical_attributes'], noise)]
                    bootstrapped_samples.append(new_sample)
        else:
            print("Bootstrapping is disabled (`do_bootstrap=False`).")

        final_dataset = size_filtered_dataset + bootstrapped_samples

        if hasattr(final_dataset[0], 'label'):
            final_counts = Counter([inv_current_final_dict[sample.label] for sample in final_dataset])
        else:
            final_counts = Counter([inv_current_final_dict[sample['label']] for sample in final_dataset])

    # STEP 5: SAVE RESULTS AND GENERATE STATISTICS
    print(f"\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS AND GENERATING STATISTICS")
    print("=" * 60)

    dataset_filename = "bootstrapped_dataset.pkl"
    label_dict_filename = "bootstrapped_label_dict.json"

    # Save the processed dataset
    output_dataset_path = output_dir / dataset_filename
    with open(output_dataset_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    print(f"\nSaved processed dataset to: {output_dataset_path}")

    # Save the final label dictionary (after size filtering, before bootstrapping adds samples)
    output_label_dict_path = output_dir / label_dict_filename
    with open(output_label_dict_path, 'w') as f:
        json.dump(current_final_label_dict, f, indent=4)
    print(f"Saved final label dictionary to: {output_label_dict_path}")

    # Save the comprehensive map from raw original names to their final normalized names (after Step 2 and deletions)
    full_mapping_path = output_dir / "full_label_processing_map.json"
    with open(full_mapping_path, 'w') as f:
        json.dump(final_mapping_from_raw_to_normalized, f, indent=4)
    print(f"Saved full label processing map to: {full_mapping_path}")

    # Save mapping from original labels (after pattern filter) to normalized labels (after general norm) for reference
    # This specifically represents the transformation in `normalize_class_name`
    # The `normalization_mapping` created in STEP 2 (if any) is what's saved here.
    if 'normalization_mapping' in locals():
        mapping_path_general_norm = output_dir / "label_normalization_map.json"
        with open(mapping_path_general_norm, 'w') as f:
            json.dump(normalization_mapping, f, indent=4)
        print(f"Saved general normalization map (Step 2) to: {mapping_path_general_norm}")
    else:
        print("No general normalization map was generated (e.g., no samples for Step 2).")

    # Generate detailed class statistics
    class_stats = []
    # Loop through the final set of class names (after size filtering, which are the keys in `current_final_label_dict`)
    for final_class_name_for_stats in sorted(current_final_label_dict.keys()):
        # Find all original raw class names that ultimately map to this final_class_name_for_stats
        original_raw_classes_mapping_to_this = [
            raw_name for raw_name, mapped_final_name in final_mapping_from_raw_to_normalized.items()
            if mapped_final_name == final_class_name_for_stats
        ]

        # Sum their counts from the truly raw original dataset
        original_raw_count_for_this_class = sum(
            original_class_counts_raw.get(name, 0) for name in original_raw_classes_mapping_to_this)

        # Get counts at other stages from the already computed Counters
        after_general_norm_count = normalized_counts.get(final_class_name_for_stats, 0)
        after_size_filter_count = size_filtered_counts.get(final_class_name_for_stats, 0)
        final_bootstrapped_count = final_counts.get(final_class_name_for_stats, 0)

        class_stats.append({
            "class": final_class_name_for_stats,
            "original_raw_classes_mapped": original_raw_classes_mapping_to_this,
            "original_raw_count": original_raw_count_for_this_class,
            "after_general_normalization_count": after_general_norm_count,
            # This count includes effects of manual fix and pattern filter
            "after_size_filtering_count": after_size_filter_count,
            "final_bootstrapped_count": final_bootstrapped_count,
            "bootstrapped_samples_added": final_bootstrapped_count - after_size_filter_count
        })

    # Save detailed statistics
    stats_path = output_dir / "class_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(class_stats, f, indent=4)
    print(f"Saved detailed statistics to: {stats_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Original dataset size (raw): {len(dataset)} samples, {len(original_class_counts_raw)} classes")
    print(f"After manual merges/deletions: {len(dataset_after_manual_fix)} samples, {len(current_label_dict)} classes")
    print(f"After pattern filtering: {len(filtered_dataset)} samples, {len(after_pattern_filtering_counts)} classes")
    print(f"After general normalization: {len(normalized_dataset)} samples, {len(normalized_counts)} classes")
    print(f"After size filtering: {len(size_filtered_dataset)} samples, {len(size_filtered_counts)} classes")
    print(f"After bootstrapping: {len(final_dataset)} samples, {len(final_counts)} classes")
    print(f"Bootstrapped samples added: {len(bootstrapped_samples)}")

    # Print class distribution after processing
    print("\nFinal class distribution (top 30 classes):")
    print("-" * 120)
    header = f"{'Class':<25} | {'Raw Orig Count':<15} | {'After Gen Norm':<18} | {'After Size Filter':<20} | {'Final Count':<15} | {'Added':<10}"
    print(header)
    print("-" * 120)

    sorted_class_stats_for_print = sorted(class_stats, key=lambda x: x['final_bootstrapped_count'], reverse=True)[:30]

    for entry in sorted_class_stats_for_print:
        line = (f"{entry['class']:<25} | {entry['original_raw_count']:<15} | "
                f"{entry['after_general_normalization_count']:<18} | {entry['after_size_filtering_count']:<20} | "
                f"{entry['final_bootstrapped_count']:<15} | {entry['bootstrapped_samples_added']:<10}")
        print(line)

    # Plot class distribution
    if final_counts:
        plt.figure(figsize=(18, 9))  # Increased figure size for more bars

        # Get top 30 classes by final count
        top_classes_for_plot = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        class_names_for_plot = [c[0] for c in top_classes_for_plot]

        x = np.arange(len(class_names_for_plot))
        width = 0.15  # Adjusted width for 4 bars

        # Get counts for these classes at different stages for plotting
        plot_original_raw_counts = []
        plot_after_gen_norm_counts = []
        plot_size_filtered_counts = []
        plot_final_counts = []

        for cls_name_normalized in class_names_for_plot:
            # Raw Original Count
            raw_names_for_plot = [
                raw for raw, final_mapped in final_mapping_from_raw_to_normalized.items()
                if final_mapped == cls_name_normalized
            ]
            plot_original_raw_counts.append(sum(original_class_counts_raw.get(name, 0) for name in raw_names_for_plot))

            # After General Normalization
            plot_after_gen_norm_counts.append(normalized_counts.get(cls_name_normalized, 0))

            # After Size Filtering
            plot_size_filtered_counts.append(size_filtered_counts.get(cls_name_normalized, 0))

            # Final Count
            plot_final_counts.append(final_counts.get(cls_name_normalized, 0))

        plt.bar(x - 1.5 * width, plot_original_raw_counts, width, label='Raw Original Count')
        plt.bar(x - 0.5 * width, plot_after_gen_norm_counts, width, label='After General Normalization')
        plt.bar(x + 0.5 * width, plot_size_filtered_counts, width, label='After Size Filtering')
        plt.bar(x + 1.5 * width, plot_final_counts, width, label='After Bootstrapping')

        plt.xlabel('Classes')
        plt.ylabel('Sample Count')
        title = 'Class Distribution (min_class_size={}, bootstrapped to min={})'.format(min_class_size, min_samples)
        plt.title(title)
        plt.xticks(x, class_names_for_plot, rotation=90, fontsize=8)  # Smaller font for many labels
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / 'class_distribution.png'
        plt.savefig(plot_path)
        print(f"\nClass distribution plot saved to {plot_path}")
    else:
        print("\nSkipping class distribution plot: No classes remain after processing.")

    # Additional plot: Cumulative distribution
    if final_counts:
        plt.figure(figsize=(10, 6))

        # Sort all classes by frequency
        all_classes_cumulative = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
        cumulative_counts = np.cumsum([c[1] for c in all_classes_cumulative])
        total_samples_cumulative = sum(final_counts.values())
        cum_percentages = cumulative_counts / total_samples_cumulative * 100

        plt.plot(range(1, len(cum_percentages) + 1), cum_percentages, marker='o')
        plt.axhline(y=80, color='r', linestyle='--', label='80% of data')

        plt.xlabel('Number of Classes')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Class Distribution (Final Dataset)')
        plt.grid(True)
        plt.legend()

        # Save cumulative plot
        cum_plot_path = output_dir / 'cumulative_distribution.png'
        plt.savefig(cum_plot_path)
        print(f"Cumulative plot saved to {cum_plot_path}")
    else:
        print("Skipping cumulative distribution plot: No classes remain after processing.")

    # Calculate and print class balance metrics
    if final_counts:
        max_count = max(final_counts.values())
        min_count = min(final_counts.values())
        mean_count = sum(final_counts.values()) / len(final_counts)
        std_count = np.std(list(final_counts.values()))

        print("\nClass balance metrics:")
        print(f"Number of classes: {len(final_counts)}")
        print(f"Min samples per class: {min_count}")
        print(f"Max samples per class: {max_count}")
        print(f"Mean samples per class: {mean_count:.2f}")
    else:
        print("\nNo classes remain after processing, so class balance metrics cannot be calculated.")

    return output_dataset_path, output_label_dict_path, stats_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="./config/preproc_config_bp.json",
                        help="Path to JSON config file.")
    args = parser.parse_args()

    config = read_json_config(args.config_path)

    processing_params = config.get('processing_params', {})

    min_class_size = processing_params.get('min_class_size', 20)
    min_samples = processing_params.get('min_samples', 50)
    filter_patterns = processing_params.get('filter_patterns',
                                            ["4DDWI", "4DCT", "4DPT", "4DCDWI", "4DFMRI", "4DPER"])
    do_bootstrap = processing_params.get('do_bootstrap', True)

    dataset_path = Path(config.paths.get('dataset_dir')) / "dataset.pkl"
    label_dict_path = Path(config.paths.get('dataset_dir')) / "label_dict.json"
    output_dir = Path(processing_params.get('output_dir', "./data/processed"))

    print("=" * 80)
    print("DICOM Dataset Processing Parameters:")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Label dictionary path: {label_dict_path}")
    print(f"Output directory: {output_dir}")
    print(f"Filter patterns: {filter_patterns}")
    print(f"Minimum class size: {min_class_size}")
    print(f"Minimum samples per class after bootstrapping: {min_samples}")
    print(f"Perform bootstrapping: {do_bootstrap}")
    print("=" * 80)

    output_dataset_path, output_label_dict_path, stats_path = process_dicom_dataset(
        dataset_path=dataset_path,
        label_dict_path=label_dict_path,
        output_dir=output_dir,
        filter_patterns=filter_patterns,
        min_class_size=min_class_size,
        min_samples=min_samples,
        do_bootstrap=do_bootstrap
    )

    print("\nProcessing complete!")
    print(f"Results saved to:")
    print(f"- Dataset: {output_dataset_path}")
    print(f"- Final Label Dictionary: {output_label_dict_path}")
    print(f"- Detailed Statistics: {stats_path}")
    print(f"- Full Label Processing Map: {output_dir / 'full_label_processing_map.json'}")
    if os.path.exists(output_dir / 'label_normalization_map.json'):
        print(f"- General Normalization Map (Step 2): {output_dir / 'label_normalization_map.json'}")


if __name__ == "__main__":
    main()