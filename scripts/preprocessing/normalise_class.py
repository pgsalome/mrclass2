import os
import pickle
import json
import re
import numpy as np
import random
from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.io import read_yaml_config


def normalize_class_name(label):
    """
    Normalize class names to handle variations in naming conventions

    Args:
        label: Original class label string

    Returns:
        Normalized class label string
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

    # Create inverse mapping from numeric label to class name
    inv_label_dict = {v: k for k, v in original_label_dict.items()}

    # Count samples per class before any processing
    if hasattr(dataset[0], 'label'):
        # If using class-based storage
        original_class_counts = Counter([inv_label_dict[sample.label] for sample in dataset])
    else:
        # If using dictionary storage
        original_class_counts = Counter([inv_label_dict[sample['label']] for sample in dataset])

    # STEP 1: FILTER OUT SPECIFIED PATTERNS
    print("\n--- STEP 1: Filtering out specified patterns ---")

    # Identify classes to filter out
    pattern_filtered_classes = []
    for pattern in filter_patterns:
        matching_classes = [cls for cls in original_label_dict.keys() if pattern in cls]
        pattern_filtered_classes.extend(matching_classes)
        print(f"Found {len(matching_classes)} classes matching pattern '{pattern}'")

    # Remove duplicates
    pattern_filtered_classes = list(set(pattern_filtered_classes))

    print(f"Filtering out {len(pattern_filtered_classes)} classes matching specified patterns:")
    for cls in pattern_filtered_classes[:10]:
        print(f"  - {cls}")
    if len(pattern_filtered_classes) > 10:
        print(f"  - ... and {len(pattern_filtered_classes) - 10} more")

    # Filter out pattern-matched classes
    filtered_dataset = []
    if hasattr(dataset[0], 'label'):
        # Class-based storage
        filtered_dataset = [sample for sample in dataset
                            if inv_label_dict[sample.label] not in pattern_filtered_classes]
    else:
        # Dictionary storage
        filtered_dataset = [sample for sample in dataset
                            if inv_label_dict[sample['label']] not in pattern_filtered_classes]

    print(f"Removed {len(dataset) - len(filtered_dataset)} samples from pattern-filtered classes")

    # STEP 2: NORMALIZE CLASS NAMES
    print("\n--- STEP 2: Normalizing class names ---")

    # Get remaining classes after filtering
    if hasattr(filtered_dataset[0], 'label'):
        remaining_class_names = set([inv_label_dict[sample.label] for sample in filtered_dataset])
    else:
        remaining_class_names = set([inv_label_dict[sample['label']] for sample in filtered_dataset])

    # Normalize each class name
    normalized_labels = {}
    for original_name in remaining_class_names:
        normalized_name = normalize_class_name(original_name)
        normalized_labels[original_name] = normalized_name

    # Count class instances after pattern filtering but before normalization
    after_pattern_filtering = Counter()
    # Count class instances after normalization
    normalized_counts = Counter()

    for sample in filtered_dataset:
        if hasattr(sample, 'label'):
            # Using class structure
            original_class_name = inv_label_dict[sample.label]
            normalized_class_name = normalized_labels[original_class_name]

            after_pattern_filtering[original_class_name] += 1
            normalized_counts[normalized_class_name] += 1
        else:
            # Using dictionary storage
            original_class_name = inv_label_dict[sample['label']]
            normalized_class_name = normalized_labels[original_class_name]

            after_pattern_filtering[original_class_name] += 1
            normalized_counts[normalized_class_name] += 1

    # Create new label dictionary with normalized names
    normalized_label_dict = {name: i for i, name in enumerate(sorted(normalized_counts.keys()))}

    # Create mapping from old labels to new normalized labels
    original_to_normalized_id = {}
    for original_class_name in remaining_class_names:
        original_id = original_label_dict[original_class_name]
        normalized_class_name = normalized_labels[original_class_name]
        new_id = normalized_label_dict[normalized_class_name]
        original_to_normalized_id[original_id] = new_id

    # Update dataset with normalized label IDs
    normalized_dataset = []
    for sample in tqdm(filtered_dataset, desc="Updating dataset with normalized labels"):
        if hasattr(sample, 'label'):
            # Using class structure
            sample.label = original_to_normalized_id[sample.label]
            normalized_dataset.append(sample)
        else:
            # Using dictionary storage
            sample['label'] = original_to_normalized_id[sample['label']]
            normalized_dataset.append(sample)

    print(f"Original classes after pattern filtering: {len(after_pattern_filtering)}")
    print(f"Normalized classes: {len(normalized_counts)}")
    print(f"Reduced by: {len(after_pattern_filtering) - len(normalized_counts)} classes through normalization")

    # Create a mapping of original class names to their normalized versions
    normalization_mapping = {}
    for original_name, normalized_name in normalized_labels.items():
        if normalized_name not in normalization_mapping:
            normalization_mapping[normalized_name] = []
        normalization_mapping[normalized_name].append(original_name)

    # Print some examples of normalization
    print("\nExamples of normalization:")
    # Find examples where normalization resulted in merging
    merged_examples = [(norm, originals) for norm, originals in normalization_mapping.items() if len(originals) > 1]

    # Sort by number of merged classes (descending)
    merged_examples.sort(key=lambda x: len(x[1]), reverse=True)

    # Print the top examples
    for normalized, originals in merged_examples[:10]:
        print(f"  - {normalized}: merged from {len(originals)} classes")
        orig_examples = originals[:3]
        if len(originals) > 3:
            orig_examples.append("...")
        print(f"    Examples: {orig_examples}")

    # STEP 3: FILTER OUT SMALL CLASSES
    print(f"\n--- STEP 3: Filtering out classes with fewer than {min_class_size} samples ---")

    # Create inverse mapping for normalized labels
    inv_normalized_dict = {v: k for k, v in normalized_label_dict.items()}

    # Identify small classes to filter out
    small_classes = [cls for cls, count in normalized_counts.items() if count < min_class_size]

    print(f"Filtering out {len(small_classes)} small classes:")
    for cls in small_classes[:20]:
        print(f"  - {cls} ({normalized_counts[cls]} samples)")
    if len(small_classes) > 20:
        print(f"  - ... and {len(small_classes) - 20} more")

    # Filter out small classes
    if hasattr(normalized_dataset[0], 'label'):
        # Class-based storage
        size_filtered_dataset = [sample for sample in normalized_dataset
                                 if inv_normalized_dict[sample.label] not in small_classes]
    else:
        # Dictionary storage
        size_filtered_dataset = [sample for sample in normalized_dataset
                                 if inv_normalized_dict[sample['label']] not in small_classes]

    print(f"Removed {len(normalized_dataset) - len(size_filtered_dataset)} samples from small classes")

    # Count samples per class after size filtering
    if hasattr(size_filtered_dataset[0], 'label'):
        size_filtered_counts = Counter([inv_normalized_dict[sample.label] for sample in size_filtered_dataset])
    else:
        size_filtered_counts = Counter([inv_normalized_dict[sample['label']] for sample in size_filtered_dataset])

    # Create new label dictionary without small classes
    filtered_label_dict = {name: i for i, name in enumerate(sorted(size_filtered_counts.keys()))}

    # Create mapping from old normalized labels to new filtered labels
    normalized_to_filtered_id = {}
    for class_name in size_filtered_counts.keys():
        old_id = normalized_label_dict[class_name]
        new_id = filtered_label_dict[class_name]
        normalized_to_filtered_id[old_id] = new_id

    # Update dataset with filtered label IDs
    filtered_labeled_dataset = []
    for sample in tqdm(size_filtered_dataset, desc="Updating dataset with filtered labels"):
        if hasattr(sample, 'label'):
            # Using class structure
            sample.label = normalized_to_filtered_id[sample.label]
            filtered_labeled_dataset.append(sample)
        else:
            # Using dictionary storage
            sample['label'] = normalized_to_filtered_id[sample['label']]
            filtered_labeled_dataset.append(sample)

    # Update variables for the next step
    size_filtered_dataset = filtered_labeled_dataset
    normalized_label_dict = filtered_label_dict

    # STEP 4: BOOTSTRAP UNDERREPRESENTED CLASSES
    print(f"\n--- STEP 4: Bootstrapping classes with fewer than {min_samples} samples ---")

    # Create inverse mapping for current labels
    inv_current_dict = {v: k for k, v in normalized_label_dict.items()}

    # Identify classes with fewer than min_samples
    classes_to_bootstrap = [cls for cls, count in size_filtered_counts.items() if count < min_samples]

    print(f"Bootstrapping {len(classes_to_bootstrap)} classes with fewer than {min_samples} samples:")
    for cls in classes_to_bootstrap[:20]:
        print(f"  - {cls} ({size_filtered_counts[cls]} samples)")
    if len(classes_to_bootstrap) > 20:
        print(f"  - ... and {len(classes_to_bootstrap) - 20} more")

    # Dictionary to keep track of original indices for each class
    class_indices = {}

    # Populate class_indices dictionary
    for i, sample in enumerate(size_filtered_dataset):
        if hasattr(sample, 'label'):
            class_name = inv_current_dict[sample.label]
        else:
            class_name = inv_current_dict[sample['label']]

        if class_name not in class_indices:
            class_indices[class_name] = []

        class_indices[class_name].append(i)

    # Perform bootstrapping for underrepresented classes
    bootstrapped_samples = []

    for class_name in tqdm(classes_to_bootstrap, desc="Bootstrapping classes"):
        count = size_filtered_counts[class_name]

        # Skip classes with enough samples
        if count >= min_samples:
            continue

        # Get indices of samples for this class
        indices = class_indices[class_name]

        # Determine how many more samples we need
        samples_needed = min_samples - count

        # Sample with replacement to create new samples
        for _ in range(samples_needed):
            # Choose a random sample from this class
            idx = random.choice(indices)
            original_sample = size_filtered_dataset[idx]

            # Create a deep copy of the sample
            if hasattr(original_sample, 'label'):
                # Class-based storage
                # Create a new instance with the same attributes
                class_attrs = original_sample.__dict__.copy()
                new_sample = type(original_sample)(**class_attrs)
            else:
                # Dictionary storage
                new_sample = original_sample.copy()

            # Add minor noise to numerical attributes to prevent exact duplication
            if hasattr(new_sample, 'numerical_attributes') and new_sample.numerical_attributes is not None:
                # Add small random noise to numerical features
                noise = np.random.normal(0, 0.01, len(new_sample.numerical_attributes))
                new_sample.numerical_attributes = [max(0, float(val) + n) for val, n in
                                                   zip(new_sample.numerical_attributes, noise)]
            elif 'numerical_attributes' in new_sample and new_sample['numerical_attributes'] is not None:
                noise = np.random.normal(0, 0.01, len(new_sample['numerical_attributes']))
                new_sample['numerical_attributes'] = [max(0, float(val) + n) for val, n in
                                                      zip(new_sample['numerical_attributes'], noise)]

            # Add the bootstrapped sample to our list
            bootstrapped_samples.append(new_sample)

    # Add bootstrapped samples to the dataset
    final_dataset = size_filtered_dataset + bootstrapped_samples

    # Count samples per class after bootstrapping
    if hasattr(final_dataset[0], 'label'):
        final_counts = Counter([inv_current_dict[sample.label] for sample in final_dataset])
    else:
        final_counts = Counter([inv_current_dict[sample['label']] for sample in final_dataset])

    # STEP 5: SAVE RESULTS AND GENERATE STATISTICS

    # Save the processed dataset as bootstrapped
    dataset_filename = "bootstrapped_dataset.pkl"
    label_dict_filename = "bootstrapped_label_dict.json"

    # Save the processed dataset
    output_dataset_path = output_dir / dataset_filename
    with open(output_dataset_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    print(f"\nSaved processed dataset to: {output_dataset_path}")

    # Save the final label dictionary
    output_label_dict_path = output_dir / label_dict_filename
    with open(output_label_dict_path, 'w') as f:
        json.dump(normalized_label_dict, f, indent=4)
    print(f"Saved label dictionary to: {output_label_dict_path}")

    # Save mapping from original to normalized classes for reference
    mapping_path = output_dir / "label_normalization_map.json"
    with open(mapping_path, 'w') as f:
        json.dump(normalization_mapping, f, indent=4)
    print(f"Saved normalization mapping to: {mapping_path}")

    # Generate detailed class statistics
    class_stats = []
    for class_name in sorted(normalized_label_dict.keys()):
        # Find original classes that map to this normalized class
        original_classes = normalization_mapping.get(class_name, [])
        original_count = sum(original_class_counts.get(orig, 0) for orig in original_classes)

        after_norm = normalized_counts.get(class_name, 0)
        after_size_filter = size_filtered_counts.get(class_name, 0)
        final_count = final_counts.get(class_name, 0)

        class_stats.append({
            "class": class_name,
            "original_classes": original_classes,
            "original_count": original_count,
            "after_normalization": after_norm,
            "after_size_filtering": after_size_filter,
            "final_count": final_count,
            "bootstrapped_samples_added": final_count - after_size_filter
        })

    # Save detailed statistics
    stats_path = output_dir / "class_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(class_stats, f, indent=4)
    print(f"Saved detailed statistics to: {stats_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Original dataset size: {len(dataset)} samples, {len(original_class_counts)} classes")
    print(f"After pattern filtering: {len(filtered_dataset)} samples, {len(after_pattern_filtering)} classes")
    print(f"After normalization: {len(normalized_dataset)} samples, {len(normalized_counts)} classes")
    print(f"After size filtering: {len(size_filtered_dataset)} samples, {len(size_filtered_counts)} classes")
    print(f"After bootstrapping: {len(final_dataset)} samples, {len(final_counts)} classes")
    print(f"Bootstrapped samples added: {len(bootstrapped_samples)}")

    # Print class distribution after processing
    print("\nFinal class distribution (top 30 classes):")
    print("-" * 100)
    header = f"{'Class':<30} | {'Original Count':<15} | {'After Normalization':<20} | {'After Size Filter':<18} | {'Final Count':<15} | {'Added':<10}"
    print(header)
    print("-" * 100)

    for cls, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
        # Find original classes that map to this normalized class
        original_classes = normalization_mapping.get(cls, [])
        original_count = sum(original_class_counts.get(orig, 0) for orig in original_classes)

        after_norm = normalized_counts.get(cls, 0)
        after_filter = size_filtered_counts.get(cls, 0)
        bootstrapped = count - size_filtered_counts.get(cls, 0)

        line = f"{cls:<30} | {original_count:<15} | {after_norm:<20} | {after_filter:<18} | {count:<15} | {bootstrapped:<10}"
        print(line)

    # Plot class distribution
    plt.figure(figsize=(15, 8))

    # Get top 30 classes by count
    top_classes = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    class_names = [c[0] for c in top_classes]

    # Get counts for these classes at different stages
    original_counts = []
    for cls in class_names:
        # Find original classes that map to this normalized class
        orig_classes = normalization_mapping.get(cls, [])
        orig_count = sum(original_class_counts.get(orig, 0) for orig in orig_classes)
        original_counts.append(orig_count)

    after_norm_counts = [normalized_counts.get(c, 0) for c in class_names]
    size_filtered_class_counts = [size_filtered_counts.get(c, 0) for c in class_names]
    final_class_counts = [final_counts.get(c, 0) for c in class_names]

    # Set the number of bars and their width based on processing steps
    x = np.arange(len(class_names))
    width = 0.2

    plt.bar(x - 1.5 * width, original_counts, width, label='Original Count')
    plt.bar(x - 0.5 * width, after_norm_counts, width, label='After Normalization')
    plt.bar(x + 0.5 * width, size_filtered_class_counts, width, label='After Size Filtering')
    plt.bar(x + 1.5 * width, final_class_counts, width, label='After Bootstrapping')

    plt.xlabel('Classes')
    plt.ylabel('Sample Count')
    title = 'Class Distribution (min_class_size={}, bootstrapped to min={})'.format(min_class_size, min_samples)
    plt.title(title)
    plt.xticks(x, class_names, rotation=90)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / 'class_distribution.png'
    plt.savefig(plot_path)
    print(f"\nClass distribution plot saved to {plot_path}")

    # Additional plot: Cumulative distribution
    plt.figure(figsize=(10, 6))

    # Sort all classes by frequency
    all_classes = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
    cumulative_counts = np.cumsum([c[1] for c in all_classes])
    total_samples = sum(final_counts.values())
    cum_percentages = cumulative_counts / total_samples * 100

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

    # Calculate and print class balance metrics
    max_count = max(final_counts.values())
    min_count = min(final_counts.values())
    mean_count = sum(final_counts.values()) / len(final_counts)
    std_count = np.std(list(final_counts.values()))

    print("\nClass balance metrics:")
    print(f"Number of classes: {len(final_counts)}")
    print(f"Min samples per class: {min_count}")
    print(f"Max samples per class: {max_count}")
    print(f"Mean samples per class: {mean_count:.2f}")

    return output_dataset_path, output_label_dict_path, stats_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file.")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/processed",
                        help="Directory to save processed dataset.")
    parser.add_argument("-m", "--min_samples", type=int, default=50,
                        help="Minimum samples per class after bootstrapping.")
    parser.add_argument("-s", "--min_class_size", type=int, default=20,
                        help="Minimum class size to keep (classes with fewer samples will be removed).")
    parser.add_argument("-f", "--filter_patterns", nargs='+',
                        default=["4DDWI", "4DCT", "4DPT", "4DCDWI", "4DFMRI", "4DPER"],
                        help="Patterns to filter out classes (e.g., 4DDWI 4DCT 4DPT).")
    parser.add_argument("-b", "--bootstrap", action="store_true", default=True,
                        help="Enable bootstrapping of underrepresented classes.")
    config_args = parser.parse_args()

    # Read the configuration file
    args = read_yaml_config(config_args.config_path)

    # Extract paths from config
    print(args)
    dataset_path = Path(args.paths.dataset_dir) / "dataset.pkl"
    label_dict_path = Path(args.paths.dataset_dir) / "label_dict.json"

    print("=" * 80)
    print("DICOM Dataset Processing Parameters:")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Label dictionary path: {label_dict_path}")
    print(f"Output directory: {config_args.output_dir}")
    print(f"Filter patterns: {config_args.filter_patterns}")
    print(f"Minimum class size: {config_args.min_class_size}")
    print(f"Minimum samples per class after bootstrapping: {config_args.min_samples}")
    print(f"Perform bootstrapping: {config_args.bootstrap}")
    print("=" * 80)

    # Call the main processing function
    output_dataset_path, output_label_dict_path, stats_path = process_dicom_dataset(
        dataset_path=dataset_path,
        label_dict_path=label_dict_path,
        output_dir=config_args.output_dir,
        filter_patterns=config_args.filter_patterns,
        min_class_size=config_args.min_class_size,
        min_samples=config_args.min_samples,
        do_bootstrap=config_args.bootstrap
    )

    print("\nProcessing complete!")
    print(f"Results saved to:")
    print(f"- Dataset: {output_dataset_path}")
    print(f"- Label dictionary: {output_label_dict_path}")
    print(f"- Statistics: {stats_path}")