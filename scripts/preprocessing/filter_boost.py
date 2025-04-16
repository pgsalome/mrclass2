import os
import pickle
import json
import numpy as np
import random
from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.io import read_yaml_config


def filter_and_bootstrap_dataset(dataset_path, label_dict_path, output_dir, min_samples=50, filter_pattern='4DDWI'):
    """
    Filter out specific classes and bootstrap underrepresented classes

    Args:
        dataset_path: Path to the dataset pickle file
        label_dict_path: Path to the label dictionary json file
        output_dir: Directory to save processed dataset
        min_samples: Minimum number of samples per class
        filter_pattern: Pattern to filter out classes (e.g., '4DDWI')
    """
    print(f"Loading dataset from {dataset_path}")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Load the label dictionary
    with open(label_dict_path, 'r') as f:
        label_dict = json.load(f)

    # Create inverse mapping from numeric label to class name
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Count samples per class before filtering
    if hasattr(dataset[0], 'label'):
        # If using class-based storage
        class_counts_before = Counter([inv_label_dict[sample.label] for sample in dataset])
    else:
        # If using dictionary storage
        class_counts_before = Counter([inv_label_dict[sample['label']] for sample in dataset])

    # Identify classes to filter out
    classes_to_filter = [cls for cls in label_dict.keys() if filter_pattern in cls]
    print(f"\nFiltering out {len(classes_to_filter)} classes containing '{filter_pattern}':")
    for cls in classes_to_filter[:10]:
        print(f"  - {cls}")
    if len(classes_to_filter) > 10:
        print(f"  - ... and {len(classes_to_filter) - 10} more")

    # Identify classes with fewer than min_samples
    classes_to_bootstrap = [cls for cls, count in class_counts_before.items()
                            if count < min_samples and cls not in classes_to_filter]
    print(f"\nBootstrapping {len(classes_to_bootstrap)} classes with fewer than {min_samples} samples:")
    for cls in classes_to_bootstrap[:10]:
        print(f"  - {cls} ({class_counts_before[cls]} samples)")
    if len(classes_to_bootstrap) > 10:
        print(f"  - ... and {len(classes_to_bootstrap) - 10} more")

    # Filter out unwanted classes
    filtered_dataset = []
    if hasattr(dataset[0], 'label'):
        # Class-based storage
        filtered_dataset = [sample for sample in dataset
                            if inv_label_dict[sample.label] not in classes_to_filter]
    else:
        # Dictionary storage
        filtered_dataset = [sample for sample in dataset
                            if inv_label_dict[sample['label']] not in classes_to_filter]

    print(f"\nRemoved {len(dataset) - len(filtered_dataset)} samples from classes containing '{filter_pattern}'")

    # Create new label dictionary after filtering
    if hasattr(filtered_dataset[0], 'label'):
        remaining_classes = set([inv_label_dict[sample.label] for sample in filtered_dataset])
    else:
        remaining_classes = set([inv_label_dict[sample['label']] for sample in filtered_dataset])

    new_label_dict = {class_name: i for i, class_name in enumerate(sorted(remaining_classes))}

    # Create mapping from old labels to new labels
    old_to_new_label = {}
    for class_name in remaining_classes:
        old_id = label_dict[class_name]
        new_id = new_label_dict[class_name]
        old_to_new_label[old_id] = new_id

    # Update labels in the filtered dataset
    for sample in filtered_dataset:
        if hasattr(sample, 'label'):
            old_label = sample.label
            sample.label = old_to_new_label[old_label]
        else:
            old_label = sample['label']
            sample['label'] = old_to_new_label[old_label]

    # Count samples per class after filtering but before bootstrapping
    if hasattr(filtered_dataset[0], 'label'):
        # Using new label dictionary
        inv_new_label_dict = {v: k for k, v in new_label_dict.items()}
        class_counts_after_filtering = Counter([inv_new_label_dict[sample.label] for sample in filtered_dataset])
    else:
        class_counts_after_filtering = Counter([inv_new_label_dict[sample['label']] for sample in filtered_dataset])

    # Bootstrap underrepresented classes
    bootstrapped_dataset = filtered_dataset.copy()
    bootstrapped_samples = []

    # Dictionary to keep track of original indices for each class
    class_indices = {}

    # Populate class_indices dictionary
    for i, sample in enumerate(filtered_dataset):
        if hasattr(sample, 'label'):
            class_name = inv_new_label_dict[sample.label]
        else:
            class_name = inv_new_label_dict[sample['label']]

        if class_name not in class_indices:
            class_indices[class_name] = []

        class_indices[class_name].append(i)

    # Perform bootstrapping for underrepresented classes
    for class_name in tqdm(class_counts_after_filtering, desc="Bootstrapping classes"):
        count = class_counts_after_filtering[class_name]

        # Skip classes with enough samples or that were filtered out
        if count >= min_samples or class_name not in new_label_dict:
            continue

        # Get indices of samples for this class
        indices = class_indices[class_name]

        # Determine how many more samples we need
        samples_needed = min_samples - count

        # Sample with replacement to create new samples
        for _ in range(samples_needed):
            # Choose a random sample from this class
            idx = random.choice(indices)
            original_sample = filtered_dataset[idx]

            # Create a deep copy of the sample
            if hasattr(original_sample, 'label'):
                # Class-based storage
                # Create a new instance with the same attributes
                # Note: This depends on your specific class structure
                # This is a simplified example; adjust according to your actual class
                class_attrs = original_sample.__dict__.copy()
                new_sample = type(original_sample)(**class_attrs)
            else:
                # Dictionary storage
                new_sample = original_sample.copy()

            # Add minor noise to numerical attributes to prevent exact duplication
            # This helps the model generalize better
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
    bootstrapped_dataset.extend(bootstrapped_samples)

    # Count samples per class after bootstrapping
    if hasattr(bootstrapped_dataset[0], 'label'):
        class_counts_after_bootstrap = Counter([inv_new_label_dict[sample.label] for sample in bootstrapped_dataset])
    else:
        class_counts_after_bootstrap = Counter([inv_new_label_dict[sample['label']] for sample in bootstrapped_dataset])

    # Print statistics
    print("\nClass distribution before filtering:")
    print(f"Total classes: {len(class_counts_before)}")
    print(f"Total samples: {len(dataset)}")

    print("\nClass distribution after filtering out 4DDWI classes:")
    print(f"Total classes: {len(class_counts_after_filtering)}")
    print(f"Total samples: {len(filtered_dataset)}")
    print(f"Classes removed: {len(class_counts_before) - len(class_counts_after_filtering)}")
    print(f"Samples removed: {len(dataset) - len(filtered_dataset)}")

    print("\nClass distribution after bootstrapping:")
    print(f"Total classes: {len(class_counts_after_bootstrap)}")
    print(f"Total samples: {len(bootstrapped_dataset)}")
    print(f"Bootstrapped samples added: {len(bootstrapped_samples)}")

    # Save the bootstrapped dataset
    output_dataset_path = output_dir / "bootstrapped_dataset.pkl"
    with open(output_dataset_path, 'wb') as f:
        pickle.dump(bootstrapped_dataset, f)

    # Save the new label dictionary
    output_label_dict_path = output_dir / "bootstrapped_label_dict.json"
    with open(output_label_dict_path, 'w') as f:
        json.dump(new_label_dict, f, indent=4)

    # Generate detailed class statistics before and after
    class_stats = []
    for class_name in sorted(new_label_dict.keys()):
        before = class_counts_before.get(class_name, 0)
        after_filter = class_counts_after_filtering.get(class_name, 0)
        after_bootstrap = class_counts_after_bootstrap.get(class_name, 0)

        class_stats.append({
            "class": class_name,
            "before_filtering": before,
            "after_filtering": after_filter,
            "after_bootstrapping": after_bootstrap,
            "bootstrapped_samples_added": after_bootstrap - after_filter
        })

    # Save detailed statistics
    stats_path = output_dir / "class_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(class_stats, f, indent=4)

    # Print class distribution after bootstrapping
    print("\nDetailed class distribution after bootstrapping (top 30):")
    print("-" * 80)
    print(f"{'Class':<30} | {'Count Before':<12} | {'Count After':<12} | {'Bootstrapped':<12}")
    print("-" * 80)

    for cls, count in sorted(class_counts_after_bootstrap.items(), key=lambda x: x[1], reverse=True)[:30]:
        before = class_counts_before.get(cls, 0)
        bootstrapped = count - class_counts_after_filtering.get(cls, 0)
        print(f"{cls:<30} | {before:<12} | {count:<12} | {bootstrapped:<12}")

    # Plot class distribution before and after bootstrapping
    plt.figure(figsize=(15, 8))

    # Get top 30 classes by count after bootstrapping
    top_classes = sorted(class_counts_after_bootstrap.items(), key=lambda x: x[1], reverse=True)[:30]
    classes = [c[0] for c in top_classes]

    # Get counts for these classes before and after
    before_counts = [class_counts_after_filtering.get(c, 0) for c in classes]
    after_counts = [class_counts_after_bootstrap.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width / 2, before_counts, width, label='Before Bootstrapping')
    plt.bar(x + width / 2, after_counts, width, label='After Bootstrapping')

    plt.xlabel('Classes')
    plt.ylabel('Sample Count')
    plt.title('Class Distribution Before and After Bootstrapping (Top 30 Classes)')
    plt.xticks(x, classes, rotation=90)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / 'bootstrap_comparison.png'
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to {plot_path}")

    # Print final paths
    print(f"\nBootstrapped dataset saved to {output_dataset_path}")
    print(f"New label dictionary saved to {output_label_dict_path}")
    print(f"Detailed statistics saved to {stats_path}")

    return output_dataset_path, output_label_dict_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file.")
    parser.add_argument("-i", "--input_dir", type=str, default=None,
                        help="Directory containing normalized dataset (overrides config).")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/bootstrapped",
                        help="Directory to save bootstrapped dataset.")
    parser.add_argument("-m", "--min_samples", type=int, default=50,
                        help="Minimum samples per class after bootstrapping.")
    parser.add_argument("-f", "--filter_pattern", type=str, default="4DDWI",
                        help="Pattern to filter out classes (e.g., '4DDWI').")
    config_args = parser.parse_args()

    args = read_yaml_config(config_args.config_path)

    if config_args.input_dir:
        input_dir = Path(config_args.input_dir)
        dataset_path = input_dir / "normalized_dataset.pkl"
        label_dict_path = input_dir / "normalized_label_dict.json"
    else:
        dataset_path = Path(args.paths.dataset_dir) / "dataset.pkl"
        label_dict_path = Path(args.paths.dataset_dir) / "label_dict.json"

    filter_and_bootstrap_dataset(
        dataset_path,
        label_dict_path,
        config_args.output_dir,
        min_samples=config_args.min_samples,
        filter_pattern=config_args.filter_pattern
    )