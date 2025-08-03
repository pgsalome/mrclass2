import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.lazy_dataset import create_lazy_datasets

# Original dataset and data structure imports
from utils.dataclass import encodedSample, MRISequenceDataset
from utils.io import load_pickle, read_json_config, ensure_dir
from models.num_encoder import NumericFeatureNormalizer

# Import MONAI dataset classes for caching
from utils.monai_dataset import MRIMonaiDataset, MRIPersistentDataset


def create_datasets_with_lazy_option(config, hierarchical=False, use_lazy=True):
    """
    Create datasets with option for lazy loading.

    Args:
        config: Configuration dictionary
        hierarchical: Whether to use hierarchical classification
        use_lazy: Whether to use lazy loading (default: True)

    Returns:
        Same as create_datasets() but with lazy loading if enabled
    """
    if use_lazy and "orientation" in config["data"]:
        # Use lazy loading
        print("Using LAZY LOADING for dataset...")
        datasets, label_dict, hierarchical_dicts, num_normalizer = create_lazy_datasets(config)
    else:
        # Fall back to original implementation
        print("Using standard (eager) loading for dataset...")
        from utils.data_loader import create_datasets
        datasets, label_dict, hierarchical_dicts, num_normalizer = create_datasets(config, hierarchical)

    return datasets, label_dict, hierarchical_dicts, num_normalizer


def prepare_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Prepare image transformations for training, validation, and test sets.

    Args:
        config: Configuration dictionary containing image size and augmentation parameters.

    Returns:
        Dictionary with train, val, and test transform pipelines.
    """
    img_size = config["data"]["img_size"]

    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # MRI-specific intensity augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Validation/test transforms without augmentation
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return {
        "train": train_transforms,
        "val": val_transforms,
        "test": val_transforms
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[List[encodedSample], Dict[str, int]]:
    """
    Load the encoded dataset and corresponding label dictionary.

    Args:
        config: Configuration dictionary with paths for the dataset and label dictionary.

    Returns:
        Tuple containing:
        - List of encodedSample objects
        - Dictionary mapping label names to indices
    """
    print(f"DEBUG: load_dataset called")
    print(f"DEBUG: config keys: {list(config.get('data', {}).keys())}")
    print(f"DEBUG: orientation in config: {'orientation' in config.get('data', {})}")
    print(f"DEBUG: orientation value: {config.get('data', {}).get('orientation', 'NOT SET')}")

    data_dir = Path(config["data"]["dataset_dir"])

    # Check if we're loading from batches
    if "orientation" in config["data"]:
        # Load from batches
        orientation = config["data"]["orientation"]
        batches_dir = data_dir / orientation / "batches"

        if not batches_dir.exists():
            raise FileNotFoundError(f"Batches directory not found: {batches_dir}")

        # Load all batch files
        full_dataset = []
        batch_files = sorted(batches_dir.glob("dataset_batch_*.pkl"))

        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {batches_dir}")

        print(f"Loading {len(batch_files)} batch files from {batches_dir}...")
        from tqdm import tqdm
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                full_dataset.extend(batch_data)

        print(f"Loaded {len(full_dataset)} samples total")

        # FIX: Pad all sequences to the same length
        print("Fixing sequence lengths...")

        # Find max length
        max_length = max(len(sample.input_ids) for sample in full_dataset)
        print(f"Max sequence length found: {max_length}")

        # Get tokenizer for padding token
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # or distilbert-base-uncased
        pad_token_id = tokenizer.pad_token_id or 0

        # Pad all sequences
        for sample in tqdm(full_dataset, desc="Padding sequences"):
            current_length = len(sample.input_ids)
            if current_length < max_length:
                padding_length = max_length - current_length
                # Pad with lists (since that's what your working dataset uses)
                sample.input_ids = sample.input_ids + [pad_token_id] * padding_length
                sample.attention_mask = sample.attention_mask + [0] * padding_length

        # Verify all sequences now have the same length
        lengths = set(len(sample.input_ids) for sample in full_dataset[:100])
        print(f"Sequence lengths after padding (first 100): {lengths}")

        # Load label dictionary
        label_dict_path = data_dir / orientation / f"label_dict_{orientation}.json"
        if not label_dict_path.exists():
            raise FileNotFoundError(f"Label dictionary not found: {label_dict_path}")

        with open(label_dict_path, 'r') as f:
            label_dict = json.load(f)

        return full_dataset, label_dict
    else:
        # Original single file loading
        dataset_path = data_dir / config["data"]["dataset_name"]
        label_dict_path = data_dir / config["data"]["label_dict_name"]

        # Load dataset from pickle file
        dataset = load_pickle(str(dataset_path))

        # Load label dictionary from JSON file
        with open(label_dict_path, 'r') as f:
            label_dict = json.load(f)

        return dataset, label_dict
def prepare_hierarchical_labels(
        dataset: List[encodedSample],
        label_dict: Dict[str, int]
) -> Dict[str, Dict[str, int]]:
    """
    Prepare hierarchical label dictionaries from the original labels.

    Args:
        dataset: List of encoded samples.
        label_dict: Original label dictionary.

    Returns:
        Dictionary of hierarchical label dictionaries including base contrast,
        sequence type, and fat suppression.
    """
    # Invert label dictionary to map indices to class names
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Extract base contrast (e.g., T1, T2, DWI)
    base_contrasts = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        base_contrast = class_name.split('-')[0] if '-' in class_name else class_name
        base_contrasts.add(base_contrast)

    base_contrast_dict = {contrast: i for i, contrast in enumerate(sorted(base_contrasts))}

    # Extract sequence type (e.g., SE, GE, IRGE)
    sequence_types = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')
        if len(parts) > 1:
            if any(seq_type in parts[1:] for seq_type in ['SE', 'GE', 'IRGE', 'IRSE']):
                for part in parts[1:]:
                    if part in ['SE', 'GE', 'IRGE', 'IRSE']:
                        sequence_types.add(part)
                        break

    sequence_type_dict = {seq_type: i for i, seq_type in enumerate(sorted(sequence_types))}

    # Fat suppression information
    fat_sup = {"Yes": 0, "No": 1}

    return {
        "base_contrast": base_contrast_dict,
        "sequence_type": sequence_type_dict,
        "fat_suppression": fat_sup
    }


def extract_hierarchical_labels(
        dataset: List[encodedSample],
        label_dict: Dict[str, int],
        hierarchical_dicts: Dict[str, Dict[str, int]]
) -> List[Dict[str, int]]:
    """
    Extract hierarchical labels for each sample.

    Args:
        dataset: List of encoded samples.
        label_dict: Original label dictionary.
        hierarchical_dicts: Dictionary of hierarchical label dictionaries.

    Returns:
        List of dictionaries with hierarchical labels for each sample.
    """
    inv_label_dict = {v: k for k, v in label_dict.items()}
    hierarchical_labels = []

    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')

        base_contrast = parts[0]
        base_contrast_label = hierarchical_dicts["base_contrast"].get(base_contrast, 0)

        sequence_type = "Unknown"
        for part in parts[1:]:
            if part in hierarchical_dicts["sequence_type"]:
                sequence_type = part
                break
        sequence_type_label = hierarchical_dicts["sequence_type"].get(sequence_type, 0)

        fat_sup_label = hierarchical_dicts["fat_suppression"]["Yes"] if "FS" in parts else \
            hierarchical_dicts["fat_suppression"]["No"]

        hierarchical_labels.append({
            "base_contrast": base_contrast_label,
            "sequence_type": sequence_type_label,
            "fat_suppression": fat_sup_label,
            "fine_class": dataset[idx].label  # Original fine-grained class label
        })

    return hierarchical_labels


def create_datasets(
        config: Dict[str, Any],
        hierarchical: bool = False,
        use_monai: bool = True,
        cache_type: str = "memory"  # Options: "memory", "disk", "none"
) -> Tuple[
    Dict[str, Union[MRIMonaiDataset, MRISequenceDataset]], Dict[str, int],
    Optional[Dict[str, Dict[str, int]]], NumericFeatureNormalizer]:
    """
    Create training, validation, and test datasets with optional MONAI caching.

    Args:
        config: Configuration dictionary.
        hierarchical: Whether to prepare hierarchical labels.
        use_monai: Flag to select MONAI dataset implementations.
        cache_type: Caching strategy ("memory", "disk", "none").

    Returns:
        A tuple containing:
          - Dictionary mapping split names to dataset objects.
          - Label dictionary.
          - Hierarchical label dictionaries (if hierarchical=True).
          - Numeric feature normalizer.
    """
    # Load dataset and label dictionary
    dataset, label_dict = load_dataset(config)

    # Handle test mode: use a subset with the two most frequent classes if enabled.
    if config["data"].get("test_mode", False):
        num_samples = config["data"].get("num_samples", 100)
        labels = [sample.label for sample in dataset]
        label_counts = Counter(labels)
        top_two = [label for label, count in label_counts.most_common(2)]
        dataset = [sample for sample in dataset if sample.label in top_two]
        dataset = dataset[:min(num_samples, len(dataset))]
        print(f"Test mode enabled: using {len(dataset)} samples from top two classes: {top_two}.")

    # Prepare image transformations
    transforms_dict = prepare_transforms(config)

    # Prepare hierarchical labels if needed
    hierarchical_dicts = None
    if hierarchical:
        hierarchical_dicts = prepare_hierarchical_labels(dataset, label_dict)
        # hierarchical_labels can be extracted later if required

    # Normalize numerical features
    num_normalizer = NumericFeatureNormalizer(method="z-score")
    num_features = [sample.numerical_attributes for sample in dataset]
    num_normalizer.fit(num_features)

    # Update numerical features in the dataset using the normalizer
    for i in range(len(dataset)):
        normalized = num_normalizer.transform([dataset[i].numerical_attributes])[0]
        dataset[i].numerical_attributes = normalized.tolist()

    # Split dataset indices into train, validation, and test sets
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]

    labels = [sample.label for sample in dataset]
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=config["seed"],
        stratify=labels
    )

    # Further split temporary indices into validation and test indices
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    temp_labels = [labels[i] for i in temp_indices]
    val_indices_from_temp, test_indices_from_temp = train_test_split(
        range(len(temp_indices)),
        test_size=(1.0 - val_ratio_of_temp),
        random_state=config["seed"],
        stratify=temp_labels
    )
    val_indices = [temp_indices[i] for i in val_indices_from_temp]
    test_indices = [temp_indices[i] for i in test_indices_from_temp]

    # Create dataset subsets for each split
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Create dataset objects based on caching strategy and MONAI usage
    if use_monai:
        if cache_type == "memory":
            num_workers = config["data"].get("cache_num_workers", 4)
            datasets_obj = {
                "train": MRIMonaiDataset(
                    train_dataset,
                    transform=transforms_dict["train"],
                    cache_rate=config["data"].get("cache_rate", 1.0),
                    num_workers=num_workers
                ),
                "val": MRIMonaiDataset(
                    val_dataset,
                    transform=transforms_dict["val"],
                    cache_rate=config["data"].get("cache_rate", 1.0),
                    num_workers=num_workers
                ),
                "test": MRIMonaiDataset(
                    test_dataset,
                    transform=transforms_dict["test"],
                    cache_rate=config["data"].get("cache_rate", 1.0),
                    num_workers=num_workers
                )
            }
        elif cache_type == "disk":
            import hashlib
            import time
            config_str = str(config).encode('utf-8')
            config_hash = hashlib.md5(config_str).hexdigest()[:8]
            cache_dir = f"./persistent_cache_{config_hash}_{int(time.time())}"
            datasets_obj = {
                "train": MRIPersistentDataset(
                    train_dataset,
                    transform=transforms_dict["train"],
                    cache_dir=f"{cache_dir}/train"
                ),
                "val": MRIPersistentDataset(
                    val_dataset,
                    transform=transforms_dict["val"],
                    cache_dir=f"{cache_dir}/val"
                ),
                "test": MRIPersistentDataset(
                    test_dataset,
                    transform=transforms_dict["test"],
                    cache_dir=f"{cache_dir}/test"
                )
            }
        else:  # "none": no caching; use original dataset implementation
            datasets_obj = {
                "train": MRISequenceDataset(
                    train_dataset,
                    transform=transforms_dict["train"],
                    img_size=config["data"]["img_size"],
                    label_dict=label_dict,
                    proportion=config["data"]["proportion"]
                ),
                "val": MRISequenceDataset(
                    val_dataset,
                    transform=transforms_dict["val"],
                    img_size=config["data"]["img_size"],
                    label_dict=label_dict
                ),
                "test": MRISequenceDataset(
                    test_dataset,
                    transform=transforms_dict["test"],
                    img_size=config["data"]["img_size"],
                    label_dict=label_dict
                )
            }
    else:
        # Fall back to original dataset implementation if not using MONAI
        datasets_obj = {
            "train": MRISequenceDataset(
                train_dataset,
                transform=transforms_dict["train"],
                img_size=config["data"]["img_size"],
                label_dict=label_dict,
                proportion=config["data"]["proportion"]
            ),
            "val": MRISequenceDataset(
                val_dataset,
                transform=transforms_dict["val"],
                img_size=config["data"]["img_size"],
                label_dict=label_dict
            ),
            "test": MRISequenceDataset(
                test_dataset,
                transform=transforms_dict["test"],
                img_size=config["data"]["img_size"],
                label_dict=label_dict
            )
        }

    print(f"Dataset splits: train={len(datasets_obj['train'])}, val={len(datasets_obj['val'])}, test={len(datasets_obj['test'])}")

    return datasets_obj, label_dict, hierarchical_dicts, num_normalizer


def create_dataloaders(
        datasets: Dict[str, Union[MRIMonaiDataset, torch.utils.data.Dataset]],
        config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and test sets.

    Args:
        datasets: Dictionary mapping split names to dataset objects.
                  These can be MONAI datasets (e.g., MRIMonaiDataset) or standard PyTorch datasets.
        config: Configuration dictionary with batch size and dataloader parameters.

    Returns:
        Dictionary mapping split names to DataLoader objects.
    """
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    pin_memory = config["data"]["pin_memory"]

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=config["data"]["shuffle"],
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }

    return dataloaders


def get_class_weights(
        dataset: MRISequenceDataset,
        num_classes: int,
        method: str = "balanced"
) -> torch.Tensor:
    """
    Calculate class weights to mitigate class imbalance during training.

    Args:
        dataset: Training dataset object.
        num_classes: Total number of classes.
        method: Weighting method ('balanced', 'inverse', 'sqrt_inverse').

    Returns:
        Tensor containing normalized class weights.
    """
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        class_counts[label] += 1

    class_counts = torch.clamp(class_counts, min=1.0)

    if method == "balanced":
        weights = len(dataset) / (num_classes * class_counts)
    elif method == "inverse":
        weights = 1.0 / class_counts
    elif method == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(class_counts)
    else:
        raise ValueError(f"Unsupported class weighting method: {method}")

    weights = weights * (num_classes / weights.sum())

    return weights
