#!/usr/bin/env python3
"""
Lazy loading dataset implementation for MRI sequence classification.
This loads data on-demand to avoid memory issues with large datasets.
"""

import pickle
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import gc

# Import the encodedSample class
from utils.dataclass import encodedSample


class LazyMRIDataset(Dataset):
    """
    Lazy loading dataset that loads samples on-demand from batch files.
    This avoids loading the entire dataset into memory at once.
    """

    def __init__(
            self,
            data_dir: str,
            orientation: str,
            indices: List[int],
            transform=None,
            label_dict: Optional[Dict[str, int]] = None
    ):
        """
        Initialize lazy dataset.

        Args:
            data_dir: Base directory containing the dataset
            orientation: Orientation (TRA, COR, SAG)
            indices: List of sample indices to use (for train/val/test split)
            transform: Image transforms to apply
            label_dict: Dictionary mapping class names to indices
        """
        self.data_dir = Path(data_dir)
        self.orientation = orientation
        self.indices = indices
        self.transform = transform
        self.label_dict = label_dict

        # Build index mapping: sample_idx -> (batch_file, position_in_batch)
        self._build_index()

    def _build_index(self):
        """Build index mapping for efficient sample access."""
        self.batch_dir = self.data_dir / self.orientation / "batches"
        self.batch_files = sorted(self.batch_dir.glob("dataset_batch_*.pkl"))

        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {self.batch_dir}")

        # Create mapping: global_idx -> (batch_idx, local_idx)
        self.index_map = {}
        global_idx = 0

        print(f"Building index for {self.orientation} dataset...")
        for batch_idx, batch_file in enumerate(self.batch_files):
            # Just peek at the batch to get its size
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                batch_size = len(batch_data)

            for local_idx in range(batch_size):
                self.index_map[global_idx] = (batch_idx, local_idx)
                global_idx += 1

        self.total_samples = global_idx
        print(f"Found {self.total_samples} samples across {len(self.batch_files)} batches")

        # Filter index_map to only include requested indices
        self.filtered_indices = [idx for idx in self.indices if idx < self.total_samples]
        if len(self.filtered_indices) < len(self.indices):
            print(f"Warning: Some indices out of range. Using {len(self.filtered_indices)} samples.")

        # Cache for recently loaded batches (keep last N batches in memory)
        self.batch_cache = {}
        self.cache_size = 3  # Adjust based on available memory
        self.cache_order = []

    def _load_batch(self, batch_idx: int):
        """Load a batch file, using cache if available."""
        if batch_idx in self.batch_cache:
            # Move to end of cache order (LRU)
            self.cache_order.remove(batch_idx)
            self.cache_order.append(batch_idx)
            return self.batch_cache[batch_idx]

        # Load batch from disk
        batch_file = self.batch_files[batch_idx]
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        # Add to cache
        self.batch_cache[batch_idx] = batch_data
        self.cache_order.append(batch_idx)

        # Remove oldest batch if cache is full
        if len(self.batch_cache) > self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.batch_cache[oldest_idx]
            gc.collect()  # Force garbage collection

        return batch_data

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Get the actual sample index
        actual_idx = self.filtered_indices[idx]

        # Get batch and position info
        batch_idx, local_idx = self.index_map[actual_idx]

        # Load the batch
        batch_data = self._load_batch(batch_idx)

        # Get the sample
        sample = batch_data[local_idx]

        # Process image
        img = sample.img
        if self.transform:
            img = self.transform(img)

        # Return as dictionary for easier batch handling
        return {
            'img': img,
            'input_ids': torch.tensor(sample.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(sample.attention_mask, dtype=torch.long),
            'numerical_attributes': torch.tensor(sample.numerical_attributes, dtype=torch.float),
            'label': torch.tensor(sample.label, dtype=torch.long)
        }


class LazyMRIDatasetWrapper:
    """
    Wrapper to create train/val/test splits using lazy loading.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config["data"]["dataset_dir"])
        self.orientation = config["data"]["orientation"]

    def create_datasets(self, transforms_dict: Dict[str, Any]) -> Tuple[Dict[str, Dataset], Dict[str, int]]:
        """
        Create train/val/test datasets with lazy loading.

        Returns:
            Tuple of (datasets_dict, label_dict)
        """
        # Load label dictionary
        label_dict_path = self.data_dir / self.orientation / f"label_dict_{self.orientation}.json"
        with open(label_dict_path, 'r') as f:
            label_dict = json.load(f)

        # Get total number of samples
        batch_dir = self.data_dir / self.orientation / "batches"
        batch_files = sorted(batch_dir.glob("dataset_batch_*.pkl"))

        total_samples = 0
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                total_samples += len(batch_data)

        print(f"Total samples available: {total_samples}")

        # Create indices for splits
        all_indices = list(range(total_samples))

        # Calculate split sizes
        train_ratio = self.config["data"]["train_split"]
        val_ratio = self.config["data"]["val_split"]
        test_ratio = self.config["data"]["test_split"]

        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size

        # For simplicity, we'll do sequential splitting
        # In production, you might want stratified splitting
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size + val_size]
        test_indices = all_indices[train_size + val_size:]

        print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # Create lazy datasets
        datasets = {
            "train": LazyMRIDataset(
                data_dir=self.config["data"]["dataset_dir"],
                orientation=self.orientation,
                indices=train_indices,
                transform=transforms_dict["train"],
                label_dict=label_dict
            ),
            "val": LazyMRIDataset(
                data_dir=self.config["data"]["dataset_dir"],
                orientation=self.orientation,
                indices=val_indices,
                transform=transforms_dict["val"],
                label_dict=label_dict
            ),
            "test": LazyMRIDataset(
                data_dir=self.config["data"]["dataset_dir"],
                orientation=self.orientation,
                indices=test_indices,
                transform=transforms_dict["test"],
                label_dict=label_dict
            )
        }

        return datasets, label_dict


def create_lazy_datasets(config: Dict[str, Any]) -> Tuple[Dict[str, Dataset], Dict[str, int], Any]:
    """
    Create datasets using lazy loading approach.

    This is a drop-in replacement for create_datasets() from data_loader.py
    """
    from utils.data_loader import prepare_transforms
    from models.num_encoder import NumericFeatureNormalizer

    # Prepare transforms
    transforms_dict = prepare_transforms(config)

    # Create wrapper and datasets
    wrapper = LazyMRIDatasetWrapper(config)
    datasets, label_dict = wrapper.create_datasets(transforms_dict)

    # For now, return a dummy normalizer
    # In production, you'd compute this from a sample of the data
    num_normalizer = NumericFeatureNormalizer(method="z-score")

    # Return in same format as original create_datasets
    return datasets, label_dict, None, num_normalizer