#!/usr/bin/env python3
"""
Updated lazy loading dataset implementation with stratified splitting support.
Save as: utils/lazy_dataset_stratified.py (or update your existing lazy_dataset.py)
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
from utils.incremental_merger import IncrementalBatchMerger


class LazyMergedDataset(Dataset):
    """Lazy dataset that loads individual samples from a merged pickle file."""

    def __init__(self, merged_file: str, indices: List[int],
                 transform=None, label_dict: Optional[Dict[str, int]] = None):
        """
        Initialize lazy merged dataset.

        Args:
            merged_file: Path to merged pickle file
            indices: List of sample indices to use
            transform: Image transforms to apply
            label_dict: Dictionary mapping class names to indices
        """
        self.merged_file = Path(merged_file)
        self.indices = indices
        self.transform = transform
        self.label_dict = label_dict

        # Cache for the dataset (load once per worker)
        self._dataset_cache = None
        self._worker_id = None

    def _ensure_dataset_loaded(self):
        """Load dataset if not already cached."""
        # Check if we're in a new worker process
        import torch.utils.data.get_worker_info
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if self._dataset_cache is None or self._worker_id != worker_id:
            print(f"Loading merged dataset (worker {worker_id})...")
            with open(self.merged_file, 'rb') as f:
                self._dataset_cache = pickle.load(f)
            self._worker_id = worker_id
            print(f"Loaded {len(self._dataset_cache)} samples in worker {worker_id}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Ensure dataset is loaded
        self._ensure_dataset_loaded()

        # Get the actual sample index
        actual_idx = self.indices[idx]

        # Get the sample
        sample = self._dataset_cache[actual_idx]

        # Process image
        img = sample.img
        if self.transform:
            img = self.transform(img)

        # Return as dictionary
        return {
            'img': img,
            'input_ids': torch.tensor(sample.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(sample.attention_mask, dtype=torch.long),
            'numerical_attributes': torch.tensor(sample.numerical_attributes, dtype=torch.float),
            'label': torch.tensor(sample.label, dtype=torch.long)
        }


class StratifiedLazyMRIDataset(Dataset):
    """Extended lazy dataset with stratified split support for batch files."""

    def __init__(self, data_dir: str, orientation: str, indices: List[int],
                 class_indices: Dict[str, List[int]] = None,
                 transform=None, label_dict: Optional[Dict[str, int]] = None):
        """
        Initialize stratified lazy dataset.

        Args:
            data_dir: Base directory containing the dataset
            orientation: Orientation (TRA, COR, SAG)
            indices: List of sample indices to use
            class_indices: Dictionary mapping class names to sample indices
            transform: Image transforms to apply
            label_dict: Dictionary mapping class names to indices
        """
        self.data_dir = Path(data_dir)
        self.orientation = orientation
        self.indices = indices
        self.class_indices = class_indices
        self.transform = transform
        self.label_dict = label_dict

        # Build index mapping
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

        # Filter to only include requested indices
        self.filtered_indices = [idx for idx in self.indices if idx < self.total_samples]
        if len(self.filtered_indices) < len(self.indices):
            print(f"Warning: Some indices out of range. Using {len(self.filtered_indices)} samples.")

        # Cache for recently loaded batches
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
            gc.collect()

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

        # Return as dictionary
        return {
            'img': img,
            'input_ids': torch.tensor(sample.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(sample.attention_mask, dtype=torch.long),
            'numerical_attributes': torch.tensor(sample.numerical_attributes, dtype=torch.float),
            'label': torch.tensor(sample.label, dtype=torch.long)
        }

    @staticmethod
    def create_stratified_splits(class_indices: Dict[str, List[int]],
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_state: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        Create stratified train/val/test splits ensuring all classes are represented.

        Args:
            class_indices: Dictionary mapping class names to sample indices
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        np.random.seed(random_state)

        train_indices = []
        val_indices = []
        test_indices = []

        print("Creating stratified splits...")
        print(f"Target ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

        # Statistics
        class_stats = []

        for class_name, indices in tqdm(class_indices.items(), desc="Stratifying classes"):
            n_samples = len(indices)

            # Ensure at least one sample per class in train if possible
            if n_samples >= 3:
                # Shuffle indices for this class
                shuffled = np.random.permutation(indices).tolist()

                # Calculate split points
                train_end = int(n_samples * train_ratio)
                val_end = train_end + int(n_samples * val_ratio)

                # Ensure at least 1 sample in each split
                train_end = max(1, train_end)
                val_end = max(train_end + 1, val_end)

                class_train = shuffled[:train_end]
                class_val = shuffled[train_end:val_end]
                class_test = shuffled[val_end:]

                train_indices.extend(class_train)
                val_indices.extend(class_val)
                test_indices.extend(class_test)

                class_stats.append({
                    'class': class_name,
                    'total': n_samples,
                    'train': len(class_train),
                    'val': len(class_val),
                    'test': len(class_test)
                })
            elif n_samples == 2:
                # Put one in train, one in val
                train_indices.extend(indices[:1])
                val_indices.extend(indices[1:2])

                class_stats.append({
                    'class': class_name,
                    'total': n_samples,
                    'train': 1,
                    'val': 1,
                    'test': 0
                })
            elif n_samples == 1:
                # Put the single sample in train
                train_indices.extend(indices)

                class_stats.append({
                    'class': class_name,
                    'total': n_samples,
                    'train': 1,
                    'val': 0,
                    'test': 0
                })

        # Shuffle final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Print statistics
        print(f"\nSplit statistics:")
        print(f"Total samples: {len(train_indices) + len(val_indices) + len(test_indices)}")
        print(
            f"Train: {len(train_indices)} ({len(train_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)")
        print(
            f"Val: {len(val_indices)} ({len(val_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)")
        print(
            f"Test: {len(test_indices)} ({len(test_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)")

        # Show per-class distribution for first few classes
        print(f"\nPer-class distribution (first 5 classes):")
        for stat in class_stats[:5]:
            print(f"  {stat['class']}: total={stat['total']}, "
                  f"train={stat['train']}, val={stat['val']}, test={stat['test']}")
        if len(class_stats) > 5:
            print(f"  ... and {len(class_stats) - 5} more classes")

        return train_indices, val_indices, test_indices


class LazyMRIDatasetWrapper:
    """Updated wrapper with stratified splitting support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config["data"]["dataset_dir"])
        self.orientation = config["data"]["orientation"]

    def create_datasets(self, transforms_dict: Dict[str, Any]) -> Tuple[Dict[str, Dataset], Dict[str, int]]:
        """Create train/val/test datasets with lazy loading and stratified splits."""

        # Check or create merged dataset
        merger = IncrementalBatchMerger(
            self.data_dir,
            self.orientation,
            chunk_size=self.config.get("merge_chunk_size", 10000)
        )

        try:
            merged_file, class_indices = merger.check_or_create_merged_dataset()
        except Exception as e:
            print(f"Could not create/load merged dataset: {e}")
            # Fall back to batch loading
            merged_file = None
            # Build class indices from batches
            class_indices = self._build_class_indices_from_batches()

        # Load label dictionary
        label_dict_path = self.data_dir / self.orientation / f"label_dict_{self.orientation}.json"
        if not label_dict_path.exists():
            raise FileNotFoundError(f"Label dictionary not found: {label_dict_path}")

        with open(label_dict_path, 'r') as f:
            label_dict = json.load(f)

        # Check if we're using merged file or batches
        if merged_file and merged_file.exists():
            print(f"Using merged dataset file: {merged_file}")
            return self._create_from_merged_file(merged_file, class_indices,
                                                 transforms_dict, label_dict)
        else:
            print("Using batch files with lazy loading")
            return self._create_from_batches(class_indices, transforms_dict, label_dict)

    def _build_class_indices_from_batches(self) -> Dict[str, List[int]]:
        """Build class indices by scanning batch files."""
        batch_dir = self.data_dir / self.orientation / "batches"
        batch_files = sorted(batch_dir.glob("dataset_batch_*.pkl"))

        class_indices = {}
        global_idx = 0

        print("Building class indices from batch files...")
        for batch_file in tqdm(batch_files, desc="Scanning batches"):
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)

            for sample in batch_data:
                label_str = str(sample.label)
                if label_str not in class_indices:
                    class_indices[label_str] = []
                class_indices[label_str].append(global_idx)
                global_idx += 1

            del batch_data
            gc.collect()

        return class_indices

    def _create_from_merged_file(self, merged_file: Path, class_indices: Dict[str, List[int]],
                                 transforms_dict: Dict[str, Any], label_dict: Dict[str, int]):
        """Create datasets from merged file with lazy loading."""

        # Create stratified splits
        train_indices, val_indices, test_indices = StratifiedLazyMRIDataset.create_stratified_splits(
            class_indices,
            train_ratio=self.config["data"]["train_split"],
            val_ratio=self.config["data"]["val_split"],
            test_ratio=self.config["data"]["test_split"],
            random_state=self.config.get("seed", 42)
        )

        # Create lazy datasets that load from the merged file
        datasets = {
            "train": LazyMergedDataset(
                merged_file=str(merged_file),
                indices=train_indices,
                transform=transforms_dict["train"],
                label_dict=label_dict
            ),
            "val": LazyMergedDataset(
                merged_file=str(merged_file),
                indices=val_indices,
                transform=transforms_dict["val"],
                label_dict=label_dict
            ),
            "test": LazyMergedDataset(
                merged_file=str(merged_file),
                indices=test_indices,
                transform=transforms_dict["test"],
                label_dict=label_dict
            )
        }

        return datasets, label_dict

    def _create_from_batches(self, class_indices: Dict[str, List[int]],
                             transforms_dict: Dict[str, Any], label_dict: Dict[str, int]):
        """Create datasets from batch files."""

        # Create stratified splits
        train_indices, val_indices, test_indices = StratifiedLazyMRIDataset.create_stratified_splits(
            class_indices,
            train_ratio=self.config["data"]["train_split"],
            val_ratio=self.config["data"]["val_split"],
            test_ratio=self.config["data"]["test_split"],
            random_state=self.config.get("seed", 42)
        )

        # Use stratified lazy dataset for batch loading
        datasets = {
            "train": StratifiedLazyMRIDataset(
                data_dir=str(self.data_dir),
                orientation=self.orientation,
                indices=train_indices,
                transform=transforms_dict["train"],
                label_dict=label_dict,
                class_indices=class_indices
            ),
            "val": StratifiedLazyMRIDataset(
                data_dir=str(self.data_dir),
                orientation=self.orientation,
                indices=val_indices,
                transform=transforms_dict["val"],
                label_dict=label_dict,
                class_indices=class_indices
            ),
            "test": StratifiedLazyMRIDataset(
                data_dir=str(self.data_dir),
                orientation=self.orientation,
                indices=test_indices,
                transform=transforms_dict["test"],
                label_dict=label_dict,
                class_indices=class_indices
            )
        }

        return datasets, label_dict


def create_lazy_datasets(config: Dict[str, Any]) -> Tuple[Dict[str, Dataset], Dict[str, int], Any]:
    """
    Main entry point for creating lazy datasets with automatic merging.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (datasets_dict, label_dict, num_normalizer)
    """
    # Import here to avoid circular dependency
    from utils.data_loader import prepare_transforms

    # Prepare transforms
    transforms_dict = prepare_transforms(config)

    # Check if orientation is specified
    if "orientation" not in config["data"]:
        raise ValueError("Orientation must be specified in config['data']['orientation']")

    # Create wrapper and datasets
    wrapper = LazyMRIDatasetWrapper(config)
    datasets, label_dict = wrapper.create_datasets(transforms_dict)

    # For compatibility with existing code
    num_normalizer = None  # You can add normalization if needed

    return datasets, label_dict, num_normalizer