from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict
import numpy as np
import torch


@dataclass
class dicomData:
    """Class for holding raw DICOM data before encoding."""
    img: np.ndarray
    text_attributes: str
    numerical_attributes: List[Union[int, float]]
    label: str


@dataclass
class encodedSample:
    """Class for holding encoded DICOM data after preprocessing."""
    img: np.ndarray
    input_ids: List[int]
    attention_mask: List[int]
    numerical_attributes: List[Union[int, float]]
    label: int


class MRISequenceDataset(torch.utils.data.Dataset):
    """Dataset class for MRI sequence classification."""

    def __init__(
            self,
            data: List[encodedSample],
            transform=None,
            img_size: int = 224,
            label_dict: Optional[Dict[str, int]] = None,
            proportion: Optional[float] = None
    ):
        """
        Initialize the dataset.

        Args:
            data: List of encoded samples
            transform: Image transforms to apply
            img_size: Size to resize images to
            label_dict: Dictionary mapping class names to indices
            proportion: Optional subsample proportion (0-1)
        """
        self.data = data
        self.transform = transform
        self.img_size = img_size
        self.label_dict = label_dict

        # Subsample if proportion is provided
        if proportion is not None and 0 < proportion < 1:
            num_samples = int(len(self.data) * proportion)
            # Use stratified sampling to maintain class distribution
            indices = self._stratified_sample(num_samples)
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if idx < 5:
            print(f"\nDEBUG Sample {idx}:")
            print(f"  Type: {type(sample)}")
            print(f"  Image shape: {sample.img.shape}")
            print(f"  Image dtype: {sample.img.dtype}")
            print(f"  Input IDs: type={type(sample.input_ids)}, len={len(sample.input_ids)}")
            print(f"  Attention mask: type={type(sample.attention_mask)}, len={len(sample.attention_mask)}")
            print(f"  Numerical attrs: type={type(sample.numerical_attributes)}, len={len(sample.numerical_attributes)}")
            print(f"  Label: {sample.label}, type={type(sample.label)}")

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

    def _stratified_sample(self, num_samples):
        """Perform stratified sampling to maintain class distribution."""
        labels = [sample.label for sample in self.data]
        unique_labels = set(labels)

        # Count samples per class
        class_counts = {label: labels.count(label) for label in unique_labels}

        # Calculate target count per class
        total = len(self.data)
        class_proportions = {label: count / total for label, count in class_counts.items()}
        target_counts = {label: int(prop * num_samples) for label, prop in class_proportions.items()}

        # Ensure we get exactly num_samples
        remaining = num_samples - sum(target_counts.values())
        if remaining > 0:
            # Distribute remaining samples to largest classes
            sorted_classes = sorted(unique_labels, key=lambda x: class_counts[x], reverse=True)
            for i in range(remaining):
                target_counts[sorted_classes[i % len(sorted_classes)]] += 1

        # Select indices for each class
        indices = []
        for label in unique_labels:
            label_indices = [i for i, sample in enumerate(self.data) if sample.label == label]
            selected = np.random.choice(label_indices, target_counts[label], replace=False)
            indices.extend(selected)

        # Shuffle the indices
        np.random.shuffle(indices)
        return indices