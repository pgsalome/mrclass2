from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import numpy as np
from monai.data import CacheDataset, PersistentDataset, Dataset
from monai.transforms import Compose, Lambda

from utils.dataclass import encodedSample
from utils.intensity_normalization import IntensityNormTransform


class MRIMonaiDataset(CacheDataset):
    """MONAI-based cached dataset for MRI sequence classification."""

    def __init__(
            self,
            data: List[encodedSample],
            transform=None,
            cache_rate: float = 1.0,
            num_workers: int = 0,
            progress: bool = True
    ):
        """
        Initialize the MONAI dataset.

        Args:
            data: List of encoded samples
            transform: Transform to apply
            cache_rate: Percentage of data to cache (0.0-1.0)
            num_workers: Number of workers for cache creation
            progress: Whether to show progress bar
        """
        # Convert encodedSample objects to dictionaries for MONAI
        data_dicts = []
        for sample in data:
            data_dict = {
                "img": sample.img,
                "input_ids": sample.input_ids,
                "attention_mask": sample.attention_mask,
                "numerical_attributes": sample.numerical_attributes,
                "label": sample.label
            }
            data_dicts.append(data_dict)

        # Create transform that handles all data types
        def transform_wrapper(data_dict):
            # Apply image transform if provided
            if transform is not None:
                data_dict["img"] = transform(data_dict["img"])

            # Convert all to tensors
            data_dict["input_ids"] = torch.tensor(data_dict["input_ids"], dtype=torch.long)
            data_dict["attention_mask"] = torch.tensor(data_dict["attention_mask"], dtype=torch.long)
            data_dict["numerical_attributes"] = torch.tensor(data_dict["numerical_attributes"], dtype=torch.float)
            data_dict["label"] = torch.tensor(data_dict["label"], dtype=torch.long)

            return data_dict

        monai_transform = Compose([Lambda(transform_wrapper)])

        # Initialize the CacheDataset
        super().__init__(
            data=data_dicts,
            transform=monai_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress
        )


class MRIPersistentDataset(PersistentDataset):
    """MONAI-based persistent dataset for MRI sequence classification."""

    def __init__(
            self,
            data: List[encodedSample],
            transform=None,
            cache_dir: str = "./persistent_cache",
            progress: bool = True
    ):
        """
        Initialize the persistent dataset.

        Args:
            data: List of encoded samples
            transform: Transform to apply
            cache_dir: Directory to store persistent cache
            progress: Whether to show progress bar
        """
        # Convert encodedSample objects to dictionaries for MONAI
        data_dicts = []
        for i, sample in enumerate(data):
            data_dict = {
                "img": sample.img,
                "input_ids": sample.input_ids,
                "attention_mask": sample.attention_mask,
                "numerical_attributes": sample.numerical_attributes,
                "label": sample.label,
                # Add unique ID for cache lookup
                "id": f"sample_{i}"
            }
            data_dicts.append(data_dict)

        # Create transform that handles all data types - same as above
        def transform_wrapper(data_dict):
            if transform is not None:
                data_dict["img"] = transform(data_dict["img"])

            data_dict["input_ids"] = torch.tensor(data_dict["input_ids"], dtype=torch.long)
            data_dict["attention_mask"] = torch.tensor(data_dict["attention_mask"], dtype=torch.long)
            data_dict["numerical_attributes"] = torch.tensor(data_dict["numerical_attributes"], dtype=torch.float)
            data_dict["label"] = torch.tensor(data_dict["label"], dtype=torch.long)

            return data_dict

        monai_transform = Compose([Lambda(transform_wrapper)])

        # Initialize the PersistentDataset
        super().__init__(
            data=data_dicts,
            transform=monai_transform,
            cache_dir=cache_dir,
            hash_func=lambda x: x["id"],
            progress=progress
        )