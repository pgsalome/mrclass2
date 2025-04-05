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

from utils.dataclass import encodedSample, MRISequenceDataset
from utils.io import load_pickle, read_json_config, ensure_dir
from models.num_encoder import NumericFeatureNormalizer
from utils.intensity_normalization import get_intensity_normalizer, IntensityNormTransform
from utils.medical_transforms import prepare_medical_transforms, MedicalImageTransform


class MRITransform:
    """Combined transform for MRI images with optional intensity normalization."""

    def __init__(
            self,
            transform: transforms.Compose,
            norm_transform: Optional[IntensityNormTransform] = None
    ):
        """
        Initialize the combined transform.

        Args:
            transform: Base torchvision transform
            norm_transform: Optional intensity normalization transform
        """
        self.transform = transform
        self.norm_transform = norm_transform

    def __call__(self, img: np.ndarray, label: Optional[int] = None) -> torch.Tensor:
        """
        Apply transforms to an image.

        Args:
            img: Input image
            label: Optional label (for class-specific normalization)

        Returns:
            Transformed image tensor
        """
        # Apply intensity normalization if available
        if self.norm_transform is not None:
            img = self.norm_transform(img, label)

        # Apply standard transforms
        return self.transform(img)


def prepare_transforms(
        config: Dict[str, Any],
        intensity_normalizer=None,
        inv_label_dict: Optional[Dict[int, str]] = None,
        use_medical_transforms: bool = True
) -> Dict[str, Union[MedicalImageTransform, MRITransform]]:
    """
    Prepare image transformations for training and validation.

    Args:
        config: Configuration dictionary
        intensity_normalizer: Optional intensity normalizer
        inv_label_dict: Optional inverse label dictionary (for class-specific normalization)
        use_medical_transforms: Whether to use medical-specific transforms

    Returns:
        Dictionary with train and val transforms
    """
    # Check if we should use medical transforms
    if use_medical_transforms:
        medical_transforms = prepare_medical_transforms(config)
        if medical_transforms:
            return medical_transforms

    # Fall back to standard transforms if medical transforms aren't available
    img_size = config["data"]["img_size"]

    # Create intensity normalization transform if needed
    norm_transform = None
    if intensity_normalizer is not None:
        norm_transform = IntensityNormTransform(intensity_normalizer, inv_label_dict)

    # Training transforms with augmentation
    train_base_transforms = transforms.Compose([
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
    val_base_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Create combined transforms
    return {
        "train": MRITransform(train_base_transforms, norm_transform),
        "val": MRITransform(val_base_transforms, norm_transform),
        "test": MRITransform(val_base_transforms, norm_transform)
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[List[encodedSample], Dict[str, int]]:
    """
    Load the encoded dataset and label dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple containing dataset and label dictionary
    """
    data_dir = Path(config["data"]["dataset_dir"])
    dataset_path = data_dir / config["data"]["dataset_name"]
    label_dict_path = data_dir / config["data"]["label_dict_name"]

    # Load dataset
    dataset = load_pickle(str(dataset_path))

    # Load label dictionary
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
        dataset: List of encoded samples
        label_dict: Original label dictionary

    Returns:
        Dictionary of hierarchical label dictionaries
    """
    # Invert label dictionary to get class names from indices
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Extract base contrast (e.g., T1, T2, DWI)
    base_contrasts = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        # Extract base contrast from the class name (everything before first hyphen)
        base_contrast = class_name.split('-')[0] if '-' in class_name else class_name
        base_contrasts.add(base_contrast)

    # Create base contrast dictionary
    base_contrast_dict = {contrast: i for i, contrast in enumerate(sorted(base_contrasts))}

    # Extract sequence type (e.g., SE, GE, IRGE)
    sequence_types = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')
        if len(parts) > 1:
            # Check if the second part is a common sequence type
            if any(seq_type in parts[1:] for seq_type in ['SE', 'GE', 'IRGE', 'IRSE']):
                for part in parts[1:]:
                    if part in ['SE', 'GE', 'IRGE', 'IRSE']:
                        sequence_types.add(part)
                        break

    # Create sequence type dictionary
    sequence_type_dict = {seq_type: i for i, seq_type in enumerate(sorted(sequence_types))}

    # Extract fat suppression info
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
        dataset: List of encoded samples
        label_dict: Original label dictionary
        hierarchical_dicts: Dictionary of hierarchical label dictionaries

    Returns:
        List of dictionaries with hierarchical labels for each sample
    """
    inv_label_dict = {v: k for k, v in label_dict.items()}
    hierarchical_labels = []

    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')

        # Extract base contrast
        base_contrast = parts[0]
        base_contrast_label = hierarchical_dicts["base_contrast"].get(base_contrast, 0)

        # Extract sequence type
        sequence_type = "Unknown"
        for part in parts[1:]:
            if part in hierarchical_dicts["sequence_type"]:
                sequence_type = part
                break
        sequence_type_label = hierarchical_dicts["sequence_type"].get(sequence_type, 0)

        # Extract fat suppression
        fat_sup_label = hierarchical_dicts["fat_suppression"]["Yes"] if "FS" in parts else \
        hierarchical_dicts["fat_suppression"]["No"]

        hierarchical_labels.append({
            "base_contrast": base_contrast_label,
            "sequence_type": sequence_type_label,
            "fat_suppression": fat_sup_label,
            "fine_class": dataset[idx].label  # Original fine-grained class
        })

    return hierarchical_labels


def create_datasets(
        config: Dict[str, Any],
        hierarchical: bool = False
) -> Tuple[
    Dict[str, MRISequenceDataset], Dict[str, int], Optional[Dict[str, Dict[str, int]]], NumericFeatureNormalizer]:
    """
    Create training, validation and test datasets.

    Args:
        config: Configuration dictionary
        hierarchical: Whether to prepare hierarchical labels

    Returns:
        Tuple containing:
        - Dictionary mapping split names to datasets
        - Label dictionary
        - Hierarchical label dictionaries (if hierarchical=True)
        - Numeric feature normalizer
    """
    # Load dataset and label dictionary
    dataset, label_dict = load_dataset(config)

    # Invert label dictionary for intensity normalization
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Initialize intensity normalizer if enabled
    intensity_normalizer = get_intensity_normalizer(config["data"])

    # Check if we should use medical transforms
    use_medical_transforms = config.get("data", {}).get("use_medical_transforms", True)

    # Prepare image transforms
    transforms_dict = prepare_transforms(
        config,
        intensity_normalizer,
        inv_label_dict,
        use_medical_transforms
    )

    # Prepare hierarchical labels if needed
    hierarchical_dicts = None
    hierarchical_labels = None
    if hierarchical:
        hierarchical_dicts = prepare_hierarchical_labels(dataset, label_dict)
        hierarchical_labels = extract_hierarchical_labels(dataset, label_dict, hierarchical_dicts)

    # Normalize numerical features
    num_normalizer = NumericFeatureNormalizer(method="z-score")
    num_features = [sample.numerical_attributes for sample in dataset]
    num_normalizer.fit(num_features)

    # Update numerical features in the dataset
    for i in range(len(dataset)):
        dataset[i].numerical_attributes = num_normalizer.transform([dataset[i].numerical_attributes])[0].tolist()

    # Split into train, validation, and test sets
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]

    # Stratified split based on labels
    labels = [sample.label for sample in dataset]

    # First split into train and temp (val+test)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=config["seed"],
        stratify=labels
    )

    # Then split temp into val and test
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

    # Create datasets for each split
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Create MRISequenceDataset objects
    datasets = {
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

    print(f"Dataset splits: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    return datasets, label_dict, hierarchical_dicts, num_normalizer


def create_dataloaders(
        datasets: Dict[str, MRISequenceDataset],
        config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and test sets.

    Args:
        datasets: Dictionary mapping split names to datasets
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataloaders
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
    Calculate class weights for handling class imbalance.

    Args:
        dataset: Training dataset
        num_classes: Number of classes
        method: Method to calculate weights ('balanced', 'inverse', 'sqrt_inverse')

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        class_counts[label] += 1

    # Ensure no zeros to avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    if method == "balanced":
        # Weight inversely proportional to class frequency
        weights = len(dataset) / (num_classes * class_counts)
    elif method == "inverse":
        # Simple inverse weighting
        weights = 1.0 / class_counts
    elif method == "sqrt_inverse":
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / torch.sqrt(class_counts)
    else:
        raise ValueError(f"Unsupported class weighting method: {method}")

    # Normalize weights to sum to number of classes
    weights = weights * (num_classes / weights.sum())

    return weights.data
    import DataLoader, Dataset, random_split


import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from utils.dataclass import encodedSample, MRISequenceDataset
from utils.io import load_pickle, read_json_config, ensure_dir
from models.num_encoder import NumericFeatureNormalizer
from utils.intensity_normalization import get_intensity_normalizer, IntensityNormTransform


class MRITransform:
    """Combined transform for MRI images with optional intensity normalization."""

    def __init__(
            self,
            transform: transforms.Compose,
            norm_transform: Optional[IntensityNormTransform] = None
    ):
        """
        Initialize the combined transform.

        Args:
            transform: Base torchvision transform
            norm_transform: Optional intensity normalization transform
        """
        self.transform = transform
        self.norm_transform = norm_transform

    def __call__(self, img: np.ndarray, label: Optional[int] = None) -> torch.Tensor:
        """
        Apply transforms to an image.

        Args:
            img: Input image
            label: Optional label (for class-specific normalization)

        Returns:
            Transformed image tensor
        """
        # Apply intensity normalization if available
        if self.norm_transform is not None:
            img = self.norm_transform(img, label)

        # Apply standard transforms
        return self.transform(img)


def prepare_transforms(
        config: Dict[str, Any],
        intensity_normalizer=None,
        inv_label_dict: Optional[Dict[int, str]] = None
) -> Dict[str, MRITransform]:
    """
    Prepare image transformations for training and validation.

    Args:
        config: Configuration dictionary
        intensity_normalizer: Optional intensity normalizer
        inv_label_dict: Optional inverse label dictionary (for class-specific normalization)

    Returns:
        Dictionary with train and val transforms
    """
    img_size = config["data"]["img_size"]

    # Create intensity normalization transform if needed
    norm_transform = None
    if intensity_normalizer is not None:
        norm_transform = IntensityNormTransform(intensity_normalizer, inv_label_dict)

    # Training transforms with augmentation
    train_base_transforms = transforms.Compose([
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
    val_base_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Create combined transforms
    return {
        "train": MRITransform(train_base_transforms, norm_transform),
        "val": MRITransform(val_base_transforms, norm_transform),
        "test": MRITransform(val_base_transforms, norm_transform)
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[List[encodedSample], Dict[str, int]]:
    """
    Load the encoded dataset and label dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple containing dataset and label dictionary
    """
    data_dir = Path(config["data"]["dataset_dir"])
    dataset_path = data_dir / config["data"]["dataset_name"]
    label_dict_path = data_dir / config["data"]["label_dict_name"]

    # Load dataset
    dataset = load_pickle(str(dataset_path))

    # Load label dictionary
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
        dataset: List of encoded samples
        label_dict: Original label dictionary

    Returns:
        Dictionary of hierarchical label dictionaries
    """
    # Invert label dictionary to get class names from indices
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Extract base contrast (e.g., T1, T2, DWI)
    base_contrasts = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        # Extract base contrast from the class name (everything before first hyphen)
        base_contrast = class_name.split('-')[0] if '-' in class_name else class_name
        base_contrasts.add(base_contrast)

    # Create base contrast dictionary
    base_contrast_dict = {contrast: i for i, contrast in enumerate(sorted(base_contrasts))}

    # Extract sequence type (e.g., SE, GE, IRGE)
    sequence_types = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')
        if len(parts) > 1:
            # Check if the second part is a common sequence type
            if any(seq_type in parts[1:] for seq_type in ['SE', 'GE', 'IRGE', 'IRSE']):
                for part in parts[1:]:
                    if part in ['SE', 'GE', 'IRGE', 'IRSE']:
                        sequence_types.add(part)
                        break

    # Create sequence type dictionary
    sequence_type_dict = {seq_type: i for i, seq_type in enumerate(sorted(sequence_types))}

    # Extract fat suppression info
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
        dataset: List of encoded samples
        label_dict: Original label dictionary
        hierarchical_dicts: Dictionary of hierarchical label dictionaries

    Returns:
        List of dictionaries with hierarchical labels for each sample
    """
    inv_label_dict = {v: k for k, v in label_dict.items()}
    hierarchical_labels = []

    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')

        # Extract base contrast
        base_contrast = parts[0]
        base_contrast_label = hierarchical_dicts["base_contrast"].get(base_contrast, 0)

        # Extract sequence type
        sequence_type = "Unknown"
        for part in parts[1:]:
            if part in hierarchical_dicts["sequence_type"]:
                sequence_type = part
                break
        sequence_type_label = hierarchical_dicts["sequence_type"].get(sequence_type, 0)

        # Extract fat suppression
        fat_sup_label = hierarchical_dicts["fat_suppression"]["Yes"] if "FS" in parts else \
        hierarchical_dicts["fat_suppression"]["No"]

        hierarchical_labels.append({
            "base_contrast": base_contrast_label,
            "sequence_type": sequence_type_label,
            "fat_suppression": fat_sup_label,
            "fine_class": dataset[idx].label  # Original fine-grained class
        })

    return hierarchical_labels


def create_datasets(
        config: Dict[str, Any],
        hierarchical: bool = False
) -> Tuple[
    Dict[str, MRISequenceDataset], Dict[str, int], Optional[Dict[str, Dict[str, int]]], NumericFeatureNormalizer]:
    """
    Create training, validation and test datasets.

    Args:
        config: Configuration dictionary
        hierarchical: Whether to prepare hierarchical labels

    Returns:
        Tuple containing:
        - Dictionary mapping split names to datasets
        - Label dictionary
        - Hierarchical label dictionaries (if hierarchical=True)
        - Numeric feature normalizer
    """
    # Load dataset and label dictionary
    dataset, label_dict = load_dataset(config)

    # Invert label dictionary for intensity normalization
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Initialize intensity normalizer if enabled
    intensity_normalizer = get_intensity_normalizer(config["data"])

    # Prepare image transforms
    transforms_dict = prepare_transforms(config, intensity_normalizer, inv_label_dict)

    # Prepare hierarchical labels if needed
    hierarchical_dicts = None
    hierarchical_labels = None
    if hierarchical:
        hierarchical_dicts = prepare_hierarchical_labels(dataset, label_dict)
        hierarchical_labels = extract_hierarchical_labels(dataset, label_dict, hierarchical_dicts)

    # Normalize numerical features
    num_normalizer = NumericFeatureNormalizer(method="z-score")
    num_features = [sample.numerical_attributes for sample in dataset]
    num_normalizer.fit(num_features)

    # Update numerical features in the dataset
    for i in range(len(dataset)):
        dataset[i].numerical_attributes = num_normalizer.transform([dataset[i].numerical_attributes])[0].tolist()

    # Split into train, validation, and test sets
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]

    # Stratified split based on labels
    labels = [sample.label for sample in dataset]

    # First split into train and temp (val+test)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=config["seed"],
        stratify=labels
    )

    # Then split temp into val and test
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

    # Create datasets for each split
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Create MRISequenceDataset objects
    datasets = {
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

    print(f"Dataset splits: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    return datasets, label_dict, hierarchical_dicts, num_normalizer


def create_dataloaders(
        datasets: Dict[str, MRISequenceDataset],
        config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and test sets.

    Args:
        datasets: Dictionary mapping split names to datasets
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataloaders
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
    Calculate class weights for handling class imbalance.

    Args:
        dataset: Training dataset
        num_classes: Number of classes
        method: Method to calculate weights ('balanced', 'inverse', 'sqrt_inverse')

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        class_counts[label] += 1

    # Ensure no zeros to avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    if method == "balanced":
        # Weight inversely proportional to class frequency
        weights = len(dataset) / (num_classes * class_counts)
    elif method == "inverse":
        # Simple inverse weighting
        weights = 1.0 / class_counts
    elif method == "sqrt_inverse":
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / torch.sqrt(class_counts)
    else:
        raise ValueError(f"Unsupported class weighting method: {method}")

    # Normalize weights to sum to number of classes
    weights = weights * (num_classes / weights.sum())

    return weightsimport
    os


import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from utils.dataclass import encodedSample, MRISequenceDataset
from utils.io import load_pickle, read_json_config, ensure_dir
from models.num_encoder import NumericFeatureNormalizer


def prepare_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Prepare image transformations for training and validation.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with train and val transforms
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
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    return {
        "train": train_transforms,
        "val": val_transforms,
        "test": val_transforms
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[List[encodedSample], Dict[str, int]]:
    """
    Load the encoded dataset and label dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple containing dataset and label dictionary
    """
    data_dir = Path(config["data"]["dataset_dir"])
    dataset_path = data_dir / config["data"]["dataset_name"]
    label_dict_path = data_dir / config["data"]["label_dict_name"]

    # Load dataset
    dataset = load_pickle(str(dataset_path))

    # Load label dictionary
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
        dataset: List of encoded samples
        label_dict: Original label dictionary

    Returns:
        Dictionary of hierarchical label dictionaries
    """
    # Invert label dictionary to get class names from indices
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Extract base contrast (e.g., T1, T2, DWI)
    base_contrasts = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        # Extract base contrast from the class name (everything before first hyphen)
        base_contrast = class_name.split('-')[0] if '-' in class_name else class_name
        base_contrasts.add(base_contrast)

    # Create base contrast dictionary
    base_contrast_dict = {contrast: i for i, contrast in enumerate(sorted(base_contrasts))}

    # Extract sequence type (e.g., SE, GE, IRGE)
    sequence_types = set()
    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')
        if len(parts) > 1:
            # Check if the second part is a common sequence type
            if any(seq_type in parts[1:] for seq_type in ['SE', 'GE', 'IRGE', 'IRSE']):
                for part in parts[1:]:
                    if part in ['SE', 'GE', 'IRGE', 'IRSE']:
                        sequence_types.add(part)
                        break

    # Create sequence type dictionary
    sequence_type_dict = {seq_type: i for i, seq_type in enumerate(sorted(sequence_types))}

    # Extract fat suppression info
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
        dataset: List of encoded samples
        label_dict: Original label dictionary
        hierarchical_dicts: Dictionary of hierarchical label dictionaries

    Returns:
        List of dictionaries with hierarchical labels for each sample
    """
    inv_label_dict = {v: k for k, v in label_dict.items()}
    hierarchical_labels = []

    for idx in range(len(dataset)):
        class_name = inv_label_dict[dataset[idx].label]
        parts = class_name.split('-')

        # Extract base contrast
        base_contrast = parts[0]
        base_contrast_label = hierarchical_dicts["base_contrast"].get(base_contrast, 0)

        # Extract sequence type
        sequence_type = "Unknown"
        for part in parts[1:]:
            if part in hierarchical_dicts["sequence_type"]:
                sequence_type = part
                break
        sequence_type_label = hierarchical_dicts["sequence_type"].get(sequence_type, 0)

        # Extract fat suppression
        fat_sup_label = hierarchical_dicts["fat_suppression"]["Yes"] if "FS" in parts else \
        hierarchical_dicts["fat_suppression"]["No"]

        hierarchical_labels.append({
            "base_contrast": base_contrast_label,
            "sequence_type": sequence_type_label,
            "fat_suppression": fat_sup_label,
            "fine_class": dataset[idx].label  # Original fine-grained class
        })

    return hierarchical_labels


def create_datasets(
        config: Dict[str, Any],
        hierarchical: bool = False
) -> Tuple[
    Dict[str, MRISequenceDataset], Dict[str, int], Optional[Dict[str, Dict[str, int]]], NumericFeatureNormalizer]:
    """
    Create training, validation and test datasets.

    Args:
        config: Configuration dictionary
        hierarchical: Whether to prepare hierarchical labels

    Returns:
        Tuple containing:
        - Dictionary mapping split names to datasets
        - Label dictionary
        - Hierarchical label dictionaries (if hierarchical=True)
        - Numeric feature normalizer
    """
    # Load dataset and label dictionary
    dataset, label_dict = load_dataset(config)

    # Prepare image transforms
    transforms_dict = prepare_transforms(config)

    # Prepare hierarchical labels if needed
    hierarchical_dicts = None
    hierarchical_labels = None
    if hierarchical:
        hierarchical_dicts = prepare_hierarchical_labels(dataset, label_dict)
        hierarchical_labels = extract_hierarchical_labels(dataset, label_dict, hierarchical_dicts)

    # Normalize numerical features
    num_normalizer = NumericFeatureNormalizer(method="z-score")
    num_features = [sample.numerical_attributes for sample in dataset]
    num_normalizer.fit(num_features)

    # Update numerical features in the dataset
    for i in range(len(dataset)):
        dataset[i].numerical_attributes = num_normalizer.transform([dataset[i].numerical_attributes])[0].tolist()

    # Split into train, validation, and test sets
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]

    # Stratified split based on labels
    labels = [sample.label for sample in dataset]

    # First split into train and temp (val+test)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=(val_ratio + test_ratio),
        random_state=config["seed"],
        stratify=labels
    )

    # Then split temp into val and test
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

    # Create datasets for each split
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Create MRISequenceDataset objects
    datasets = {
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

    print(f"Dataset splits: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    return datasets, label_dict, hierarchical_dicts, num_normalizer


def create_dataloaders(
        datasets: Dict[str, MRISequenceDataset],
        config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and test sets.

    Args:
        datasets: Dictionary mapping split names to datasets
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataloaders
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
    Calculate class weights for handling class imbalance.

    Args:
        dataset: Training dataset
        num_classes: Number of classes
        method: Method to calculate weights ('balanced', 'inverse', 'sqrt_inverse')

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        class_counts[label] += 1

    # Ensure no zeros to avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    if method == "balanced":
        # Weight inversely proportional to class frequency
        weights = len(dataset) / (num_classes * class_counts)
    elif method == "inverse":
        # Simple inverse weighting
        weights = 1.0 / class_counts
    elif method == "sqrt_inverse":
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / torch.sqrt(class_counts)
    else:
        raise ValueError(f"Unsupported class weighting method: {method}")

    # Normalize weights to sum to number of classes
    weights = weights * (num_classes / weights.sum())

    return weights