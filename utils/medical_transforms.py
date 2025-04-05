import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torchvision.transforms as transforms

# Import MONAI transforms if available
try:
    import monai
    from monai.transforms import (
        Compose,
        RandFlip,
        RandRotate,
        RandSpatialCrop,
        Resize,
        RandGaussianNoise,
        GibbsNoise,
        RandBiasField,
        RandAdjustContrast,
        RandGaussianSmooth,
        RandHistogramShift
    )

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class MedicalImageTransform:
    """Transform for medical images with foreground-specific standardization."""

    def __init__(
            self,
            transform: Optional[Callable] = None,
            standardize_foreground: bool = True,
            foreground_threshold: int = 10,
            min_foreground_pixels: int = 50
    ):
        """
        Initialize the transform.

        Args:
            transform: Base transform to apply
            standardize_foreground: Whether to apply foreground-specific standardization
            foreground_threshold: Intensity threshold for identifying foreground
            min_foreground_pixels: Minimum number of pixels above threshold to consider as valid foreground
        """
        self.transform = transform
        self.standardize_foreground = standardize_foreground
        self.foreground_threshold = foreground_threshold
        self.min_foreground_pixels = min_foreground_pixels

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform.

        Args:
            img: Input image tensor

        Returns:
            Transformed image tensor
        """
        # Apply base transform if provided
        if self.transform is not None:
            img = self.transform(img)

        # Apply foreground-specific standardization
        if self.standardize_foreground:
            # Identify foreground based on threshold
            not_bg_mask = img > self.foreground_threshold

            # Check if enough pixels are considered foreground
            if not_bg_mask.sum() > self.min_foreground_pixels:
                # Standardize only foreground pixels
                mean = img[not_bg_mask].mean()
                std = img[not_bg_mask].std()
                img[not_bg_mask] = (img[not_bg_mask] - mean) / (std + 1e-6)
            else:
                # If not enough foreground pixels, standardize entire image
                mean = img.mean()
                std = img.std()
                img = (img - mean) / (std + 1e-6)

        return img


def get_monai_transforms(img_size: int = 224, augment: bool = True) -> Optional[Callable]:
    """
    Get MONAI transforms for medical images.

    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation

    Returns:
        MONAI transforms compose or None if MONAI is not available
    """
    if not MONAI_AVAILABLE:
        return None

    if augment:
        # Training transforms with augmentation
        return Compose([
            RandFlip(prob=0.4, spatial_axis=[0]),
            RandFlip(prob=0.4, spatial_axis=[1]),
            RandRotate(range_x=30, prob=0.2),  # Rotate within +/- 30 degrees
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),  # Gaussian noise
            RandSpatialCrop(
                roi_size=[img_size, img_size],
                random_size=False
            ),
            Resize(spatial_size=[img_size, img_size]),
            GibbsNoise(prob=0.2, alpha=0.1),
            RandBiasField(prob=0.2),  # MRI-specific bias field artifact
            RandAdjustContrast(prob=0.2, gamma=(0.8, 1.2)),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0)),
            RandHistogramShift(prob=0.2)
        ])
    else:
        # Validation/test transforms without augmentation
        return Compose([
            Resize(spatial_size=[img_size, img_size])
        ])


def prepare_medical_transforms(
        config: Dict[str, Any],
        use_monai: bool = True
) -> Dict[str, MedicalImageTransform]:
    """
    Prepare transforms for medical images.

    Args:
        config: Configuration dictionary
        use_monai: Whether to use MONAI transforms if available

    Returns:
        Dictionary with train and val transforms
    """
    img_size = config["data"]["img_size"]

    # Base torchvision transforms
    train_base_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    val_base_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Use MONAI transforms if available and requested
    if use_monai and MONAI_AVAILABLE:
        train_transforms = get_monai_transforms(img_size, augment=True)
        val_transforms = get_monai_transforms(img_size, augment=False)
    else:
        train_transforms = train_base_transforms
        val_transforms = val_base_transforms

    # Create medical image transforms
    return {
        "train": MedicalImageTransform(
            transform=train_transforms,
            standardize_foreground=True
        ),
        "val": MedicalImageTransform(
            transform=val_transforms,
            standardize_foreground=True
        ),
        "test": MedicalImageTransform(
            transform=val_transforms,
            standardize_foreground=True
        )
    }