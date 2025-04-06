import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
import torch
import warnings
import logging

# Optional dependencies for intensity normalization
try:
    import intensity_normalization
    from intensity_normalization.normalize import fcm, gmm, hm, kde, whitestripe, zscore

    INTENSITY_NORM_AVAILABLE = True
except ImportError:
    INTENSITY_NORM_AVAILABLE = False
    warnings.warn("intensity-normalization package not found. Install with: pip install intensity-normalization")


class IntensityNormalizer:
    """Utility class for normalizing MRI images using the intensity-normalization package."""

    VALID_METHODS = [
        "fcm",  # Fuzzy C-Means
        "gmm",  # Gaussian Mixture Model
        "hm",  # Histogram Matching
        "kde",  # Kernel Density Estimation
        "whitestripe",  # WhiteStripe
        "zscore"  # Z-score
    ]

    def __init__(
            self,
            method: str = "zscore",
            class_specific: bool = False,
            class_method_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the intensity normalizer.

        Args:
            method: Normalization method
            class_specific: Whether to use class-specific normalization methods
            class_method_map: Dictionary mapping class names to normalization methods
        """
        if not INTENSITY_NORM_AVAILABLE:
            raise ImportError("intensity-normalization package is required for this feature")

        self.method = method.lower() if method else None
        self.class_specific = class_specific
        self.class_method_map = class_method_map or {}

        # Validate methods
        if self.method and self.method not in self.VALID_METHODS:
            raise ValueError(f"Invalid normalization method: {self.method}. "
                             f"Valid methods are: {', '.join(self.VALID_METHODS)}")

        for cls, cls_method in self.class_method_map.items():
            if cls_method and cls_method not in self.VALID_METHODS:
                raise ValueError(f"Invalid normalization method for class {cls}: {cls_method}. "
                                 f"Valid methods are: {', '.join(self.VALID_METHODS)}")

        # Create normalizer functions map
        self.normalizers = {
            "fcm": self._normalize_fcm,
            "gmm": self._normalize_gmm,
            "hm": self._normalize_hm,
            "kde": self._normalize_kde,
            "whitestripe": self._normalize_whitestripe,
            "zscore": self._normalize_zscore
        }

    def normalize(self, img: np.ndarray, class_name: Optional[str] = None) -> np.ndarray:
        """
        Normalize an image using the specified method.

        Args:
            img: Image to normalize
            class_name: Optional class name (for class-specific normalization)

        Returns:
            Normalized image
        """
        # Skip normalization if no method is specified
        if not self.method and not (self.class_specific and class_name in self.class_method_map):
            return img

        # Determine which method to use
        if self.class_specific and class_name in self.class_method_map:
            method = self.class_method_map.get(class_name)
            if not method:  # Skip if method is None for this class
                return img
        else:
            method = self.method

        # Get the appropriate normalizer function
        normalizer = self.normalizers.get(method)
        if not normalizer:
            warnings.warn(f"Normalization method '{method}' not implemented. Skipping.")
            return img

        # Ensure image is suitable for normalization
        # Most MRI normalization methods expect 3D volumes, but we have 2D slices
        # Handle with a dummy 3D volume (make the slice into a 1-slice volume)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
            was_2d = True
        else:
            # If already 3D, keep as is
            was_2d = False

        # Apply normalization
        try:
            normalized_img = normalizer(img)

            # Convert back to 2D if input was 2D
            if was_2d:
                normalized_img = normalized_img[0]

            return normalized_img
        except Exception as e:
            logging.warning(f"Error during {method} normalization: {str(e)}. Returning original image.")
            if was_2d:
                return img[0]
            return img

    def _normalize_fcm(self, img: np.ndarray) -> np.ndarray:
        """Normalize using Fuzzy C-Means."""
        return fcm.fcm_normalize(img)

    def _normalize_gmm(self, img: np.ndarray) -> np.ndarray:
        """Normalize using Gaussian Mixture Model."""
        return gmm.gmm_normalize(img)

    def _normalize_hm(self, img: np.ndarray) -> np.ndarray:
        """Normalize using Histogram Matching."""
        # Histogram matching requires a template/reference image which we don't have
        # Use standard histogram matching to a reference histogram
        return hm.hm_normalize(img)

    def _normalize_kde(self, img: np.ndarray) -> np.ndarray:
        """Normalize using Kernel Density Estimation."""
        return kde.kde_normalize(img)

    def _normalize_whitestripe(self, img: np.ndarray) -> np.ndarray:
        """Normalize using WhiteStripe."""
        try:
            # WhiteStripe needs to estimate tissue masks
            return whitestripe.whitestripe_normalize(img)
        except:
            # If WhiteStripe fails (common for non-brain MRIs), fall back to zscore
            warnings.warn("WhiteStripe normalization failed. Falling back to zscore.")
            return self._normalize_zscore(img)

    def _normalize_zscore(self, img: np.ndarray) -> np.ndarray:
        """Normalize using Z-score."""
        return zscore.zscore_normalize(img)


def get_intensity_normalizer(config: Dict[str, Any]) -> Optional[IntensityNormalizer]:
    """
    Create an intensity normalizer based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        IntensityNormalizer instance or None if normalization is disabled
    """
    if not INTENSITY_NORM_AVAILABLE:
        warnings.warn("intensity-normalization package not found. Skipping intensity normalization.")
        return None

    norm_config = config.get("intensity_normalization", {})

    # Skip if normalization is disabled
    if not norm_config.get("enabled", False):
        return None

    # Get method
    method = norm_config.get("method")

    # Check for class-specific normalization
    class_specific = norm_config.get("class_specific", {}).get("enabled", False)
    class_method_map = {}

    if class_specific:
        for class_name, class_method in norm_config.get("class_specific", {}).items():
            if class_name != "enabled" and class_method:
                class_method_map[class_name] = class_method

    # Create normalizer
    try:
        return IntensityNormalizer(
            method=method,
            class_specific=class_specific,
            class_method_map=class_method_map
        )
    except ImportError:
        warnings.warn("intensity-normalization package not available. Skipping intensity normalization.")
        return None
    except ValueError as e:
        warnings.warn(f"Error creating intensity normalizer: {str(e)}. Skipping intensity normalization.")
        return None


def apply_intensity_normalization(
        img: np.ndarray,
        normalizer: Optional[IntensityNormalizer],
        class_name: Optional[str] = None
) -> np.ndarray:
    """
    Apply intensity normalization to an image.

    Args:
        img: Image to normalize
        normalizer: Intensity normalizer instance
        class_name: Optional class name (for class-specific normalization)

    Returns:
        Normalized image or original image if normalizer is None
    """
    if normalizer is None:
        return img

    return normalizer.normalize(img, class_name)


class IntensityNormTransform:
    """Transform for applying intensity normalization during data loading."""

    def __init__(
            self,
            normalizer: Optional[IntensityNormalizer],
            label_dict: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the transform.

        Args:
            normalizer: Intensity normalizer instance
            label_dict: Dictionary mapping label indices to class names
        """
        self.normalizer = normalizer
        self.label_dict = label_dict

    def __call__(self, img: np.ndarray, label: Optional[int] = None) -> np.ndarray:
        """
        Apply normalization to an image.

        Args:
            img: Image to normalize
            label: Optional label index

        Returns:
            Normalized image
        """
        if self.normalizer is None:
            return img

        # Get class name if label_dict is available
        class_name = None
        if label is not None and self.label_dict is not None:
            class_name = self.label_dict.get(label)
            # Extract base class (e.g., 'T1' from 'T1-SE')
            if class_name and '-' in class_name:
                class_name = class_name.split('-')[0]

        return self.normalizer.normalize(img, class_name)