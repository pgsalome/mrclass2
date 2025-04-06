import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

# Try to import MONAI for NetAdapter
try:
    from monai.networks.nets import NetAdapter
    from torchvision.models import resnet50, ResNet50_Weights

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    warnings.warn("MONAI not found. radImageNet models will not be available. "
                  "Install with: pip install monai")


class RadImageNet(nn.Module):
    """
    RadImageNet pre-trained model using MONAI's NetAdapter.
    Uses ResNet50 pre-trained on the RadImageNet dataset.
    """

    def __init__(self, output_dim: int = 512, weights_path: Optional[str] = None):
        """
        Initialize the RadImageNet model.

        Args:
            output_dim: Dimension of the output features
            weights_path: Path to pre-trained weights
        """
        super().__init__()

        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for RadImageNet models. "
                              "Install with: pip install monai")

        # Initialize with ImageNet weights first
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Adapt the model to output the specified dimension
        self.model = NetAdapter(resnet, num_classes=output_dim)

        # Load RadImageNet weights if provided
        if weights_path is not None:
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str):
        """
        Load pre-trained weights.

        Args:
            weights_path: Path to pre-trained weights
        """
        try:
            state_dict = torch.load(weights_path, map_location='cpu')

            # Check if state dict is wrapped (from DataParallel)
            if list(state_dict.keys())[0].startswith('module.'):
                # Remove 'module.' prefix
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # Load weights into the model
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded RadImageNet weights from {weights_path}")
        except Exception as e:
            warnings.warn(f"Failed to load RadImageNet weights: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.model(x)


def get_rad_imagenet(output_dim: int = 512, weights_path: Optional[str] = None) -> nn.Module:
    """
    Get a RadImageNet model.

    Args:
        output_dim: Dimension of the output features
        weights_path: Path to pre-trained weights

    Returns:
        RadImageNet model
    """
    if not MONAI_AVAILABLE:
        warnings.warn("MONAI is required for RadImageNet models. "
                      "Using standard ResNet50 instead.")
        # Fallback to standard PyTorch model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the final layer
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    return RadImageNet(output_dim, weights_path)