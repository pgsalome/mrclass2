import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional

from utils.rad_imagenet import get_rad_imagenet, MONAI_AVAILABLE


class ImageEncoder(nn.Module):
    """Image encoder for MRI sequence classification."""

    def __init__(
            self,
            model_name: str = "resnet50",
            pretrained: bool = True,
            freeze_backbone: bool = False,
            weights_path: Optional[str] = None,
            output_dim: int = 512
    ):
        """
        Initialize the image encoder.

        Args:
            model_name: Name of the backbone CNN model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone weights
            weights_path: Optional path to custom weights
            output_dim: Dimension of the output features
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.output_dim = output_dim

        # Initialize the backbone CNN
        self.backbone = self._get_backbone(weights_path)

        # Get the backbone output dimension
        backbone_output_dim = self._get_backbone_output_dim()

        # Add a projection layer to get the desired output dimension
        self.projection = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_output_dim, output_dim),
            nn.ReLU()
        )

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_backbone(self, weights_path: Optional[str] = None) -> nn.Module:
        """
        Get the backbone CNN model.

        Args:
            weights_path: Optional path to custom weights

        Returns:
            Backbone CNN model
        """
        # Check for special model: radImageNet
        if self.model_name == "radimagenet":
            if not MONAI_AVAILABLE:
                raise ImportError("MONAI is required for RadImageNet models. Install with: pip install monai")
            model = get_rad_imagenet(output_dim=self.output_dim, weights_path=weights_path)
            return model

        # Standard torchvision models
        if self.model_name == "resnet18":
            model = models.resnet18(pretrained=self.pretrained)
            # Remove the classification head
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "resnet34":
            model = models.resnet34(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "densenet121":
            model = models.densenet121(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Load custom weights if provided
        if weights_path is not None and self.model_name != "radimagenet":
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"Loaded custom weights for {self.model_name} from {weights_path}")
                # Re-create the backbone with loaded weights
                backbone = nn.Sequential(*list(model.children())[:-1])
            except Exception as e:
                print(f"Warning: Failed to load custom weights: {e}")

        return backbone

    def _get_backbone_output_dim(self) -> int:
        """
        Get the output dimension of the backbone.

        Returns:
            Output dimension of the backbone
        """
        if self.model_name == "radimagenet":
            return self.output_dim  # RadImageNet already outputs the desired dimension
        elif self.model_name == "resnet18":
            return 512
        elif self.model_name == "resnet34":
            return 512
        elif self.model_name == "resnet50":
            return 2048
        elif self.model_name == "densenet121":
            return 1024
        elif self.model_name == "efficientnet_b0":
            return 1280
        elif self.model_name == "mobilenet_v2":
            return 1280
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Convert to 3 channels if input is grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Special case for RadImageNet
        if self.model_name == "radimagenet":
            return self.backbone(x)

        # Forward pass through the backbone
        x = self.backbone(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Project to the desired output dimension
        x = self.projection(x)

        return x


def get_image_encoder(config: Dict[str, Any]) -> ImageEncoder:
    """
    Get an image encoder based on the config.

    Args:
        config: Dictionary with image encoder configuration

    Returns:
        Initialized ImageEncoder
    """
    return ImageEncoder(
        model_name=config["name"],
        pretrained=config["pretrained"],
        freeze_backbone=config["freeze_backbone"],
        weights_path=config["weights_path"],
        output_dim=config["output_dim"]
    )