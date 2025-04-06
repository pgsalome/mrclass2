import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional

from models.img_encoder import get_image_encoder
from models.txt_encoder import get_text_encoder
from models.num_encoder import get_numeric_encoder
from models.fusion import get_fusion_module


class MRISequenceClassifier(nn.Module):
    """Multimodal MRI sequence classifier."""

    def __init__(
            self,
            num_classes: int,
            img_encoder_config: Dict[str, Any],
            txt_encoder_config: Dict[str, Any],
            num_encoder_config: Dict[str, Any],
            fusion_config: Dict[str, Any],
            classifier_config: Dict[str, Any],
            num_features: int = 5,
            vocab_size: Optional[int] = None
    ):
        """
        Initialize the multimodal classifier.

        Args:
            num_classes: Number of classes
            img_encoder_config: Image encoder configuration
            txt_encoder_config: Text encoder configuration
            num_encoder_config: Numeric encoder configuration
            fusion_config: Fusion module configuration
            classifier_config: Classifier head configuration
            num_features: Number of numeric features
            vocab_size: Size of the vocabulary (for simple text encoder)
        """
        super().__init__()
        self.num_classes = num_classes

        # Initialize encoders
        self.img_encoder = get_image_encoder(img_encoder_config)
        self.txt_encoder = get_text_encoder(txt_encoder_config, vocab_size)
        self.num_encoder = get_numeric_encoder(num_encoder_config, num_features)

        # Initialize fusion module
        self.fusion = get_fusion_module(
            fusion_config,
            [
                img_encoder_config["output_dim"],
                txt_encoder_config["output_dim"],
                num_encoder_config["output_dim"]
            ]
        )

        # Initialize classifier head
        self.classifier = self._build_classifier(
            fusion_config["hidden_size"],
            classifier_config["hidden_dims"],
            num_classes,
            classifier_config["dropout"]
        )

    def _build_classifier(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout: float
    ) -> nn.Sequential:
        """
        Build the classifier head.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability

        Returns:
            Classifier head as a sequential module
        """
        layers = []
        current_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(
            self,
            img: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numerical_attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img: Image tensor
            input_ids: Text input IDs
            attention_mask: Text attention mask
            numerical_attributes: Numerical attributes

        Returns:
            Classification logits
        """
        # Encode each modality
        img_features = self.img_encoder(img)
        txt_features = self.txt_encoder(input_ids, attention_mask)
        num_features = self.num_encoder(numerical_attributes)

        # Fusion
        fused_features = self.fusion([img_features, txt_features, num_features])

        # Classification
        logits = self.classifier(fused_features)

        return logits


class HierarchicalMRISequenceClassifier(nn.Module):
    """Hierarchical multimodal MRI sequence classifier with multiple output heads."""

    def __init__(
            self,
            num_classes: Dict[str, int],  # Dict mapping task names to num classes
            img_encoder_config: Dict[str, Any],
            txt_encoder_config: Dict[str, Any],
            num_encoder_config: Dict[str, Any],
            fusion_config: Dict[str, Any],
            classifier_config: Dict[str, Any],
            num_features: int = 5,
            vocab_size: Optional[int] = None
    ):
        """
        Initialize the hierarchical multimodal classifier.

        Args:
            num_classes: Dictionary mapping task names to number of classes
            img_encoder_config: Image encoder configuration
            txt_encoder_config: Text encoder configuration
            num_encoder_config: Numeric encoder configuration
            fusion_config: Fusion module configuration
            classifier_config: Classifier head configuration
            num_features: Number of numeric features
            vocab_size: Size of the vocabulary (for simple text encoder)
        """
        super().__init__()
        self.num_classes = num_classes

        # Initialize encoders
        self.img_encoder = get_image_encoder(img_encoder_config)
        self.txt_encoder = get_text_encoder(txt_encoder_config, vocab_size)
        self.num_encoder = get_numeric_encoder(num_encoder_config, num_features)

        # Initialize fusion module
        self.fusion = get_fusion_module(
            fusion_config,
            [
                img_encoder_config["output_dim"],
                txt_encoder_config["output_dim"],
                num_encoder_config["output_dim"]
            ]
        )

        # Initialize classifier heads (one for each task)
        self.classifiers = nn.ModuleDict({
            task: self._build_classifier(
                fusion_config["hidden_size"],
                classifier_config["hidden_dims"],
                n_classes,
                classifier_config["dropout"]
            ) for task, n_classes in num_classes.items()
        })

    def _build_classifier(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout: float
    ) -> nn.Sequential:
        """
        Build a classifier head.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability

        Returns:
            Classifier head as a sequential module
        """
        layers = []
        current_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(
            self,
            img: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numerical_attributes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            img: Image tensor
            input_ids: Text input IDs
            attention_mask: Text attention mask
            numerical_attributes: Numerical attributes

        Returns:
            Dictionary mapping task names to classification logits
        """
        # Encode each modality
        img_features = self.img_encoder(img)
        txt_features = self.txt_encoder(input_ids, attention_mask)
        num_features = self.num_encoder(numerical_attributes)

        # Fusion
        fused_features = self.fusion([img_features, txt_features, num_features])

        # Classification for each task
        logits = {
            task: classifier(fused_features) for task, classifier in self.classifiers.items()
        }

        return logits


def get_classifier(
        config: Dict[str, Any],
        num_classes: int,
        num_features: int,
        vocab_size: Optional[int] = None,
        hierarchical: bool = False,
        task_classes: Optional[Dict[str, int]] = None
) -> nn.Module:
    """
    Get a classifier based on the config.

    Args:
        config: Model configuration dictionary
        num_classes: Number of classes
        num_features: Number of numeric features
        vocab_size: Size of the vocabulary (for simple text encoder)
        hierarchical: Whether to use hierarchical classification
        task_classes: Dictionary mapping task names to number of classes (for hierarchical)

    Returns:
        Initialized classifier model
    """
    if hierarchical:
        if task_classes is None:
            raise ValueError("task_classes must be provided for hierarchical classifier")
        return HierarchicalMRISequenceClassifier(
            num_classes=task_classes,
            img_encoder_config=config["img_encoder"],
            txt_encoder_config=config["txt_encoder"],
            num_encoder_config=config["num_encoder"],
            fusion_config=config["fusion"],
            classifier_config=config["classifier"],
            num_features=num_features,
            vocab_size=vocab_size
        )
    else:
        return MRISequenceClassifier(
            num_classes=num_classes,
            img_encoder_config=config["img_encoder"],
            txt_encoder_config=config["txt_encoder"],
            num_encoder_config=config["num_encoder"],
            fusion_config=config["fusion"],
            classifier_config=config["classifier"],
            num_features=num_features,
            vocab_size=vocab_size
        )