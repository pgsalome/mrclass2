import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class OVRLoss(nn.Module):
    """
    One-vs-Rest Loss using binary cross-entropy for each class.
    Automatically handles class imbalance by weighting based on class frequencies.
    """

    def __init__(self, num_classes: int, class_counts: List[int]):
        """
        Initialize the One-vs-Rest Loss.

        Args:
            num_classes: Number of classes
            class_counts: List of sample counts for each class
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_counts = class_counts
        self.loss_funcs = []  # A list of num_classes loss functions for binary classification

        for i in range(num_classes):
            # Weight is ratio of negative examples to positive examples
            pos_weight = torch.tensor((sum(class_counts) - class_counts[i]) / (class_counts[i])).float()
            self.loss_funcs.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum'))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits: Model predictions
            labels: Ground truth labels

        Returns:
            Weighted sum of binary losses
        """
        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        # Compute loss for each class
        losses = torch.empty(self.num_classes, dtype=float, device=logits.device)
        for i in range(self.num_classes):
            class_logits = logits[:, i].unsqueeze(1)
            class_labels = labels_one_hot[:, i].unsqueeze(1)

            loss_func = self.loss_funcs[i]
            losses[i] = loss_func(class_logits, class_labels)

        # Return sum of losses
        return losses.sum()


def get_loss_function(
        loss_type: str,
        num_classes: int,
        class_counts: Optional[List[int]] = None,
        class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Get a loss function based on configuration.

    Args:
        loss_type: Type of loss function ('ce', 'weighted_ce', 'ovr')
        num_classes: Number of classes
        class_counts: List of sample counts for each class (required for 'ovr' loss)
        class_weights: Tensor of class weights (optional for 'weighted_ce' loss)

    Returns:
        Loss function
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted_ce":
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce loss")
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "ovr":
        if class_counts is None:
            raise ValueError("class_counts must be provided for ovr loss")
        return OVRLoss(num_classes, class_counts)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")