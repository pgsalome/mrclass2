import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


class ConcatFusion(nn.Module):
    """Concatenation-based fusion of multimodal features."""

    def __init__(
            self,
            input_dims: List[int],
            hidden_dim: int = 512
    ):
        """
        Initialize the concatenation fusion module.

        Args:
            input_dims: List of input dimensions for each modality
            hidden_dim: Dimension of the hidden layer
        """
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.total_input_dim = sum(input_dims)

        # Linear projection after concatenation
        self.projection = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from each modality

        Returns:
            Fused features tensor
        """
        # Ensure we have the correct number of input features
        assert len(features) == len(self.input_dims), \
            f"Expected {len(self.input_dims)} feature tensors, got {len(features)}"

        # Concatenate along the feature dimension
        concat_features = torch.cat(features, dim=1)

        # Project to hidden dimension
        fused_features = self.projection(concat_features)

        return fused_features


class AttentionFusion(nn.Module):
    """Attention-based fusion of multimodal features."""

    def __init__(
            self,
            input_dims: List[int],
            hidden_dim: int = 512
    ):
        """
        Initialize the attention fusion module.

        Args:
            input_dims: List of input dimensions for each modality
            hidden_dim: Dimension of the hidden layer
        """
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        # Project each modality to a common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from each modality

        Returns:
            Fused features tensor
        """
        # Ensure we have the correct number of input features
        assert len(features) == len(self.input_dims), \
            f"Expected {len(self.input_dims)} feature tensors, got {len(features)}"

        # Project each modality to a common dimension
        projected_features = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]

        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)  # (batch_size, num_modalities, hidden_dim)

        # Calculate attention weights
        attention_logits = self.attention(stacked_features).squeeze(-1)  # (batch_size, num_modalities)
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)  # (batch_size, num_modalities, 1)

        # Apply attention weights
        attended_features = stacked_features * attention_weights
        fused_features = torch.sum(attended_features, dim=1)  # (batch_size, hidden_dim)

        return fused_features


class GatedFusion(nn.Module):
    """Gated fusion of multimodal features."""

    def __init__(
            self,
            input_dims: List[int],
            hidden_dim: int = 512
    ):
        """
        Initialize the gated fusion module.

        Args:
            input_dims: List of input dimensions for each modality
            hidden_dim: Dimension of the hidden layer
        """
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        # Project each modality to a common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])

        # Gate networks for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sum(input_dims), 1),
                nn.Sigmoid()
            ) for _ in input_dims
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from each modality

        Returns:
            Fused features tensor
        """
        # Ensure we have the correct number of input features
        assert len(features) == len(self.input_dims), \
            f"Expected {len(self.input_dims)} feature tensors, got {len(features)}"

        # Concatenate all features for gate input
        concat_features = torch.cat(features, dim=1)

        # Project each modality to common dimension
        projected_features = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]

        # Calculate gates
        gates = [gate(concat_features) for gate in self.gates]

        # Apply gates and sum
        gated_features = [
            gate * feat for gate, feat in zip(gates, projected_features)
        ]

        fused_features = sum(gated_features)

        return fused_features


def get_fusion_module(
        config: Dict[str, Any],
        input_dims: List[int]
) -> nn.Module:
    """
    Get a fusion module based on the config.

    Args:
        config: Dictionary with fusion configuration
        input_dims: List of input dimensions for each modality

    Returns:
        Initialized fusion module
    """
    method = config["method"]
    hidden_dim = config["hidden_size"]

    if method == "concat":
        return ConcatFusion(input_dims, hidden_dim)
    elif method == "attention":
        return AttentionFusion(input_dims, hidden_dim)
    elif method == "gated":
        return GatedFusion(input_dims, hidden_dim)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")