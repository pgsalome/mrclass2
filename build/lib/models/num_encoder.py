import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class NumericEncoder(nn.Module):
    """Encoder for numerical DICOM attributes."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 32,
            num_layers: int = 2,
            dropout: float = 0.2
    ):
        """
        Initialize the numeric encoder.

        Args:
            input_dim: Dimension of the input features
            hidden_dim: Dimension of the hidden layers
            output_dim: Dimension of the output features
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Create MLP layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.mlp(x)


def get_numeric_encoder(config: Dict[str, Any], input_dim: int) -> NumericEncoder:
    """
    Get a numeric encoder based on the config.

    Args:
        config: Dictionary with numeric encoder configuration
        input_dim: Dimension of the input features

    Returns:
        Initialized NumericEncoder
    """
    return NumericEncoder(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )


class NumericFeatureNormalizer:
    """Utility class for normalizing numerical features."""

    def __init__(self, method: str = "z-score"):
        """
        Initialize the normalizer.

        Args:
            method: Normalization method, one of "z-score", "min-max"
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False

    def fit(self, data: List[List[float]]) -> None:
        """
        Fit the normalizer to the data.

        Args:
            data: List of numerical feature vectors
        """
        data = torch.tensor(data, dtype=torch.float)

        if self.method == "z-score":
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            # Handle zero std (constant features)
            self.std[self.std == 0] = 1.0
        elif self.method == "min-max":
            self.min = data.min(dim=0)[0]
            self.max = data.max(dim=0)[0]
            # Handle min == max (constant features)
            is_constant = (self.max == self.min)
            self.max[is_constant] = self.min[is_constant] + 1.0
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        self.fitted = True

    def transform(self, data: List[List[float]]) -> torch.Tensor:
        """
        Transform the data.

        Args:
            data: List of numerical feature vectors

        Returns:
            Normalized features as a tensor
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")

        data = torch.tensor(data, dtype=torch.float)

        if self.method == "z-score":
            return (data - self.mean) / self.std
        elif self.method == "min-max":
            return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform the normalized data.

        Args:
            data: Normalized features as a tensor

        Returns:
            Original scale features as a tensor
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")

        if self.method == "z-score":
            return data * self.std + self.mean
        elif self.method == "min-max":
            return data * (self.max - self.min) + self.min

    def save(self, path: str) -> None:
        """
        Save the normalizer.

        Args:
            path: Path to save the normalizer
        """
        state = {
            "method": self.method,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "fitted": self.fitted
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """
        Load the normalizer.

        Args:
            path: Path to the saved normalizer
        """
        state = torch.load(path)
        self.method = state["method"]
        self.mean = state["mean"]
        self.std = state["std"]
        self.min = state["min"]
        self.max = state["max"]
        self.fitted = state["fitted"]