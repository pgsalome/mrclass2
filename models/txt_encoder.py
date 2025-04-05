import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any, Optional, Tuple


class TextEncoder(nn.Module):
    """Text encoder for MRI sequence metadata."""

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            freeze_backbone: bool = False,
            max_length: int = 128,
            output_dim: int = 256
    ):
        """
        Initialize the text encoder.

        Args:
            model_name: Name of the pretrained transformer model
            freeze_backbone: Whether to freeze the backbone weights
            max_length: Maximum sequence length
            output_dim: Dimension of the output features
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        self.output_dim = output_dim

        # Initialize the transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)

        # Get the backbone output dimension
        if "bert" in model_name.lower() or "distilbert" in model_name.lower():
            backbone_output_dim = self.transformer.config.hidden_size
        elif "roberta" in model_name.lower():
            backbone_output_dim = self.transformer.config.hidden_size
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Add a projection layer to get the desired output dimension
        self.projection = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_output_dim, output_dim),
            nn.ReLU()
        )

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Forward pass through the transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the [CLS] token embedding or pooled output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            # BERT and similar models have pooler_output
            sequence_output = outputs.pooler_output
        else:
            # For models without pooler_output (like DistilBERT),
            # take the hidden state of the [CLS] token (first token)
            sequence_output = outputs.last_hidden_state[:, 0, :]

        # Project to the desired output dimension
        x = self.projection(sequence_output)

        return x


class SimpleTextEncoder(nn.Module):
    """Simple text encoder for MRI sequence metadata without using transformers."""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            output_dim: int = 256,
            num_layers: int = 1,
            bidirectional: bool = True
    ):
        """
        Initialize the simple text encoder.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the token embeddings
            hidden_dim: Dimension of the LSTM hidden state
            output_dim: Dimension of the output features
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.projection = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Compute sequence lengths from attention mask
        lengths = attention_mask.sum(dim=1).cpu()

        # Get token embeddings
        embedded = self.embedding(input_ids)

        # Pack the padded sequences for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward pass through LSTM
        output, (hidden, cell) = self.lstm(packed)

        # Concatenate the final hidden states from both directions if bidirectional
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Project to the desired output dimension
        x = self.projection(hidden)

        return x


def get_text_encoder(config: Dict[str, Any], vocab_size: Optional[int] = None) -> nn.Module:
    """
    Get a text encoder based on the config.

    Args:
        config: Dictionary with text encoder configuration
        vocab_size: Size of the vocabulary, required for SimpleTextEncoder

    Returns:
        Initialized text encoder
    """
    if "bert" in config["name"].lower() or "roberta" in config["name"].lower() or "distilbert" in config[
        "name"].lower():
        return TextEncoder(
            model_name=config["name"],
            freeze_backbone=config["freeze_backbone"],
            max_length=config["max_length"],
            output_dim=config["output_dim"]
        )
    elif config["name"] == "simple":
        if vocab_size is None:
            raise ValueError("vocab_size must be provided for SimpleTextEncoder")
        return SimpleTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            output_dim=config["output_dim"],
            num_layers=config.get("num_layers", 1),
            bidirectional=config.get("bidirectional", True)
        )
    else:
        raise ValueError(f"Unsupported model name: {config['name']}")