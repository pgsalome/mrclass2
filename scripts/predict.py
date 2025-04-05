import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import torch.nn.functional as F
from PIL import Image

from utils.io import read_json_config, ensure_dir, load_pickle
from utils.dataclass import encodedSample
from models.classifier import get_classifier
from models.num_encoder import NumericFeatureNormalizer
from utils.intensity_normalization import get_intensity_normalizer
from data_loader import prepare_transforms


def load_model(model_dir: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a trained model and its configuration.

    Args:
        model_dir: Directory containing the model and configuration
        device: Device to load the model on

    Returns:
        Tuple containing the loaded model and its configuration
    """
    model_dir = Path(model_dir)

    # Load configuration
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    config = read_json_config(str(config_path))

    # Load label dictionary
    data_dir = Path(config["data"]["dataset_dir"])
    label_dict_path = data_dir / config["data"]["label_dict_name"]

    with open(label_dict_path, 'r') as f:
        label_dict = json.load(f)

    # Load numeric feature normalizer
    normalizer_path = model_dir / "num_normalizer.pth"
    if normalizer_path.exists():
        normalizer = NumericFeatureNormalizer()
        normalizer.load(str(normalizer_path))
    else:
        print("Warning: Numeric feature normalizer not found. Using default normalization.")
        normalizer = None

    # Initialize model
    num_classes = len(label_dict)
    num_features = 5  # Default number of numeric features

    # Initialize model
    model = get_classifier(
        config["model"],
        num_classes=num_classes,
        num_features=num_features,
        hierarchical=False
    )

    # Load weights
    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config, label_dict, normalizer


def preprocess_image(
        image: np.ndarray,
        config: Dict[str, Any],
        intensity_norm_class: Optional[str] = None
) -> torch.Tensor:
    """
    Preprocess an image for inference.

    Args:
        image: Input image as numpy array
        config: Model configuration
        intensity_norm_class: Optional class name for intensity normalization

    Returns:
        Preprocessed image tensor
    """
    # Initialize intensity normalizer if enabled
    intensity_normalizer = get_intensity_normalizer(config["data"])

    # Invert label dictionary for intensity normalization
    inv_label_dict = None

    # Prepare transforms
    transforms_dict = prepare_transforms(config, intensity_normalizer, inv_label_dict)
    transform = transforms_dict["test"]

    # Apply transform
    if intensity_norm_class:
        # If class is known, pass it for class-specific normalization
        preprocessed = transform(image, label=None)
    else:
        preprocessed = transform(image)

    # Add batch dimension
    preprocessed = preprocessed.unsqueeze(0)

    return preprocessed


def predict(
        model: torch.nn.Module,
        image: torch.Tensor,
        text_features: torch.Tensor,
        numerical_features: torch.Tensor,
        label_dict: Dict[str, int],
        device: torch.device
) -> Tuple[str, Dict[str, float]]:
    """
    Make a prediction with the model.

    Args:
        model: Trained model
        image: Preprocessed image tensor
        text_features: Text features tensor
        numerical_features: Numerical features tensor
        label_dict: Dictionary mapping class names to indices
        device: Device to run inference on

    Returns:
        Tuple containing the predicted class name and class probabilities
    """
    # Move inputs to device
    image = image.to(device)
    input_ids = text_features["input_ids"].to(device)
    attention_mask = text_features["attention_mask"].to(device)
    numerical_features = numerical_features.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(
            img=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_attributes=numerical_features
        )

        # Get probabilities
        probs = F.softmax(outputs, dim=1)

        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()

    # Convert index to class name
    inv_label_dict = {v: k for k, v in label_dict.items()}
    predicted_class = inv_label_dict[predicted_idx]

    # Create probabilities dictionary
    probs_dict = {inv_label_dict[i]: prob.item() for i, prob in enumerate(probs[0])}

    return predicted_class, probs_dict


def main():
    parser = argparse.ArgumentParser(description="Make predictions with a trained model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Text metadata (e.g., series description)")
    parser.add_argument("--numeric", type=str, required=True, help="Comma-separated numeric features")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save prediction outputs")
    args = parser.parse_args()

    # Create output directory
    ensure_dir(args.output_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config, label_dict, num_normalizer = load_model(args.model_dir, device)

    # Load and preprocess image
    print("Loading and preprocessing image...")
    image = np.array(Image.open(args.image_path).convert("L"))  # Convert to grayscale
    processed_image = preprocess_image(image, config)

    # Process text metadata
    print("Processing text metadata...")
    from transformers import AutoTokenizer

    # Get tokenizer based on config
    tokenizer_name = config["model"]["txt_encoder"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize text
    text_features = tokenizer(
        args.text,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        max_length=config["model"]["txt_encoder"]["max_length"],
        truncation=True,
        return_tensors="pt"
    )

    # Process numeric features
    print("Processing numeric features...")
    numeric_features = [float(x) for x in args.numeric.split(",")]

    # Normalize numeric features if normalizer is available
    if num_normalizer:
        numeric_features = num_normalizer.transform([numeric_features])[0].tolist()

    numeric_tensor = torch.tensor([numeric_features], dtype=torch.float)

    # Make prediction
    print("Making prediction...")
    predicted_class, class_probs = predict(
        model=model,
        image=processed_image,
        text_features=text_features,
        numerical_features=numeric_tensor,
        label_dict=label_dict,
        device=device
    )

    # Print prediction
    print(f"\nPredicted class: {predicted_class}")
    print("\nClass probabilities:")

    # Sort probabilities by value
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs[:10]:  # Show top 10
        print(f"  {cls}: {prob:.4f}")

    # Save prediction to file
    prediction_file = Path(args.output_dir) / "prediction.json"
    with open(prediction_file, "w") as f:
        json.dump({
            "predicted_class": predicted_class,
            "class_probabilities": class_probs,
            "image_path": args.image_path,
            "text_metadata": args.text,
            "numeric_features": args.numeric
        }, f, indent=2)

    print(f"\nPrediction saved to {prediction_file}")


if __name__ == "__main__":
    main()