import os
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from utils.io import read_json_config, ensure_dir, load_pickle, save_json
from utils.metrics import compute_metrics, compute_hierarchical_metrics
from utils.visualize import plot_confusion_matrix, plot_per_class_metrics, plot_misclassified_examples
from data_loader import create_datasets, create_dataloaders
from models.classifier import get_classifier
from models.num_encoder import NumericFeatureNormalizer

import numpy as np
import torch.nn.functional as F


def evaluate_model(
        model_dir: str,
        dataset_dir: Optional[str] = None,
        output_dir: Optional[str] = None
):
    """
    Evaluate a trained model.

    Args:
        model_dir: Directory containing the model and configuration.
        dataset_dir: Optional directory containing the dataset (defaults to config value).
        output_dir: Optional directory to save evaluation results (defaults to model_dir/evaluation).
    """
    model_dir = Path(model_dir)

    # Load configuration
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    config = read_json_config(str(config_path))

    # Override dataset directory if provided
    if dataset_dir is not None:
        config["data"]["dataset_dir"] = dataset_dir

    # Set output directory
    if output_dir is None:
        output_dir = model_dir / "evaluation"
    else:
        output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    hierarchical = False  # Set to True for hierarchical classification
    datasets, label_dict, hierarchical_dicts, _ = create_datasets(config, hierarchical)

    # Create dataloaders
    dataloaders = create_dataloaders(datasets, config)

    # Load numeric feature normalizer
    normalizer_path = model_dir / "num_normalizer.pth"
    if normalizer_path.exists():
        normalizer = NumericFeatureNormalizer()
        normalizer.load(str(normalizer_path))
    else:
        print("Numeric feature normalizer not found. Using default normalization.")
        normalizer = None

    # Load model
    print("Loading model...")
    num_features = len(datasets["train"][0]["numerical_attributes"])
    num_classes = len(label_dict)

    # Initialize model
    model = get_classifier(
        config["model"],
        num_classes=num_classes,
        num_features=num_features,
        hierarchical=hierarchical,
        task_classes=hierarchical_dicts if hierarchical else None
    )

    # Load weights
    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare for evaluation
    all_preds = []
    all_targets = []
    all_probs = []
    misclassified_images = []
    misclassified_true_labels = []
    misclassified_pred_labels = []

    # For hierarchical evaluation
    task_preds = {} if hierarchical else None
    task_targets = {} if hierarchical else None

    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch in dataloaders["test"]:
            # Move batch to device
            img = batch["img"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_attributes = batch["numerical_attributes"].to(device)

            if hierarchical:
                # For hierarchical model, labels are stored in a dictionary.
                targets = {task: batch[task].to(device) for task in batch if task not in ["img", "input_ids", "attention_mask", "numerical_attributes"]}
            else:
                # For standard model, label is a single tensor.
                targets = batch["label"].to(device)

            # Forward pass
            outputs = model(
                img=img,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_attributes=numerical_attributes
            )

            # Process predictions
            if hierarchical:
                for task in outputs:
                    if task not in task_preds:
                        task_preds[task] = []
                        task_targets[task] = []

                    # Get predicted classes for each task
                    _, predicted = torch.max(outputs[task], 1)
                    task_preds[task].append(predicted.detach().cpu())
                    task_targets[task].append(targets[task].detach().cpu())
            else:
                # Get class probabilities
                probs = F.softmax(outputs, dim=1)
                # Get predicted classes
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_probs.append(probs.detach().cpu())

                # Collect misclassified examples
                mask = (predicted != targets)
                if mask.any():
                    misclassified_idx = mask.nonzero().squeeze().cpu()
                    for idx in misclassified_idx:
                        misclassified_images.append(img[idx].cpu().numpy())
                        misclassified_true_labels.append(targets[idx].item())
                        misclassified_pred_labels.append(predicted[idx].item())

    # Process evaluation results
    if hierarchical:
        # Process hierarchical results
        for task in task_preds:
            # Concatenate predictions and targets
            task_pred_tensor = torch.cat(task_preds[task], 0)
            task_target_tensor = torch.cat(task_targets[task], 0)

            # Plot confusion matrix for each task
            task_cm_path = output_dir / f"confusion_matrix_{task}.png"
            task_class_names = list(hierarchical_dicts[task].keys())
            plot_confusion_matrix(
                task_pred_tensor.numpy(),
                task_target_tensor.numpy(),
                task_class_names,
                task_cm_path,
                normalize=True
            )

        # Compute hierarchical metrics
        num_classes_dict = {task: len(hierarchical_dicts[task]) for task in task_preds}

        # Convert tensors to dictionaries for metrics computation
        task_pred_dict = {task: torch.cat(task_preds[task], 0) for task in task_preds}
        task_target_dict = {task: torch.cat(task_targets[task], 0) for task in task_targets}

        metrics = compute_hierarchical_metrics(task_pred_dict, task_target_dict, num_classes_dict)
    else:
        # Concatenate predictions and targets for standard classification
        all_pred_tensor = torch.cat(all_preds, 0)
        all_target_tensor = torch.cat(all_targets, 0)
        all_prob_tensor = torch.cat(all_probs, 0)

        # Compute overall metrics
        metrics = compute_metrics(all_prob_tensor, all_target_tensor, num_classes)

        # Get class names
        class_names = list(label_dict.keys())

        # Plot overall confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            all_pred_tensor.numpy(),
            all_target_tensor.numpy(),
            class_names,
            cm_path,
            normalize=True
        )

        # Calculate per-class metrics
        per_class_metrics = {}
        for class_idx in range(num_classes):
            # Binary mask for this class
            class_mask = (all_target_tensor == class_idx)

            # Skip if no samples for this class
            if class_mask.sum() == 0:
                continue

            # Accuracy for this class
            class_correct = (all_pred_tensor[class_mask] == class_idx).sum().item()
            class_total = class_mask.sum().item()
            class_acc = class_correct / class_total

            # Compute precision, recall, and F1 score
            class_pred_mask = (all_pred_tensor == class_idx)
            true_positives = (class_mask & class_pred_mask).sum().item()
            false_positives = ((~class_mask) & class_pred_mask).sum().item()
            false_negatives = (class_mask & (~class_pred_mask)).sum().item()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics[class_idx] = {
                "accuracy": class_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": class_total
            }

        metrics["per_class"] = per_class_metrics

        # Optionally plot per-class metrics
        per_class_plot_path = output_dir / "per_class_metrics.png"
        plot_per_class_metrics(per_class_metrics, per_class_plot_path)

        # Plot misclassified examples if available
        if misclassified_images:
            misclassified_plot_path = output_dir / "misclassified_examples.png"
            plot_misclassified_examples(misclassified_images, misclassified_true_labels, misclassified_pred_labels, misclassified_plot_path)

    # Save evaluation metrics to a JSON file
    metrics_path = output_dir / "metrics.json"
    save_json(metrics, metrics_path)
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model and configuration")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    args = parser.parse_args()

    evaluate_model(args.model_dir, args.dataset_dir, args.output_dir)
