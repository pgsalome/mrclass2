import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


def plot_training_curves(
        train_metrics: List[Dict[str, float]],
        val_metrics: List[Dict[str, float]],
        save_dir: Union[str, Path],
        metric_names: List[str] = ["loss", "accuracy", "f1"]
):
    """
    Plot training and validation metrics over epochs.

    Args:
        train_metrics: List of training metrics for each epoch
        val_metrics: List of validation metrics for each epoch
        save_dir: Directory to save plots
        metric_names: List of metric names to plot
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_metrics) + 1)

    for metric in metric_names:
        if metric in train_metrics[0] and metric in val_metrics[0]:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [m[metric] for m in train_metrics], label=f'Training {metric}')
            plt.plot(epochs, [m[metric] for m in val_metrics], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric}_curve.png")
            plt.close()


def plot_confusion_matrix(
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: List[str],
        save_path: Union[str, Path],
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot and save a confusion matrix.

    Args:
        predictions: Predicted class indices
        targets: True class indices
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size (width, height)
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # Plot
    plt.figure(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f' if normalize else 'd')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_attention(
        img: torch.Tensor,
        attention_weights: torch.Tensor,
        save_path: Union[str, Path],
        text: Optional[List[str]] = None
):
    """
    Visualize attention weights between image and text.

    Args:
        img: Image tensor (1, C, H, W)
        attention_weights: Attention weights
        save_path: Path to save the visualization
        text: Optional list of text tokens
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert image tensor to numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # Convert attention weights to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # If image is 3D (C, H, W), convert to 2D by taking the first channel
    if len(img.shape) == 3:
        img = img[0]

    # If image is 4D (N, C, H, W), take the first sample and first channel
    if len(img.shape) == 4:
        img = img[0, 0]

    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Show the image
    ax.imshow(img, cmap='gray')
    ax.set_title('Attention Visualization')
    ax.axis('off')

    # Plot attention heatmap over the image
    attention_resized = np.resize(attention_weights, img.shape)
    ax.imshow(attention_resized, alpha=0.5, cmap='jet')

    # Add text tokens if provided
    if text is not None:
        plt.figtext(0.05, 0.01, f"Text: {' '.join(text)}", wrap=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_misclassified_examples(
        images: List[np.ndarray],
        true_labels: List[int],
        pred_labels: List[int],
        class_names: List[str],
        save_dir: Union[str, Path],
        max_examples: int = 20,
        figsize: Tuple[int, int] = (15, 15)
):
    """
    Plot misclassified examples.

    Args:
        images: List of misclassified images
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: List of class names
        save_dir: Directory to save the plots
        max_examples: Maximum number of examples to plot
        figsize: Figure size (width, height)
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Limit the number of examples
    n_examples = min(max_examples, len(images))

    # Adjust grid size based on the number of examples
    grid_size = int(np.ceil(np.sqrt(n_examples)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)

    # Convert axs to 1D array for easier indexing
    axs = axs.flatten()

    for i in range(n_examples):
        img = images[i]

        # Handle different image formats
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            # Convert (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # If grayscale (H, W, 1) or (1, H, W), squeeze to (H, W)
        if len(img.shape) == 3 and (img.shape[0] == 1 or img.shape[2] == 1):
            img = np.squeeze(img)

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)

        # Display image
        axs[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axs[i].set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}")
        axs[i].axis('off')

    # Hide unused subplots
    for i in range(n_examples, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "misclassified_examples.png", bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(
        per_class_metrics: Dict[str, float],
        class_names: List[str],
        save_path: Union[str, Path],
        metric_name: str = "F1 Score",
        figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot per-class metrics.

    Args:
        per_class_metrics: Dictionary mapping class indices to metric values
        class_names: List of class names
        save_path: Path to save the plot
        metric_name: Name of the metric
        figsize: Figure size (width, height)
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)

    # Convert dictionary to lists for plotting
    classes = []
    values = []
    for i, name in enumerate(class_names):
        if i in per_class_metrics:
            classes.append(name)
            values.append(per_class_metrics[i])

    # Sort by metric value
    indices = np.argsort(values)
    classes = [classes[i] for i in indices]
    values = [values[i] for i in indices]

    # Plot
    plt.barh(classes, values, color='skyblue')
    plt.xlabel(metric_name)
    plt.ylabel('Class')
    plt.title(f'Per-Class {metric_name}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_hierarchical_results(
        metrics: Dict[str, float],
        task_names: List[str],
        save_dir: Union[str, Path]
):
    """
    Plot metrics for hierarchical classification tasks.

    Args:
        metrics: Dictionary with metrics for each task
        task_names: List of task names
        save_dir: Directory to save plots
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics for each task
    task_metrics = {}
    metrics_to_plot = ["accuracy", "f1", "precision", "recall"]

    for task in task_names:
        task_metrics[task] = {
            metric: metrics.get(f"{task}_{metric}", 0) for metric in metrics_to_plot
        }

    # Plot comparison of tasks for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        values = [task_metrics[task][metric] for task in task_names]
        plt.bar(task_names, values, alpha=0.7)
        plt.title(f'{metric.capitalize()} by Task')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir / f"hierarchical_{metric}.png")
        plt.close()

    # Also create a combined plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(task_names))
    width = 0.2

    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        values = [task_metrics[task][metric] for task in task_names]
        plt.bar(x + i * width, values, width, label=metric.capitalize(), alpha=0.7)

    plt.xlabel('Task')
    plt.ylabel('Score')
    plt.title('Hierarchical Classification Performance')
    plt.xticks(x + width * 1.5, task_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / "hierarchical_combined.png")
    plt.close()