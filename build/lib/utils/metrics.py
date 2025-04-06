import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary with metrics
    """
    # Convert to numpy for sklearn metrics
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Convert logits to predicted class indices
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Handle multi-class case
        pred_classes = np.argmax(predictions, axis=1)
    else:
        # Handle binary case
        pred_classes = (predictions > 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(targets, pred_classes)

    # Handle the case where some classes might not be present in this batch
    if num_classes > 2:
        # Use 'macro' averaging for multi-class
        f1 = f1_score(targets, pred_classes, average='macro', zero_division=0)
        precision = precision_score(targets, pred_classes, average='macro', zero_division=0)
        recall = recall_score(targets, pred_classes, average='macro', zero_division=0)

        # Per-class F1 scores
        per_class_f1 = f1_score(targets, pred_classes, average=None, zero_division=0)
        per_class_f1_dict = {f"f1_class_{i}": f1 for i, f1 in enumerate(per_class_f1)}

        # Compute confusion matrix
        cm = confusion_matrix(targets, pred_classes, labels=range(num_classes))

        # Calculate class-normalized confusion matrix (row-normalized)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Compute class-balanced accuracy
        class_balanced_acc = np.mean(np.diag(cm_norm))

    else:
        # Binary classification
        f1 = f1_score(targets, pred_classes, zero_division=0)
        precision = precision_score(targets, pred_classes, zero_division=0)
        recall = recall_score(targets, pred_classes, zero_division=0)
        per_class_f1_dict = {}
        class_balanced_acc = accuracy

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "class_balanced_acc": class_balanced_acc,
        **per_class_f1_dict
    }

    return metrics


def compute_hierarchical_metrics(
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_classes: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute metrics for hierarchical classification.

    Args:
        predictions: Dictionary mapping task names to predictions
        targets: Dictionary mapping task names to targets
        num_classes: Dictionary mapping task names to number of classes

    Returns:
        Dictionary with metrics for each task
    """
    all_metrics = {}

    for task in predictions:
        task_preds = predictions[task]
        task_targets = targets[task]
        task_num_classes = num_classes[task]

        # Compute metrics for this task
        task_metrics = compute_metrics(task_preds, task_targets, task_num_classes)

        # Add prefix to metrics keys
        task_metrics_with_prefix = {f"{task}_{k}": v for k, v in task_metrics.items()}

        # Update all metrics
        all_metrics.update(task_metrics_with_prefix)

    # Compute average metrics across tasks
    accuracies = [v for k, v in all_metrics.items() if k.endswith("_accuracy")]
    f1s = [v for k, v in all_metrics.items() if k.endswith("_f1")]

    all_metrics["avg_accuracy"] = np.mean(accuracies)
    all_metrics["avg_f1"] = np.mean(f1s)

    return all_metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.001,
            mode: str = "max"
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: One of {'min', 'max'}, whether to monitor for decrease or increase in validation metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == "min" else -np.Inf

    def __call__(self, val_score: float) -> Tuple[bool, bool]:
        """
        Check if training should stop.

        Args:
            val_score: Validation metric value

        Returns:
            Tuple containing (is_best_score, should_stop)
        """
        score = -val_score if self.mode == "min" else val_score

        if self.best_score is None:
            # First epoch
            self.best_score = score
            return True, False

        if score < self.best_score + self.min_delta:
            # Score did not improve significantly
            self.counter += 1
            if self.counter >= self.patience:
                # Patience exceeded, stop training
                self.early_stop = True
                return False, True
            return False, False
        else:
            # Score improved
            self.best_score = score
            self.counter = 0
            return True, False