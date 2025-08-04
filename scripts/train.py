import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# Import wandb for experiment tracking
import wandb

from utils.io import read_json_config, ensure_dir, save_model, save_json
from utils.metrics import compute_metrics, compute_hierarchical_metrics, EarlyStopping
from utils.visualize import plot_training_curves, plot_confusion_matrix, plot_per_class_metrics
from utils.loss import get_loss_function
from utils.data_loader import create_datasets, create_dataloaders, get_class_weights
from models.classifier import get_classifier
import torch.amp


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        scaler: torch.amp.GradScaler,  # Updated type hint
        device: torch.device,
        config: Dict[str, Any],
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        hierarchical: bool = False,
        threshold: float = 0.0
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        scaler: Gradient scaler for mixed precision training
        device: Device to train on
        config: Training configuration
        epoch: Current epoch number
        writer: TensorBoard writer
        hierarchical: Whether using hierarchical classification
        threshold: Classification threshold (for OVR loss)

    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # For hierarchical training
    task_preds = {} if hierarchical else None
    task_targets = {} if hierarchical else None

    # Progress tracking
    log_interval = config["logging"]["log_interval"]
    total_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        img = batch["img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical_attributes = batch["numerical_attributes"].to(device)

        if hierarchical:
            # For hierarchical model, labels are a dictionary
            targets = {task: batch[task].to(device) for task in batch if task != "img" and task != "input_ids"
                       and task != "attention_mask" and task != "numerical_attributes"}
        else:
            # For standard model, label is a single tensor
            targets = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=config["training"]["mixed_precision"]):
            # Forward pass
            outputs = model(
                img=img,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_attributes=numerical_attributes
            )

            # Compute loss
            if hierarchical:
                # Sum losses from all tasks
                loss = sum([loss_fn(outputs[task], targets[task]) for task in outputs])

                # Store predictions and targets for metrics
                for task in outputs:
                    if task not in task_preds:
                        task_preds[task] = []
                        task_targets[task] = []
                    task_preds[task].append(outputs[task].detach().cpu())
                    task_targets[task].append(targets[task].detach().cpu())
            else:
                loss = loss_fn(outputs, targets)

                # Handle thresholding for OVR loss
                if threshold > 0.0 and "type" in config["training"]["loss"] and config["training"]["loss"][
                    "type"] == "ovr":
                    # Get class probabilities
                    probs = torch.sigmoid(outputs)

                    # Apply threshold
                    unclassified_mask = torch.max(probs, dim=1).values < threshold

                    # Get predicted classes
                    _, predicted = torch.max(outputs, 1)

                    # Mark unclassified samples
                    predicted[unclassified_mask] = -1  # -1 as unclassified label

                    all_preds.append(predicted.detach().cpu())
                else:
                    # Standard predictions for regular loss
                    all_preds.append(outputs.detach().cpu())

                all_targets.append(targets.detach().cpu())

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update running loss
        running_loss += loss.item()

        # Log progress with improved formatting
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed_time = time.time() - start_time
            examples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed_time

            print(f"Epoch {epoch:3d} [{batch_idx + 1:4d}/{total_batches:4d}] "
                  f"({(batch_idx + 1) / total_batches * 100:5.1f}%) - "
                  f"Loss: {running_loss / (batch_idx + 1):.4f}, "
                  f"Examples/sec: {examples_per_sec:.1f}")

    # Compute metrics
    if hierarchical:
        # Concatenate predictions and targets for each task
        for task in task_preds:
            task_preds[task] = torch.cat(task_preds[task], 0)
            task_targets[task] = torch.cat(task_targets[task], 0)

        # Compute hierarchical metrics
        num_classes = {task: model.num_classes[task] for task in task_preds}
        metrics = compute_hierarchical_metrics(task_preds, task_targets, num_classes)
    else:
        # Concatenate predictions and targets
        all_preds = torch.cat(all_preds, 0)
        all_targets = torch.cat(all_targets, 0)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, model.num_classes if not threshold > 0.0 else None)

    # Add loss to metrics
    metrics["loss"] = running_loss / len(dataloader)

    # Log to TensorBoard
    if writer is not None:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"train/{metric_name}", metric_value, epoch)

    return metrics


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        hierarchical: bool = False,
        threshold: float = 0.0
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        config: Training configuration
        epoch: Current epoch number
        writer: TensorBoard writer
        hierarchical: Whether using hierarchical classification
        threshold: Classification threshold (for OVR loss)

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # For hierarchical validation
    task_preds = {} if hierarchical else None
    task_targets = {} if hierarchical else None

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            img = batch["img"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_attributes = batch["numerical_attributes"].to(device)

            if hierarchical:
                # For hierarchical model, labels are a dictionary
                targets = {task: batch[task].to(device) for task in batch if task != "img" and task != "input_ids"
                           and task != "attention_mask" and task != "numerical_attributes"}
            else:
                # For standard model, label is a single tensor
                targets = batch["label"].to(device)

            # Forward pass
            outputs = model(
                img=img,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_attributes=numerical_attributes
            )

            # Compute loss
            if hierarchical:
                # Sum losses from all tasks
                loss = sum([loss_fn(outputs[task], targets[task]) for task in outputs])

                # Store predictions and targets for metrics
                for task in outputs:
                    if task not in task_preds:
                        task_preds[task] = []
                        task_targets[task] = []
                    task_preds[task].append(outputs[task].detach().cpu())
                    task_targets[task].append(targets[task].detach().cpu())
            else:
                loss = loss_fn(outputs, targets)

                # Handle thresholding for OVR loss
                if threshold > 0.0 and "type" in config["training"]["loss"] and config["training"]["loss"][
                    "type"] == "ovr":
                    # Get class probabilities
                    probs = torch.sigmoid(outputs)

                    # Apply threshold
                    unclassified_mask = torch.max(probs, dim=1).values < threshold

                    # Get predicted classes
                    _, predicted = torch.max(outputs, 1)

                    # Mark unclassified samples
                    predicted[unclassified_mask] = -1  # -1 as unclassified label

                    all_preds.append(predicted.detach().cpu())
                else:
                    # Standard predictions for regular loss
                    all_preds.append(outputs.detach().cpu())

                all_targets.append(targets.detach().cpu())

            # Update running loss
            running_loss += loss.item()

    # Compute metrics
    if hierarchical:
        # Concatenate predictions and targets for each task
        for task in task_preds:
            task_preds[task] = torch.cat(task_preds[task], 0)
            task_targets[task] = torch.cat(task_targets[task], 0)

        # Compute hierarchical metrics
        num_classes = {task: model.num_classes[task] for task in task_preds}
        metrics = compute_hierarchical_metrics(task_preds, task_targets, num_classes)
    else:
        # Concatenate predictions and targets
        all_preds = torch.cat(all_preds, 0)
        all_targets = torch.cat(all_targets, 0)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, model.num_classes if not threshold > 0.0 else None)

    # Add loss to metrics
    metrics["loss"] = running_loss / len(dataloader)

    # Log to TensorBoard
    if writer is not None:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

    return metrics


def test(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        class_names: List[str],
        output_dir: Path,
        hierarchical: bool = False
) -> Dict[str, float]:
    """
    Test the model and generate visualizations.

    Args:
        model: Model to test
        dataloader: Test data loader
        device: Device to test on
        config: Training configuration
        class_names: List of class names
        output_dir: Directory to save outputs
        hierarchical: Whether using hierarchical classification

    Returns:
        Dictionary with test metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    # For hierarchical testing
    task_preds = {} if hierarchical else None
    task_targets = {} if hierarchical else None
    task_probs = {} if hierarchical else None

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            img = batch["img"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_attributes = batch["numerical_attributes"].to(device)

            if hierarchical:
                # For hierarchical model, labels are a dictionary
                targets = {task: batch[task].to(device) for task in batch if task != "img" and task != "input_ids"
                           and task != "attention_mask" and task != "numerical_attributes"}
            else:
                # For standard model, label is a single tensor
                targets = batch["label"].to(device)

            # Forward pass
            outputs = model(
                img=img,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_attributes=numerical_attributes
            )

            # Store predictions and targets
            if hierarchical:
                for task in outputs:
                    if task not in task_preds:
                        task_preds[task] = []
                        task_targets[task] = []
                        task_probs[task] = []

                    # Get class probabilities
                    probs = F.softmax(outputs[task], dim=1)

                    # Get predicted classes
                    _, predicted = torch.max(outputs[task], 1)

                    task_preds[task].append(predicted.detach().cpu())
                    task_targets[task].append(targets[task].detach().cpu())
                    task_probs[task].append(probs.detach().cpu())
            else:
                # Get class probabilities
                probs = F.softmax(outputs, dim=1)

                # Get predicted classes
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_probs.append(probs.detach().cpu())

    # Compute metrics and generate visualizations
    if hierarchical:
        # Process each task
        for task in task_preds:
            # Concatenate predictions and targets
            task_pred_tensor = torch.cat(task_preds[task], 0)
            task_target_tensor = torch.cat(task_targets[task], 0)
            task_prob_tensor = torch.cat(task_probs[task], 0)

            # Convert to numpy for visualization
            task_pred_np = task_pred_tensor.numpy()
            task_target_np = task_target_tensor.numpy()

            # Save confusion matrix
            task_cm_path = output_dir / f"confusion_matrix_{task}.png"
            task_class_names = class_names[task] if isinstance(class_names, dict) else class_names
            plot_confusion_matrix(
                task_pred_np,
                task_target_np,
                task_class_names,
                task_cm_path,
                normalize=True
            )

        # Compute hierarchical metrics
        num_classes = {task: model.num_classes[task] for task in task_preds}

        # Convert tensors to dictionaries for metrics computation
        task_pred_dict = {task: torch.cat(task_preds[task], 0) for task in task_preds}
        task_target_dict = {task: torch.cat(task_targets[task], 0) for task in task_targets}

        metrics = compute_hierarchical_metrics(task_pred_dict, task_target_dict, num_classes)
    else:
        # Concatenate predictions and targets
        all_pred_tensor = torch.cat(all_preds, 0)
        all_target_tensor = torch.cat(all_targets, 0)
        all_prob_tensor = torch.cat(all_probs, 0)

        # Convert to numpy for visualization
        all_pred_np = all_pred_tensor.numpy()
        all_target_np = all_target_tensor.numpy()

        # Save confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            all_pred_np,
            all_target_np,
            class_names,
            cm_path,
            normalize=True
        )

        # Compute metrics
        metrics = compute_metrics(all_prob_tensor, all_target_tensor, model.num_classes)

        # Calculate per-class metrics for analysis
        per_class_metrics = {}
        for class_idx in range(model.num_classes):
            # Binary mask for this class
            class_mask = (all_target_tensor == class_idx)
            class_pred = (all_pred_tensor == class_idx)

            # Skip if no samples of this class
            if class_mask.sum() == 0:
                continue

            # Accuracy for this class
            class_acc = (class_pred & class_mask).sum().item() / class_mask.sum().item()
            per_class_metrics[class_idx] = class_acc

        # Save per-class metrics
        per_class_path = output_dir / "per_class_accuracy.png"
        plot_per_class_metrics(
            per_class_metrics,
            class_names,
            per_class_path,
            metric_name="Accuracy"
        )

    # Save metrics to JSON
    metrics_path = output_dir / "test_metrics.json"
    save_json(metrics, str(metrics_path))

    return metrics


def get_optimizer(
        model: nn.Module,
        config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Get optimizer based on config.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        Configured optimizer
    """
    optimizer_name = config["name"].lower()
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
        optimizer: optim.Optimizer,
        config: Dict[str, Any],
        num_epochs: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on config.

    Args:
        optimizer: Optimizer
        config: Scheduler configuration
        num_epochs: Total number of training epochs

    Returns:
        Configured scheduler or None
    """
    scheduler_name = config["name"].lower()

    if scheduler_name == "cosine":
        # Cosine annealing with warmup
        warmup_epochs = config["params"]["warmup_epochs"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                # Cosine decay from 1 to 0 over the rest of training
                return 0.5 * (1 + torch.cos(torch.tensor(
                    (epoch - warmup_epochs) / (num_epochs - warmup_epochs) * torch.pi
                )).item())

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == "step":
        step_size = config["params"].get("step_size", 10)
        gamma = config["params"].get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "reduce_on_plateau":
        patience = config["params"].get("patience", 5)
        factor = config["params"].get("factor", 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=factor
        )
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train(config: dict):
    """
    Train and evaluate a model based on the provided configuration.

    Args:
        config: Dict including configuration param
    """
    # WandB login (should be done once per run)
    wandb.login()

    # Set random seed for reproducibility
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["logging"]["save_model_dir"]) / f"run_{timestamp}_config{config['config_num']}"
    ensure_dir(output_dir)

    # Save configuration for reproducibility
    save_json(config, str(output_dir / "config.json"))

    # Initialize Weights & Biases if enabled
    wandb_config = config["logging"].get("wandb", {})
    use_wandb = wandb_config.get("enabled", False)
    if use_wandb:
        wandb_run_name = wandb_config.get("name", f"run_{timestamp}_config{config['config_num']}")
        wandb_project = wandb_config.get("project", "mr-sequence-classification")
        wandb_entity = wandb_config.get("entity")
        wandb_tags = wandb_config.get("tags", [])
        wandb_notes = wandb_config.get("notes")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config,
            tags=wandb_tags,
            notes=wandb_notes,
            dir=str(output_dir)
        )
        # Optionally: wandb.save("*.py")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if hierarchical classification is used
    hierarchical = False  # Change to True if needed

    # Check for cached objects - this is the key part for using MONAI cache
    if "_cached_objects" in config:
        print("Using cached datasets")
        cached_objects = config["_cached_objects"]
        datasets = cached_objects["datasets"]
        label_dict = cached_objects["label_dict"]
        hierarchical_dicts = cached_objects["hierarchical_dicts"]
        num_normalizer = cached_objects["num_normalizer"]

        # Create dataloaders from cached datasets
        dataloaders = create_dataloaders(datasets, config)
    else:
        # Original code path - create datasets and dataloaders
        print("Creating new datasets (no caching between runs)")
        # Check if MONAI caching should be used within this single run
        use_monai = config["data"].get("use_monai", True)
        cache_type = config["data"].get("cache_type", "memory")

        datasets, label_dict, hierarchical_dicts, normalizer = create_datasets_with_lazy_support(
            config,
            hierarchical=config.get("hierarchical", False),
            use_monai=config.get("use_monai", True),
            cache_type=config.get("cache_type", "memory")
        )

    # Prepare model
    num_classes = len(label_dict)
    num_features = len(datasets["train"][0]["numerical_attributes"])
    model = get_classifier(
        config["model"],
        num_classes=num_classes,
        num_features=num_features,
        hierarchical=hierarchical,
        task_classes=hierarchical_dicts if hierarchical else None
    )
    model.to(device)

    # Print model summary
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Compute class counts for loss configuration
    class_counts = []
    for cls in range(num_classes):
        count = sum(1 for i in range(len(datasets["train"])) if datasets["train"][i]["label"] == cls)
        class_counts.append(count)

    # Setup loss function
    loss_config = config["training"].get("loss", {"type": "weighted_ce"})
    loss_type = loss_config.get("type", "weighted_ce")
    threshold = loss_config.get("threshold", 0.0)
    if loss_type == "ovr":
        loss_fn = get_loss_function("ovr", num_classes, class_counts)
        print(f"Using OVR loss with threshold {threshold}")
    elif loss_type == "weighted_ce" or config["training"]["class_weights"] == "balanced":
        class_weights = get_class_weights(datasets["train"], num_classes, method="balanced").to(device)
        loss_fn = get_loss_function("weighted_ce", num_classes, class_weights=class_weights)
        print(f"Using weighted cross-entropy loss with class weights")
    else:
        loss_fn = get_loss_function("ce", num_classes)
        print("Using standard cross-entropy loss")

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, config["training"]["optimizer"])
    scheduler = get_scheduler(optimizer, config["training"]["scheduler"], config["training"]["num_epochs"])

    # Setup early stopping, gradient scaler, and TensorBoard writer
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["min_delta"],
        mode="max"
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config["training"]["mixed_precision"])
    writer = None
    if config["logging"]["tensorboard"]:
        writer_dir = Path(config["logging"]["log_dir"]) / timestamp
        writer = SummaryWriter(log_dir=writer_dir)

    # Watch model with wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    train_metrics_history = []
    val_metrics_history = []
    best_val_metric = 0.0
    best_model_path = output_dir / "best_model.pth"
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
            writer=writer,
            hierarchical=hierarchical,
            threshold=threshold
        )
        train_metrics_history.append(train_metrics)

        val_metrics = validate(
            model=model,
            dataloader=dataloaders["val"],
            loss_fn=loss_fn,
            device=device,
            config=config,
            epoch=epoch,
            writer=writer,
            hierarchical=hierarchical,
            threshold=threshold
        )
        val_metrics_history.append(val_metrics)

        print(f"Epoch {epoch}/{num_epochs}:")
        print(
            f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(
            f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        if use_wandb:
            wandb_metrics = {
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/f1": train_metrics["f1"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1"],
                "epoch": epoch,
                "train/learning_rate": optimizer.param_groups[0]["lr"]
            }
            # Include any additional metrics if present
            for key, value in train_metrics.items():
                if key not in ["loss", "accuracy", "f1"]:
                    wandb_metrics[f"train/{key}"] = value
            for key, value in val_metrics.items():
                if key not in ["loss", "accuracy", "f1"]:
                    wandb_metrics[f"val/{key}"] = value
            wandb.log(wandb_metrics)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["f1"])
            else:
                scheduler.step()

        # Check early stopping
        monitor_metric = val_metrics["f1"]
        is_best, should_stop = early_stopping(monitor_metric)
        if is_best:
            print(f"  New best model with F1: {monitor_metric:.4f}")
            best_val_metric = monitor_metric
            save_model(model, best_model_path)
            if use_wandb:
                wandb.run.summary["best_val_f1"] = best_val_metric
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Plot and log training curves
    curves_dir = output_dir / "plots"
    ensure_dir(curves_dir)
    plot_training_curves(train_metrics_history, val_metrics_history, curves_dir)
    if use_wandb:
        for plot_file in curves_dir.glob("*.png"):
            wandb.log({f"plots/{plot_file.name}": wandb.Image(str(plot_file))})

    # Load best model for evaluation
    model = get_classifier(
        config["model"],
        num_classes=num_classes,
        num_features=num_features,
        hierarchical=hierarchical,
        task_classes=hierarchical_dicts if hierarchical else None
    )
    model.to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get class names for evaluation
    if hierarchical:
        class_names = {task: list(task_dict.keys()) for task, task_dict in hierarchical_dicts.items()}
    else:
        class_names = list(label_dict.keys())

    print("Evaluating best model on test set...")
    test_metrics = test(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        config=config,
        class_names=class_names,
        output_dir=output_dir,
        hierarchical=hierarchical
    )

    print("Test Results:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    if use_wandb:
        wandb_test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
        wandb.log(wandb_test_metrics)
        cm_path = output_dir / "confusion_matrix.png"
        if cm_path.exists():
            wandb.log({"test/confusion_matrix": wandb.Image(str(cm_path))})
        per_class_path = output_dir / "per_class_accuracy.png"
        if per_class_path.exists():
            wandb.log({"test/per_class_accuracy": wandb.Image(str(per_class_path))})

    # Save normalizer for inference
    num_normalizer.save(output_dir / "num_normalizer.pth")

    if use_wandb:
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Trained MRI sequence classifier (config {config['config_num']})"
        )
        model_artifact.add_file(str(best_model_path))
        wandb.log_artifact(model_artifact)
        wandb.finish()

    if writer is not None:
        writer.close()

    print(f"Training complete. Model saved to {best_model_path}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI sequence classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = read_json_config(args.config)
    train(config)