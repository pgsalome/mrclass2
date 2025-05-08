import os
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
import glob

# Import Optuna for Bayesian optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import Trial

# Import wandb for handling runs
import wandb

from utils.io import read_json_config, ensure_dir, save_json
from train import train

os.environ["WANDB_DIR"] = "/media/e210/portable_hdd/wandb"


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def load_base_config(config_path: str) -> Dict[str, Any]:
    """
    Load the base configuration file.

    Args:
        config_path: Path to the base configuration file

    Returns:
        Base configuration dictionary
    """
    return read_json_config(config_path)


def update_config_with_params(base_config: Dict[str, Any], param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with parameters.

    Args:
        base_config: Base configuration dictionary
        param_dict: Dictionary of parameters (with nested paths as keys)

    Returns:
        Updated configuration dictionary
    """
    config = copy.deepcopy(base_config)

    for param_name, param_value in param_dict.items():
        # Convert period-separated path to nested dictionary keys
        keys = param_name.split(".")

        # Navigate to the right level in the config
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = param_value

    return config


def define_study_params(trial: Trial, high_dim: bool = False) -> Dict[str, Any]:
    """
    Define the parameters for a trial in the Optuna study.

    Args:
        trial: Optuna trial object
        high_dim: Whether to use high-dimensional search space

    Returns:
        Dictionary of parameters
    """
    params = {}

    # Common parameters (always included)
    params["model.img.encoder.name"] = trial.suggest_categorical(
        "model.img.encoder.name", ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
    )

    params["model.img.encoder.freeze.backbone"] = trial.suggest_categorical(
        "model.img.encoder.freeze.backbone", [True, False]
    )

    params["model.txt.encoder.name"] = trial.suggest_categorical(
        "model.txt.encoder.name", ["distilbert-base-uncased", "bert-base-uncased"]
    )

    params["model.fusion.method"] = trial.suggest_categorical(
        "model.fusion.method", ["concat", "attention", "gated"]
    )

    params["data.batch.size"] = trial.suggest_categorical(
        "data.batch.size", [16, 32, 64]
    )

    params["training.optimizer.lr"] = trial.suggest_float(
        "training.optimizer.lr", 1e-6, 1e-4, log=True
    )

    params["data.intensity.normalization.enabled"] = trial.suggest_categorical(
        "data.intensity.normalization.enabled", [True, False]
    )

    # Use None as a valid option for normalization method
    method_options = ["zscore", "whitestripe", None]
    method_idx = trial.suggest_categorical("data.intensity.normalization.method_idx", list(range(len(method_options))))
    params["data.intensity.normalization.method"] = method_options[method_idx]

    params["model.classifier.dropout"] = trial.suggest_float(
        "model.classifier.dropout", 0.1, 0.5
    )

    # Additional parameters for high-dimensional search
    if high_dim:
        params["model.img.encoder.output_dim"] = trial.suggest_int(
            "model.img.encoder.output_dim", 256, 1024
        )

        params["model.txt.encoder.freeze.backbone"] = trial.suggest_categorical(
            "model.txt.encoder.freeze.backbone", [True, False]
        )

        params["model.txt.encoder.output_dim"] = trial.suggest_int(
            "model.txt.encoder.output_dim", 128, 512
        )

        params["model.num.encoder.hidden_dim"] = trial.suggest_int(
            "model.num.encoder.hidden_dim", 32, 256
        )

        params["model.num.encoder.output_dim"] = trial.suggest_int(
            "model.num.encoder.output_dim", 16, 128
        )

        params["model.num.encoder.num_layers"] = trial.suggest_int(
            "model.num.encoder.num_layers", 1, 3
        )

        params["model.num.encoder.dropout"] = trial.suggest_float(
            "model.num.encoder.dropout", 0.0, 0.5
        )

        params["model.fusion.hidden_size"] = trial.suggest_int(
            "model.fusion.hidden_size", 256, 1024
        )

        params["training.optimizer.weight_decay"] = trial.suggest_float(
            "training.optimizer.weight_decay", 1e-7, 1e-5, log=True
        )

    return params


def extract_trial_number_from_filename(filename):
    """Extract trial number from config filename."""
    match = re.search(r'trial(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return -1


def extract_suggestions_from_config(config_path):
    """Extract parameter suggestions from a configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    suggestions = {}

    # Extract model.img.encoder.name
    if "img" in config["model"] and "encoder" in config["model"]["img"]:
        suggestions["model.img.encoder.name"] = config["model"]["img"]["encoder"].get("name", "resnet18")

    # Extract model.img.encoder.freeze.backbone
    if "img" in config["model"] and "encoder" in config["model"]["img"]:
        suggestions["model.img.encoder.freeze.backbone"] = config["model"]["img"]["encoder"].get("freeze", {}).get(
            "backbone", False)

    # Extract model.txt.encoder.name
    if "txt" in config["model"] and "encoder" in config["model"]["txt"]:
        suggestions["model.txt.encoder.name"] = config["model"]["txt"]["encoder"].get("name", "distilbert-base-uncased")

    # Extract model.fusion.method
    if "fusion" in config["model"]:
        suggestions["model.fusion.method"] = config["model"]["fusion"].get("method", "concat")

    # Extract data.batch.size
    if "batch" in config["data"]:
        suggestions["data.batch.size"] = config["data"]["batch"].get("size", 32)

    # Extract training.optimizer.lr
    if "optimizer" in config["training"]:
        suggestions["training.optimizer.lr"] = config["training"]["optimizer"].get("lr", 1e-5)

    # Extract data.intensity.normalization.enabled
    if "intensity" in config["data"]:
        suggestions["data.intensity.normalization.enabled"] = config["data"]["intensity"].get("normalization", {}).get(
            "enabled", False)

    # Extract data.intensity.normalization.method
    method = None
    if "intensity" in config["data"]:
        method = config["data"]["intensity"].get("normalization", {}).get("method", None)

    # Map method to index
    methods = ["zscore", "whitestripe", None]
    if method in methods:
        suggestions["data.intensity.normalization.method_idx"] = methods.index(method)
    else:
        suggestions["data.intensity.normalization.method_idx"] = 2  # Default to None

    # Extract model.classifier.dropout
    if "classifier" in config["model"]:
        suggestions["model.classifier.dropout"] = config["model"]["classifier"].get("dropout", 0.2)

    return suggestions


def find_completed_trials(output_dir: str) -> tuple:
    """
    Find all completed trials in the output directory.

    Args:
        output_dir: Directory containing configuration files

    Returns:
        Tuple of (max_trial_number, config_files_dict)
    """
    # Find all config files
    config_files = glob.glob(os.path.join(output_dir, "config_*_trial*.json"))

    # Extract trial numbers and create mapping
    config_files_dict = {}
    max_trial_number = -1

    for config_file in config_files:
        trial_number = extract_trial_number_from_filename(config_file)
        if trial_number >= 0:
            config_files_dict[trial_number] = config_file
            max_trial_number = max(max_trial_number, trial_number)

    return max_trial_number, config_files_dict


def load_previous_trials(results_file, trial_numbers):
    """
    Load previous trial results from CSV file.

    Args:
        results_file: Path to CSV with previous results
        trial_numbers: List of trial numbers to load

    Returns:
        Dictionary mapping trial numbers to results
    """
    if not os.path.exists(results_file):
        return {}

    results_df = pd.read_csv(results_file)

    # Filter for the specified trial numbers and convert to dictionary
    trial_results = {}
    for _, row in results_df.iterrows():
        if 'trial_number' in row and row['trial_number'] in trial_numbers:
            trial_results[row['trial_number']] = row.to_dict()

    return trial_results


def run_custom_bayesian_optimization(
        base_config: Dict[str, Any],
        output_dir: str,
        start_trial: int = 0,
        n_trials: int = 20,
        random_state: int = 42,
        results_file: str = "experiment_results.csv",
        high_dim: bool = False
):
    """
    Simpler implementation that explicitly controls trial numbering.
    This approach bypasses Optuna's automatic trial numbering to ensure consistency.

    Args:
        base_config: Base configuration dictionary
        output_dir: Directory to save experiment configurations
        start_trial: Trial number to start from
        n_trials: Total number of trials to run
        random_state: Random seed
        results_file: Path to save experiment results
        high_dim: Whether to use high-dimensional search space
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if we need to load previous results
    resuming = start_trial > 0

    if resuming:
        print(f"Resuming from trial {start_trial} to {n_trials}.")
        # Load previous trials to guide the new trials (not strictly necessary)
        previous_results = []
        if os.path.exists(results_file):
            try:
                previous_df = pd.read_csv(results_file)
                previous_results = previous_df.to_dict('records')

                # Find best result so far
                best_f1 = max([r.get('f1', 0) for r in previous_results], default=0)
                print(f"Loaded {len(previous_results)} previous results. Best F1 so far: {best_f1:.4f}")
            except Exception as e:
                print(f"Error loading previous results: {e}")
                previous_results = []
    else:
        print(f"Starting fresh optimization with {n_trials} trials.")
        previous_results = []

    # List to store new results
    new_results = []

    # Create a sampler for parameter exploration
    sampler = optuna.samplers.TPESampler(seed=random_state)

    # Create a study object (but we won't use study.optimize directly)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Initialize tracking variables
    best_score = -np.inf
    best_config = None
    best_params = None

    # Load previous best score if available
    if previous_results:
        for result in previous_results:
            f1_score = result.get('f1', 0)
            if f1_score > best_score:
                best_score = f1_score
                # Extract parameters from this result
                params = {k: v for k, v in result.items()
                          if k.startswith('model.') or k.startswith('data.') or k.startswith('training.')}
                best_params = params

    # For each trial to run
    trials_to_run = n_trials - start_trial
    print(f"Will run {trials_to_run} additional trials (from {start_trial} to {n_trials - 1}).")

    if trials_to_run <= 0:
        print("No additional trials needed.")
        return best_config, best_score, new_results

    for trial_num in range(start_trial, n_trials):
        # Create a trial object
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

        print(f"\nTrial {trial_num}/{n_trials - 1}")

        # Get parameters for this trial
        params = define_study_params(trial, high_dim)
        print(f"Parameters: {params}")

        # Create a new config with the parameters
        config = update_config_with_params(base_config, params)

        # Set config number and modify the wandb settings
        config["config_num"] = trial_num

        # Modify Wandb configuration for this trial
        if "logging" in config and "wandb" in config["logging"] and config["logging"]["wandb"]["enabled"]:
            # Terminate any existing wandb run
            if wandb.run is not None:
                wandb.finish()

            # Create a unique run name for this trial
            config["logging"]["wandb"]["name"] = f"trial_{trial_num}"

            # Add group to associate all trials together
            group_prefix = "resumed_opt" if resuming else "opt"
            config["logging"]["wandb"]["group"] = f"{group_prefix}_{timestamp}"

            # Add trial-specific tags
            if "tags" not in config["logging"]["wandb"]:
                config["logging"]["wandb"]["tags"] = []

            tag_prefix = "resumed_" if resuming else ""
            config["logging"]["wandb"]["tags"].extend([f"{tag_prefix}optimization", f"trial_{trial_num}"])

            # Make sure wandb doesn't try to resume the previous run
            os.environ["WANDB_RESUME"] = "never"

        # Save the configuration
        config_path = os.path.join(output_dir, f"config_{timestamp}_trial{trial_num}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        try:
            # Run training
            metrics = train(config)

            # Get the score (F1 score to maximize)
            score = metrics.get("f1", 0)

            # Record the results
            experiment_result = {
                "timestamp": timestamp,
                "config_num": trial_num,
                "trial_number": trial_num,
                "f1": score,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "class_balanced_acc": metrics.get("class_balanced_acc", 0)
            }

            # Add parameters to results
            for param_name, param_value in params.items():
                experiment_result[param_name] = param_value

            # Add to results list
            new_results.append(experiment_result)

            # Save results to CSV
            if os.path.exists(results_file):
                # Load existing results and append new ones
                existing_df = pd.read_csv(results_file)
                new_df = pd.DataFrame([experiment_result])
                results_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Create new results file
                results_df = pd.DataFrame([experiment_result])

            results_df.to_csv(results_file, index=False)

            # Update best score and config
            if score > best_score:
                best_score = score
                best_config = config
                best_params = params

                # Save the best configuration separately
                best_config_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
                with open(best_config_path, "w") as f:
                    json.dump(best_config, f, indent=2, cls=NumpyEncoder)

                print(f"New best score: {score:.4f}")

            # Tell Optuna about this result for future sampling
            study.tell(trial, score)

            print(f"Trial {trial_num} completed with F1 score: {score:.4f}")

        except Exception as e:
            print(f"Error in trial {trial_num}: {str(e)}")

            # Tell Optuna this trial failed
            study.tell(trial, 0.01)  # Severe penalty

        finally:
            # Ensure wandb run is finished
            if wandb.run is not None:
                wandb.finish()

    # Create a comprehensive report
    report = {
        "timestamp": timestamp,
        "best_score": float(best_score),
        "best_parameters": best_params,
        "total_trials": n_trials,
        "resumed_from_trial": start_trial if resuming else None,
        "runtime_info": {
            "start_time": timestamp,
            "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "n_trials": n_trials,
            "random_state": random_state
        }
    }

    with open(os.path.join(output_dir, f"optimization_report_{timestamp}.json"), "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    # Generate visualization if enough trials completed
    if len(study.trials) > 2:
        try:
            # Plot optimization history
            fig = plot_optimization_history(study)
            fig.update_layout(width=1000, height=600)
            fig.write_image(os.path.join(output_dir, f"optimization_history_{timestamp}.png"))

            # Plot parameter importances
            fig = plot_param_importances(study)
            fig.update_layout(width=1000, height=600)
            fig.write_image(os.path.join(output_dir, f"param_importances_{timestamp}.png"))

        except Exception as e:
            print(f"Warning: Could not generate visualization plots: {e}")

    return best_config, best_score, new_results


def find_last_completed_trial(output_dir: str) -> int:
    """
    Find the highest trial number in the output directory.

    Args:
        output_dir: Directory containing configuration files

    Returns:
        Highest trial number found, or -1 if none found
    """
    max_trial, _ = find_completed_trials(output_dir)
    return max_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MRI sequence classification experiments with Bayesian optimization")
    parser.add_argument("--base_config", default="./config/default.json", type=str, help="Path to base config file")
    parser.add_argument("--output_dir", type=str, default="./config/bayesian_opt_sag_anal",
                        help="Directory to save experiment configs")
    parser.add_argument("--results_file", type=str, default="logs/bayesian_opt_results_anal_sag.csv",
                        help="Path to save experiment results")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start_trial", type=int, default=None,
                        help="Trial number to start from (if None, auto-detect)")
    parser.add_argument("--high_dim", action="store_true", help="Use high-dimensional search space")
    args = parser.parse_args()

    # Load base configuration
    base_config = load_base_config(args.base_config)

    # Determine start trial
    if args.start_trial is not None:
        start_trial = args.start_trial
    else:
        # Auto-detect the last completed trial
        last_trial = find_last_completed_trial(args.output_dir)
        start_trial = last_trial + 1 if last_trial >= 0 else 0

    print(f"Starting from trial {start_trial}")

    # Run custom Bayesian optimization
    best_config, best_score, results = run_custom_bayesian_optimization(
        base_config=base_config,
        output_dir=args.output_dir,
        start_trial=start_trial,
        n_trials=args.n_trials,
        random_state=args.random_seed,
        results_file=args.results_file,
        high_dim=args.high_dim
    )

    print("\nBayesian optimization completed.")
    print(f"Best F1 score: {best_score:.4f}")
    print(f"Results saved to {args.results_file}")