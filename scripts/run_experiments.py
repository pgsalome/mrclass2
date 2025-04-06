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


def run_bayesian_optimization(
        base_config: Dict[str, Any],
        output_dir: str,
        n_trials: int = 20,
        random_state: int = 42,
        results_file: str = "experiment_results.csv",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        high_dim: bool = False,
        n_jobs: int = 1
):
    """
    Run Bayesian optimization for hyperparameter tuning using Optuna.

    Args:
        base_config: Base configuration dictionary
        output_dir: Directory to save experiment configurations
        n_trials: Number of optimization trials
        random_state: Random seed
        results_file: Path to save experiment results
        study_name: Name for the Optuna study (for persistence)
        storage: Storage URL for Optuna (for persistence)
        high_dim: Whether to use high-dimensional search space
        n_jobs: Number of parallel jobs
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results tracking
    results = []
    best_score = -np.inf
    best_config = None
    best_params = None

    # Define the objective function for Optuna
    def objective(trial):
        nonlocal best_score, best_config, best_params, results

        # Get parameters for this trial
        params = define_study_params(trial, high_dim)

        # Create a new config with the parameters
        config = update_config_with_params(base_config, params)

        # Set config number and modify the wandb settings
        config_num = len(results)
        config["config_num"] = config_num

        # Modify Wandb configuration for this trial
        if "logging" in config and "wandb" in config["logging"] and config["logging"]["wandb"]["enabled"]:
            # Terminate any existing wandb run
            if wandb.run is not None:
                wandb.finish()

            # Create a unique run name for this trial
            config["logging"]["wandb"]["name"] = f"optuna_trial_{trial.number}"

            # Add group to associate all trials together
            config["logging"]["wandb"]["group"] = f"optuna_opt_{timestamp}"

            # Add trial-specific tags
            if "tags" not in config["logging"]["wandb"]:
                config["logging"]["wandb"]["tags"] = []

            config["logging"]["wandb"]["tags"].extend(["optuna", f"trial_{trial.number}"])

            # Make sure wandb doesn't try to resume the previous run
            os.environ["WANDB_RESUME"] = "never"

        # Print the configuration
        print(f"\nOptuna Trial {trial.number + 1}/{n_trials}")
        print(f"Parameters: {params}")

        # Save the configuration
        config_path = os.path.join(output_dir, f"config_{timestamp}_trial{trial.number}.json")
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
                "config_num": config_num,
                "trial_number": trial.number,
                "f1": score,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "class_balanced_acc": metrics.get("class_balanced_acc", 0)
            }

            # Add parameters to results
            for param_name, param_value in params.items():
                experiment_result[param_name] = param_value

            results.append(experiment_result)

            # Save results
            results_df = pd.DataFrame(results)
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

            print(f"Trial {trial.number} completed with F1 score: {score:.4f}")

            # Ensure wandb run is finished
            if wandb.run is not None:
                wandb.finish()

            return score

        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")

            # Ensure wandb run is finished even if there's an error
            if wandb.run is not None:
                wandb.finish()

            # # More graceful error handling
            # if "FixedLocator locations" in str(e) or "confusion_matrix" in str(e):
            #     print("Visualization error, but training may have completed successfully.")
            #     # Try to recover metrics if possible
            #     try:
            #         # Look for metrics in the logs or try to load the model and evaluate
            #         # This is a placeholder - implement model loading and evaluation logic
            #         print("Using default penalty for visualization error.")
            #         return 0.5  # Moderate penalty - Note: Optuna maximizes
            #     except:
            #         print("Could not recover metrics. Using larger penalty.")
            #         return 0.1  # Less severe than complete failure

            # Return a small value for failed trials
            return 0.01  # Severe penalty

    # Create or load an Optuna study
    if study_name and storage:
        # Resume or create a persistent study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
    else:
        # Create a new study for this run only
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

    print(f"Starting Optuna optimization with {n_trials} trials...")

    try:
        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        # Generate and save visualizations
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
            print(f"Warning: Could not generate Optuna visualization plots: {e}")

        # Get best parameters from the study
        best_trial = study.best_trial

        # Print and save the best results
        print("\nOptuna Optimization Results:")
        print(f"Best F1 score: {best_trial.value:.4f}")
        print("Best parameters:")
        for param_name, param_value in best_trial.params.items():
            print(f"  {param_name}: {param_value}")

        # Create a comprehensive report
        report = {
            "timestamp": timestamp,
            "best_score": float(best_trial.value),
            "best_parameters": best_trial.params,
            "total_trials": len(study.trials),
            "runtime_info": {
                "start_time": timestamp,
                "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_trials": n_trials,
                "random_state": random_state
            }
        }

        with open(os.path.join(output_dir, f"optimization_report_{timestamp}.json"), "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Saving current best results...")

        # Save what we have so far
        with open(os.path.join(output_dir, f"interrupted_best_config_{timestamp}.json"), "w") as f:
            if best_config:
                json.dump(best_config, f, indent=2, cls=NumpyEncoder)
            else:
                json.dump({"status": "interrupted_before_best_found"}, f, indent=2, cls=NumpyEncoder)

    # Ensure all wandb runs are closed
    if wandb.run is not None:
        wandb.finish()

    return best_config, best_score, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MRI sequence classification experiments with Bayesian optimization using Optuna")
    parser.add_argument("--base_config", default="./config/default.json", type=str, help="Path to base config file")
    parser.add_argument("--output_dir", type=str, default="./config/bayesian_opt",
                        help="Directory to save experiment configs")
    parser.add_argument("--results_file", type=str, default="logs/bayesian_opt_results.csv",
                        help="Path to save experiment results")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study_name", type=str, default=None, help="Name for Optuna study (for persistence)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    parser.add_argument("--high_dim", action="store_true", help="Use high-dimensional search space")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()

    # Load base configuration
    base_config = load_base_config(args.base_config)

    # Run Bayesian optimization with Optuna
    best_config, best_score, results = run_bayesian_optimization(
        base_config=base_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        random_state=args.random_seed,
        results_file=args.results_file,
        study_name=args.study_name,
        storage=args.storage,
        high_dim=args.high_dim,
        n_jobs=args.n_jobs
    )

    print("\nBayesian optimization completed.")
    print(f"Best F1 score: {best_score:.4f}")
    print(f"Results saved to {args.results_file}")