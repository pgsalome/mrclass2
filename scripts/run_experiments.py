import os
import json
import copy
import argparse
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from tqdm import tqdm

from utils.io import read_json_config, ensure_dir, save_json
from train import train


def load_base_config(config_path: str) -> Dict[str, Any]:
    """
    Load the base configuration file.

    Args:
        config_path: Path to the base configuration file

    Returns:
        Base configuration dictionary
    """
    return read_json_config(config_path)


def generate_experiment_configs(
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        output_dir: str
) -> List[str]:
    """
    Generate experiment configurations from a parameter grid.

    Args:
        base_config: Base configuration dictionary
        param_grid: Dictionary mapping parameter paths to lists of values
        output_dir: Directory to save experiment configurations

    Returns:
        List of paths to generated configuration files
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate total number of experiments
    total_experiments = 1
    for values in param_values:
        total_experiments *= len(values)

    print(f"Generating {total_experiments} experiment configurations...")

    config_paths = []

    # Generate all combinations
    for i, combination in enumerate(itertools.product(*param_values)):
        # Create a new config
        config = copy.deepcopy(base_config)
        config["config_num"] = i

        # Set wandb run name to reflect config number
        if "wandb" in config["logging"] and config["logging"]["wandb"]["enabled"]:
            config["logging"]["wandb"]["name"] = f"experiment_{i}"

        # Update config with parameter values
        for j, param_path in enumerate(param_names):
            value = combination[j]

            # Convert period-separated path to nested dictionary keys
            keys = param_path.split(".")

            # Navigate to the right level in the config
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

        # Save config
        config_path = os.path.join(output_dir, f"config_{i}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        config_paths.append(config_path)

    print(f"Generated {len(config_paths)} configuration files in {output_dir}")
    return config_paths


def run_experiments(config_paths: List[str], results_file: str = "experiment_results.csv"):
    """
    Run experiments for each configuration.

    Args:
        config_paths: List of paths to configuration files
        results_file: Path to save experiment results
    """
    results = []

    print(f"Running {len(config_paths)} experiments...")

    for i, config_path in enumerate(config_paths):
        print(f"\nExperiment {i + 1}/{len(config_paths)}")
        print(f"Configuration: {config_path}")

        try:
            # Load configuration
            config = read_json_config(config_path)

            # Run training
            metrics = train(config_path)

            # Extract key parameters for results table
            experiment_result = {
                "config_num": config["config_num"],
                "accuracy": metrics.get("accuracy", 0),
                "f1": metrics.get("f1", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "class_balanced_acc": metrics.get("class_balanced_acc", 0)
            }

            # Add key hyperparameters
            experiment_result["img_encoder"] = config["model"]["img_encoder"]["name"]
            experiment_result["txt_encoder"] = config["model"]["txt_encoder"]["name"]
            experiment_result["fusion_method"] = config["model"]["fusion"]["method"]
            experiment_result["batch_size"] = config["data"]["batch_size"]
            experiment_result["learning_rate"] = config["training"]["optimizer"]["lr"]

            # Add intensity normalization if used
            if config["data"].get("intensity_normalization", {}).get("enabled", False):
                experiment_result["intensity_norm"] = config["data"]["intensity_normalization"]["method"]
            else:
                experiment_result["intensity_norm"] = "None"

            results.append(experiment_result)

            # Save results after each experiment
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file, index=False)

        except Exception as e:
            print(f"Error in experiment {i + 1}: {str(e)}")
            # Continue with next experiment even if one fails
            continue

    # Create final results table
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

        # Print summary
        print("\nExperiment Results Summary:")
        print(results_df.sort_values(by="f1", ascending=False).head(10))

        # Find best configuration
        best_idx = results_df["f1"].idxmax()
        best_config_num = results_df.loc[best_idx, "config_num"]
        best_f1 = results_df.loc[best_idx, "f1"]

        print(f"\nBest configuration: {best_config_num} with F1 score: {best_f1:.4f}")
    else:
        print("No successful experiments to report.")


def create_parameter_grid() -> Dict[str, List[Any]]:
    """
    Create a parameter grid for experiments.

    Returns:
        Parameter grid dictionary
    """
    return {
        # Image encoders
        "model.img_encoder.name": ["resnet18", "resnet50", "densenet121", "efficientnet_b0"],

        # Text encoders
        "model.txt_encoder.name": ["distilbert-base-uncased", "bert-base-uncased"],

        # Fusion methods
        "model.fusion.method": ["concat", "attention", "gated"],

        # Batch sizes
        "data.batch_size": [16, 32, 64],

        # Learning rates
        "training.optimizer.lr": [1e-4, 1e-5],

        # Intensity normalization
        "data.intensity_normalization.enabled": [False, True],
        "data.intensity_normalization.method": [None, "zscore", "whitestripe"]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple MRI sequence classification experiments")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config file")
    parser.add_argument("--output_dir", type=str, default="./config/experiments",
                        help="Directory to save experiment configs")
    parser.add_argument("--results_file", type=str, default="experiment_results.csv",
                        help="Path to save experiment results")
    parser.add_argument("--custom_grid", type=str, default=None, help="Path to custom parameter grid JSON")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate configs without running experiments")
    args = parser.parse_args()

    # Load base configuration
    base_config = load_base_config(args.base_config)

    # Load or create parameter grid
    if args.custom_grid:
        with open(args.custom_grid, 'r') as f:
            param_grid = json.load(f)
    else:
        param_grid = create_parameter_grid()

    # Generate experiment configurations
    config_paths = generate_experiment_configs(
        base_config=base_config,
        param_grid=param_grid,
        output_dir=args.output_dir
    )

    # Run experiments if not in generate-only mode
    if not args.generate_only:
        run_experiments(config_paths, args.results_file)
    else:
        print("Configurations generated. Use --generate_only=False to run experiments.")

        # Print parameter combinations summary
        param_summary = {}
        for param, values in param_grid.items():
            param_summary[param] = len(values)
            print(f"{param}: {values}")

        print(f"\nTotal configurations: {len(config_paths)}")
        print(f"Configurations saved to {args.output_dir}")