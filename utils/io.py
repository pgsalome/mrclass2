import os
import json
import yaml
import pickle
from pathlib import Path
from typing import Dict, Any, Union, List
import torch
import numpy as np
import wandb
import os
from typing import Optional, List # Optional type hinting


def read_yaml_config(config_path: str) -> Any:
    """
    Read YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed configuration object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert to namespace for dot notation access
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = dict_to_namespace(value)
            return Namespace(**d)
        return d

    return dict_to_namespace(config)


def read_json_config(config_path: str) -> Dict[str, Any]:
    """
    Read JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save dictionary as JSON.

    Args:
        data: Dictionary to save
        path: Path to save JSON file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_pickle(path: str) -> Any:
    """
    Load pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Unpickled object
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, path: str) -> None:
    """
    Save object as pickle.

    Args:
        data: Object to save
        path: Path to save pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save PyTorch model.

    Args:
        model: PyTorch model to save
        path: Path to save model
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device = None) -> torch.nn.Module:
    """
    Load weights into PyTorch model.

    Args:
        model: PyTorch model instance
        path: Path to model weights file
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def list_files(directory: str, extension: str = None) -> List[str]:
    """
    List all files in a directory, optionally filtered by extension.

    Args:
        directory: Directory path
        extension: Optional file extension to filter by

    Returns:
        List of file paths
    """
    files = []
    for file in os.listdir(directory):
        if extension is None or file.endswith(extension):
            files.append(os.path.join(directory, file))
    return files


def get_wandb_run_metrics(run_path: str) -> Optional[List[str]]:
    """
    Fetches and returns the names of metrics logged over time for a specific W&B run.

    Args:
        run_path: The path to the W&B run (e.g., "username/project/run_id").

    Returns:
        A list of metric names (strings) logged in the run's history,
        excluding internal W&B columns (like _runtime, _timestamp, _step).
        Returns None if an error occurs during fetching or processing.
    """
    # Make sure you are logged in via CLI `wandb login`
    # or set the WANDB_API_KEY environment variable

    try:
        api = wandb.Api()
        run = api.run(run_path)

        # --- Get all metrics logged over time ---
        history_df = run.history()  # Fetch the history data

        metric_names = []
        # Collect column names (these are your available metrics)
        for col_name in history_df.columns:
            # Skip internal wandb columns unless needed
            if not col_name.startswith('_'):
                metric_names.append(col_name)

        return metric_names

    except Exception as e:
        print(f"Error accessing W&B run '{run_path}': {e}")
        print("Please ensure the run path is correct and you have access.")
        return None  # Indicate failure by returning None