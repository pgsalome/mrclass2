import wandb
import os
from typing import Optional, List  # Optional type hinting


# Ensure imports are usually at the top level of your script/module
# import wandb
# import os

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


# --- Example Usage ---
if __name__ == "__main__":
    # --- Use the run path from your previous message ---
    target_run_path = "pgsalome/mrclass2/0ta0u2bm"

    print(f"Attempting to fetch metrics for run: {target_run_path}")

    # Call the function
    metrics = get_wandb_run_metrics(target_run_path)

    if metrics is not None:
        print("\n--- Available Metrics Logged Over Time ---")
        if metrics:
            # Format as a markdown list (or simple print)
            for metric_name in metrics:
                print(f"* `{metric_name}`")  # Using backticks for code formatting
        else:
            print("No user-defined metrics found in the run's history.")
    else:
        print("\nFailed to retrieve metrics.")

    # Example with a potentially non-existent run path (will trigger the error)
    # print("\nAttempting to fetch metrics for a non-existent run:")
    # non_existent_path = "user/project/invalidrunid"
    # metrics_invalid = get_wandb_run_metrics(non_existent_path)
    # if metrics_invalid is None:
    #     print("As expected, failed to retrieve metrics for the invalid path.")