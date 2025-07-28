import sys
import os
import pickle
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass


# Define the encodedSample dataclass here, matching its definition in final_cleaned.py
# This is crucial so pickle can correctly reconstruct the objects.
@dataclass
class encodedSample:
    """Class for holding encoded DICOM data after preprocessing."""
    img: any  # np.ndarray
    input_ids: List[int]
    attention_mask: List[int]
    numerical_attributes: List[float]
    label: int
    orientation: str


def inspect_pickle_file(file_path: Union[str, Path], description: str, num_samples_to_inspect: int = 5):
    """
    Loads a pickle file and prints details about its contents.
    Assumes the pickle file contains a list of encodedSample objects.
    """
    file_path = Path(file_path)

    print(f"\n--- Inspecting {description} file: {file_path} ---")

    if not file_path.exists():
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded '{file_path.name}'.")

        if not isinstance(data, list):
            print(f"Warning: Expected a list, but loaded data is of type: {type(data)}")
            print(f"First few elements (if any): {str(data)[:200]}...")
            return

        print(f"Total samples in {file_path.name}: {len(data)}")

        if not data:
            print("No samples found in this file.")
            return

        print(f"First {min(num_samples_to_inspect, len(data))} samples:")
        for i, sample in enumerate(data[:num_samples_to_inspect]):
            print(f"  --- Sample {i + 1} ---")
            if isinstance(sample, encodedSample):
                print(f"    Type: encodedSample")
                print(f"    Image shape: {sample.img.shape if hasattr(sample.img, 'shape') else 'N/A'}")
                print(f"    Label (int ID): {sample.label}")
                print(f"    Orientation: {sample.orientation}")
                print(f"    Input IDs length: {len(sample.input_ids)}")
                print(f"    Attention Mask length: {len(sample.attention_mask)}")
                if sample.numerical_attributes is not None:
                    print(f"    Numerical Attributes: {sample.numerical_attributes[:5]}... (first 5 values)")
            else:
                print(f"    Unexpected sample type: {type(sample)}")
                print(f"    Sample content (raw): {str(sample)[:200]}...")

    except Exception as e:
        print(f"An error occurred during unpickling '{file_path.name}': {e}")


# --- Define the specific path to dataset_batch_final.pkl for COR orientation ---
output_base_dir = Path("/media/e210/portable_hdd/d_bodypart_final_cleaned")
final_batch_cor_path = output_base_dir / "COR" / "batches" / "dataset_batch_final.pkl"

# --- Perform inspection ---
inspect_pickle_file(final_batch_cor_path, "final COR batch file")

# You can remove or comment out the 'first_cor_batch_file' logic
# and the conditional check for it, as your request is very specific.
# If dataset_batch_final.pkl doesn't exist, it will correctly report FileNotFoundError.