import json
from pathlib import Path
from collections import Counter
from typing import List, Union, Set
from argparse import ArgumentParser, Namespace
import sys  # For printing to stderr
import re  # For normalize_class_name logic if needed (though _get_estimated_label_from_path simplifies it)


# --- Simplified read_json_config for standalone script ---
# This is a minimalistic version that only extracts what's strictly necessary.
def read_json_config_simple(config_path: Union[str, Path]) -> Namespace:
    if not Path(config_path).exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Simple recursive conversion to Namespace
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
            elif isinstance(v, list):
                # Ensure lists for filtering are converted to sets for efficiency
                # This assumes filtering lists contain simple, hashable types.
                if k in ['axes', 'anatomies', 'sequences', 'ignored_modalities', 'filter_patterns']:
                    d[k] = set(v)
                else:
                    d[k] = v
        return Namespace(**d)

    return dict_to_namespace(config_dict)


# --- Helper to get label from path (copied from main script for consistency) ---
def _get_estimated_label_from_path(sequence_dir_path: Path, args: Namespace) -> str:
    """Estimates the label string from the path based on classification target."""
    # Assuming sequence_dir_path is the 'series_description_folder' (6th level after pycurt_dir)
    # Structure: pycurt_dir/patient_id/study_id/axis_name/anatomy_name/sequence_type_folder/series_description_folder/

    # sequence_dir_path.name: 'series_description_folder' (e.g., '1-ABDOMENKM50B31F')
    # sequence_dir_path.parent.name: 'sequence_type_folder' (e.g., 'CCT')
    # sequence_dir_path.parent.parent.name: 'anatomy_name' (e.g., 'ANP')
    # sequence_dir_path.parent.parent.parent.name: 'axis_name' (e.g., 'TRA')

    if hasattr(args.classification, 'target'):
        if args.classification.target == 'body_part':
            # Label is the anatomy_name
            return sequence_dir_path.parent.parent.name
        elif args.classification.target == 'sequence':
            # Label is the sequence_type_folder (first part)
            return sequence_dir_path.parent.name.split("-")[0]
    # Default behavior if classification.target is not specified or recognized
    return sequence_dir_path.parent.name.split("-")[0]


def main():
    parser = ArgumentParser(description="Generate statistics of labels found in raw DICOM directories.")
    parser.add_argument("-c", "--config_path", type=str, default="./config/preproc_config_bp.json",
                        help="Path to JSON config file.")
    args = parser.parse_args()

    config = read_json_config_simple(args.config_path)

    pycurt_dirs = []
    if hasattr(config.paths, 'pycurt_dir'):
        pycurt_dirs = [Path(config.paths.pycurt_dir)]
    elif hasattr(config.paths, 'pycurt_dirs'):
        pycurt_dirs = [Path(dir_path) for dir_path in config.paths.pycurt_dirs]
    else:
        print("Error: No pycurt directory specified in config.", file=sys.stderr)
        sys.exit(1)

    # --- Filtering parameters from config ---
    # These are used for initial filtering *before* counting for statistics.
    # We retrieve them using getattr to be robust to missing keys.
    ignored_modalities_set = getattr(config, 'ignored_modalities', set())

    use_axes_filter = False
    accepted_axes_set = set()
    if hasattr(config.filtering, 'axes') and config.filtering.axes:
        accepted_axes_set = set(config.filtering.axes)
        use_axes_filter = True

    use_anatomies_filter = False
    accepted_anatomies_set = set()
    if hasattr(config.filtering, 'anatomies') and config.filtering.anatomies:
        accepted_anatomies_set = set(config.filtering.anatomies)
        use_anatomies_filter = True

    use_sequences_filter = False
    accepted_sequences_set = set()
    if hasattr(config.filtering, 'sequences') and config.filtering.sequences:
        accepted_sequences_set = set(config.filtering.sequences)
        use_sequences_filter = True

    # We don't apply `filter_patterns` (from processing_params) here, as its purpose is later cleaning.
    # This script focuses on the raw counts from directories after basic structural filters.

    print(f"\n--- Scanning directories and collecting label statistics ---")
    print(f"Source directories: {[str(p) for p in pycurt_dirs]}")
    print(f"Ignoring modalities: {ignored_modalities_set}")
    print(f"Filtering by axes: {accepted_axes_set if use_axes_filter else 'None (all included)'}")
    print(f"Filtering by anatomies: {accepted_anatomies_set if use_anatomies_filter else 'None (all included)'}")
    print(f"Filtering by sequences: {accepted_sequences_set if use_sequences_filter else 'None (all included)'}")

    # Use the 6-level glob pattern consistent with your data structure
    glob_pattern = "*/*/*/*/*/*/"

    label_counts_raw = Counter()
    total_dirs_scanned = 0

    for pycurt_dir in pycurt_dirs:
        print(f"Scanning source directory: {pycurt_dir}")

        for sequence_dir_path in pycurt_dir.glob(glob_pattern):
            total_dirs_scanned += 1
            if not sequence_dir_path.is_dir() or "patches" in sequence_dir_path.name:
                continue

            try:
                relative_path_parts = sequence_dir_path.relative_to(pycurt_dir).parts
                if len(relative_path_parts) < 6:
                    print(
                        f"Warning: Path {sequence_dir_path} relative to {pycurt_dir} is too short ({len(relative_path_parts)} parts). Expected 6 levels. Skipping.")
                    continue

                sequence_type_folder_name = relative_path_parts[-2]
                anatomy_name_from_path = relative_path_parts[-3]
                axis_name_from_path = relative_path_parts[-4]

                sequence_name_for_filter = sequence_type_folder_name.split("-")[0]

            except ValueError:
                print(f"Warning: Path {sequence_dir_path} is not a proper descendant of {pycurt_dir}. Skipping.")
                continue
            except IndexError:
                print(f"Warning: Unexpected path structure or indexing error for {sequence_dir_path}. Skipping.")
                continue

            # Apply filters using the extracted names
            if sequence_name_for_filter in ignored_modalities_set: continue
            if anatomy_name_from_path in ignored_modalities_set: continue
            if use_axes_filter and axis_name_from_path not in accepted_axes_set: continue
            if use_anatomies_filter and anatomy_name_from_path not in accepted_anatomies_set: continue
            if use_sequences_filter and sequence_name_for_filter not in accepted_sequences_set: continue
            if "RGB" in anatomy_name_from_path: continue

            # Get the estimated label for counting
            estimated_label_str = _get_estimated_label_from_path(sequence_dir_path, config)
            if estimated_label_str:
                label_counts_raw[estimated_label_str] += 1

    print(f"\n--- Statistics Summary ---")
    print(f"Total directories scanned (before all filters): {total_dirs_scanned}")
    print(f"Total directories matching initial structural filters: {sum(label_counts_raw.values())}")
    print(f"Total unique labels found: {len(label_counts_raw)}")

    if not label_counts_raw:
        print("No labels found after scanning and initial filtering.")
        return

    print(f"\nLabel Distribution (Count of Images/Series per Label):")
    print("-" * 50)
    print(f"{'Label':<30} | {'Count':<10}")
    print("-" * 50)

    sorted_labels = sorted(label_counts_raw.items(), key=lambda item: item[1], reverse=True)
    for label, count in sorted_labels:
        print(f"{label:<30} | {count:<10}")
    print("-" * 50)

    # Optional: Calculate and print some basic stats like min/max/avg
    if len(label_counts_raw) > 0:
        counts = list(label_counts_raw.values())
        print(f"\nAdditional Label Stats:")
        print(f"  Min images per label: {min(counts)}")
        print(f"  Max images per label: {max(counts)}")
        print(f"  Average images per label: {sum(counts) / len(counts):.2f}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()