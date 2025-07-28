import json
import shutil  # Import shutil for moving contents
from pathlib import Path
from argparse import ArgumentParser, Namespace
import sys
import re
from collections import Counter  # Still useful for internal checks or future use
from typing import Union  # For type hinting


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
            else:
                d[k] = v
        return Namespace(**d)

    return dict_to_namespace(config_dict)


def main():
    parser = ArgumentParser(description="Rename specific folders in raw DICOM data based on predefined rules.")
    parser.add_argument("-c", "--config_path", type=str, default="./config/preproc_config_bp.json",
                        help="Path to JSON config file containing pycurt_dirs.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Perform a dry run: print what would be renamed without actually renaming.")
    args = parser.parse_args()

    config = read_json_config_simple(args.config_path)

    pycurt_dirs = []
    if hasattr(config.paths, 'pycurt_dir'):
        pycurt_dirs = [Path(config.paths.pycurt_dir)]
    elif hasattr(config.paths, 'pycurt_dirs'):
        pycurt_dirs = [Path(dir_path) for dir_path in config.paths.pycurt_dirs]
    else:
        print("Error: No 'pycurt_dir' or 'pycurt_dirs' specified in config under 'paths'.", file=sys.stderr)
        sys.exit(1)

    # --- Define Renaming Rules ---
    renaming_rules = {
        "SPINE": "WS",
        "ABD-TS": "TLS",
        "HNCANP": "HNC-ANP",
        "PLV-EG": "PLV-LEG",
        "LUNGANP": "LUNG-ANP",
        "ANPLS": "LS",
        "ABD-TS": "TLS",
        "BS":"TS"
    }

    print(f"\n--- Starting Folder Renaming Utility ---")
    if args.dry_run:
        print("!!! DRY RUN ACTIVE: No files will be renamed. Showing planned changes only. !!!")

    # Use the 6-level glob pattern consistent with your data structure
    glob_pattern = "*/*/*/*/*/*/"  # 6 wildcards for 6 levels of directories

    folders_to_rename_operations = []  # List to collect (old_path_obj, new_path_obj, rel_old_str, rel_new_str)
    total_folders_scanned = 0

    print(f"\n--- Phase 1: Scanning directories to identify renames ---")
    for pycurt_dir_root in pycurt_dirs:
        print(f"Scanning root directory: {pycurt_dir_root}")

        # Iterate over all series_description_folders (the actual DICOM directories)
        for series_description_folder_path in pycurt_dir_root.glob(glob_pattern):
            total_folders_scanned += 1

            if not series_description_folder_path.is_dir() or "patches" in series_description_folder_path.name:
                continue

            try:
                # Extract the anatomy_name, which is the folder we want to check for renaming
                # series_description_folder_path.parent.parent is the Path object for the 'anatomy_name' folder
                anatomy_folder_path = series_description_folder_path.parent.parent
                current_anatomy_name = anatomy_folder_path.name

                if current_anatomy_name in renaming_rules:
                    new_anatomy_name = renaming_rules[current_anatomy_name]
                    new_anatomy_folder_path = anatomy_folder_path.parent / new_anatomy_name

                    # Only add to list if it's a valid rename operation (not renaming to self)
                    if anatomy_folder_path != new_anatomy_folder_path:
                        folders_to_rename_operations.append((
                            anatomy_folder_path,
                            new_anatomy_folder_path,
                            str(anatomy_folder_path.relative_to(pycurt_dir_root)),
                            str(new_anatomy_folder_path.relative_to(pycurt_dir_root))
                        ))

            except Exception as e:
                print(f"Error identifying rename for {series_description_folder_path}: {e}", file=sys.stderr)
                continue

    # Deduplicate rename operations, as multiple series can point to the same anatomy folder.
    # We only want to attempt to rename each unique anatomy folder once.
    unique_renames_map = {}  # Key: old_path_obj, Value: (new_path_obj, rel_old_str, rel_new_str)
    for old_path_obj, new_path_obj, rel_old_str, rel_new_str in folders_to_rename_operations:
        unique_renames_map[old_path_obj] = (new_path_obj, rel_old_str, rel_new_str)

    # Convert back to a list of tuples to iterate
    unique_renames_list = [(k, v[0], v[1], v[2]) for k, v in unique_renames_map.items()]

    print(f"\n--- Phase 2: Performing Renames ({len(unique_renames_list)} unique renames to attempt) ---")
    total_folders_successfully_renamed = 0
    total_folders_merged = 0

    # Sort rename operations from deepest to shallowest.
    # This might help if there's a nested rename (e.g., folder A/B and rule for B)
    # but for your anatomy level, it's mostly about ensuring source exists before move.
    # The primary issue was the glob's internal state, fixed by 2-phase approach.
    # Sorting by `depth` or `string representation` isn't strictly necessary here,
    # as the `rename` or `merge` operation applies to distinct `anatomy_folder_path` objects.
    # We already deduped by `anatomy_folder_path`.

    for old_anatomy_path, new_anatomy_path, rel_old_str, rel_new_str in unique_renames_list:
        if not old_anatomy_path.exists():
            print(
                f"Warning: Original folder '{rel_old_str}' no longer exists (likely already renamed or moved by another operation). Skipping.",
                file=sys.stderr)
            continue

        try:
            if new_anatomy_path.exists():
                # Target folder exists, so merge contents
                print(f"Merging: '{rel_old_str}' into existing '{rel_new_str}'")
                if not args.dry_run:
                    # Move all contents from old_anatomy_path to new_anatomy_path
                    for item in old_anatomy_path.iterdir():
                        shutil.move(str(item), str(new_anatomy_path / item.name))
                    # Remove the now empty old directory
                    old_anatomy_path.rmdir()
                    total_folders_merged += 1
                else:
                    print(
                        f"DRY RUN: Would merge contents of '{rel_old_str}' into '{rel_new_str}' and then delete '{rel_old_str}'.")
                    total_folders_merged += 1  # Count for dry run too
            else:
                # Target folder does not exist, simple rename
                if not args.dry_run:
                    old_anatomy_path.rename(new_anatomy_path)
                    print(f"Renamed: '{rel_old_str}' -> '{rel_new_str}'")
                    total_folders_successfully_renamed += 1
                else:
                    print(f"DRY RUN: Would rename: '{rel_old_str}' -> '{rel_new_str}'")
                    total_folders_successfully_renamed += 1  # Count for dry run too

        except Exception as e:
            print(f"Error processing rename/merge for '{rel_old_str}' to '{rel_new_str}': {e}", file=sys.stderr)
            # If an error occurs during merge, the old folder might be partially moved.
            # Manual inspection will be required.

    print(f"\n--- Renaming Summary ---")
    print(f"Total folders scanned (during identification): {total_folders_scanned}")
    print(f"Total unique rename/merge operations identified: {len(unique_renames_list)}")
    print(f"Total folders successfully renamed (new destination created): {total_folders_successfully_renamed}")
    print(f"Total folders merged (contents moved to existing destination): {total_folders_merged}")

    if args.dry_run:
        print("This was a DRY RUN. No actual changes were made to your file system.")
    else:
        print("Renaming/Merging complete. Please **carefully verify your data** before proceeding with preprocessing.")
        print(
            "Remember to re-run your main preprocessing script (`final_cleaned.py`) after confirming physical changes.")


if __name__ == "__main__":
    main()