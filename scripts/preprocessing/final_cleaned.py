import dicom2nifti
import nibabel as nib
import pydicom
import os
import pickle
import json
import gc
import re
import warnings
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from transformers import AutoTokenizer
from loguru import logger
from typing import Union, List, Tuple, Dict, Set
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from pycurtv2.processing.common import get_n_slice, resize_or_pad_image
import matplotlib.pyplot as plt
from pycurtv2.converters.dicom import DicomConverter
from tqdm import tqdm
import glob
from datetime import datetime
import numpy as np
import uuid  # Added for unique filename generation

# Suppress FutureWarning from external libraries for cleaner console output
warnings.filterwarnings("ignore", category=FutureWarning)


# --- DataClass Definitions (unchanged) ---
@dataclass
class dicomData:
    """
    Class for holding raw DICOM data.
    If 'ALL' orientation processing, img will be Dict[str, np.ndarray].
    If single orientation, img will be np.ndarray.
    """
    img: Union[np.ndarray, Dict[str, np.ndarray]]
    text_attributes: str
    numerical_attributes: List[float]
    label: str


@dataclass
class encodedSample:
    """Class for holding encoded DICOM data after preprocessing for a specific orientation."""
    img: np.ndarray
    input_ids: List[int]
    attention_mask: List[int]
    numerical_attributes: List[float]
    label: int
    orientation: str


# --- read_yaml_config (unchanged) ---
def read_yaml_config(config_path: Union[str, Path]) -> Namespace:
    """Reads a JSON/YAML config file and returns it as an argparse.Namespace object."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
            elif isinstance(v, list):  # Convert lists to tuples if they are to be sets
                # Convert filtering sets to set type for faster lookups later
                if k in ['axes', 'anatomies', 'sequences', 'ignored_modalities', 'filter_patterns']:
                    d[k] = set(v)
                else:
                    d[k] = v
        return Namespace(**d)

    return dict_to_namespace(config_dict)


# --- normalize_class_name (unchanged) ---
def normalize_class_name(label: str) -> str:
    """
    Normalize class names to handle variations in naming conventions.
    This function applies general, algorithmic normalization rules.
    """
    original = label
    if label.startswith("4D") and not label.startswith("4DCCT") and not label.startswith("4DCT"):
        parts = label.split("-")
        new_base = parts[0][2:]
        if len(parts) > 1:
            return f"{new_base}-{'-'.join(parts[1:])}"
        else:
            return new_base
    if "-FS-SE" in label:
        label = label.replace("-FS-SE", "-SE-FS")
    modifiers = ["FS", "GE", "SE", "IRSE", "IRGE", "VFA", "MIP", "CE"]
    if "-" in label:
        base_type = label.split("-")[0]
        rest = label[len(base_type) + 1:]
        if not re.match(r"^[A-Z]+\d+", base_type):
            base_type = base_type.upper()
        if base_type == "DWI" and rest.startswith("b") and not any(x in rest for x in ["ADC", "FA", "TRACE", "EADC"]):
            rest_parts = rest.split("-")
            b_value_str = rest_parts[0][1:]
            modifiers_part = rest_parts[1:] if len(rest_parts) > 1 else []
            try:
                b_value = float(b_value_str)
                if b_value == 0:
                    new_b_value = "b0"
                elif 1 <= b_value <= 75:
                    new_b_value = "b1-75"
                elif 76 <= b_value <= 150:
                    new_b_value = "b76-150"
                elif 151 <= b_value <= 500:
                    new_b_value = "b151-500"
                elif 501 <= b_value <= 850:
                    new_b_value = "b501-850"
                elif 851 <= b_value <= 1050:
                    new_b_value = "b851-1050"
                elif 1051 <= b_value <= 1450:
                    new_b_value = "b1051-1450"
                elif 1451 <= b_value <= 1950:
                    new_b_value = "b1451-1950"
                elif b_value > 1950:
                    new_b_value = "b1951plus"
                else:
                    return original
                found_modifiers = [part for part in modifiers_part if part in modifiers]
                remaining_parts = [part for part in modifiers_part if part not in modifiers]
                found_modifiers.sort()
                if found_modifiers and remaining_parts:
                    return f"DWI-{new_b_value}-{'-'.join(remaining_parts)}-{'-'.join(found_modifiers)}"
                elif found_modifiers:
                    return f"DWI-{new_b_value}-{'-'.join(found_modifiers)}"
                elif remaining_parts:
                    return f"DWI-{new_b_value}-{'-'.join(remaining_parts)}"
                else:
                    return f"DWI-{new_b_value}"
            except ValueError:
                return original
        found_modifiers = [part for part in rest.split("-") if part in modifiers]
        remaining_parts = [part for part in rest.split("-") if part not in modifiers]
        found_modifiers.sort()
        if found_modifiers and remaining_parts:
            label = f"{base_type}-{'-'.join(remaining_parts)}-{'-'.join(found_modifiers)}"
        elif found_modifiers:
            label = f"{base_type}-{'-'.join(found_modifiers)}"
        elif remaining_parts:
            label = f"{base_type}-{'-'.join(remaining_parts)}"
        else:
            label = base_type
    return label


# --- Helper to get label from path for early filtering (unchanged) ---
def _get_estimated_label_from_path(sequence_dir_path: Path, args: Namespace) -> str:
    """Estimates the label string from the path based on classification target."""
    if hasattr(args.classification, 'target'):
        if args.classification.target == 'body_part':
            return sequence_dir_path.parent.parent.name
        elif args.classification.target == 'sequence':
            return sequence_dir_path.parent.name.split("-")[0]
    return sequence_dir_path.parent.name.split("-")[0]


# --- MODIFIED: Function to process and save individual DICOM data ---
def _process_and_save_single_dicom_data(dicom_dir_str: str, args: Namespace, target_orientation: str,
                                        body_part_labeler, output_folder: Path, processed_dirs_log_path: Path) -> Union[
    Path, None]:
    """
    Processes a single DICOM directory to extract an image for a specific orientation,
    assigns the final string label, and saves the dicomData object to a pickle file.
    Appends the processed DICOM directory path to the log file upon successful saving.
    Returns the path to the saved file or None if processing fails.
    """
    dicom_dir = Path(dicom_dir_str)
    nifti_output_path = Path(str(dicom_dir) + ".nii.gz")

    try:
        slices = list(dicom_dir.glob("*"))
        if not slices:
            logger.warning(f"No DICOM slices found in {dicom_dir}. Skipping series.")
            return None

        dicom_slice_path = None
        for s in slices:
            if s.is_file() and s.suffix.lower() in ['.dcm', '']:
                dicom_slice_path = s
                break
        if not dicom_slice_path:
            logger.warning(f"No valid DICOM slice file found in {dicom_dir}. Skipping series.")
            return None

        try:
            DicomConverter(toConvert=str(dicom_dir)).convert_ps()
        except Exception as e:
            logger.warning(f"DICOM conversion failed for {dicom_dir}: {e}. Skipping series.")
            return None

        np_image = None
        try:
            np_image = resize_or_pad_image(
                get_n_slice(str(nifti_output_path), str(dicom_slice_path), desired_orientation=target_orientation))
        except FileNotFoundError:
            logger.warning(
                f"NIfTI file not found after conversion for {dicom_dir} ({target_orientation}). Skipping series.")
        except Exception as e:
            logger.warning(
                f"Failed to load NIfTI or extract slice for {dicom_dir} ({target_orientation}): {e}. Skipping series.")

        if np_image is None or np_image.size == 0:
            logger.warning(f"Final image is empty or None for {dicom_dir} ({target_orientation}). Skipping series.")
            if nifti_output_path.exists():
                try:
                    nifti_output_path.unlink(missing_ok=True)
                except OSError as e_unlink:
                    logger.warning(f"Error cleaning up NIfTI: {e_unlink}")
            return None

        metadata = pydicom.dcmread(str(dicom_slice_path), stop_before_pixels=True, force=True)
        modality_from_metadata = getattr(metadata, 'Modality', 'UNKNOWN_MODALITY')

        text_attributes_list = []
        for attribute in args.text_attributes:
            attr_value = getattr(metadata, attribute, None)
            if attr_value is not None:
                text_attributes_list.append(f"{attribute} is {attr_value}")
            else:
                text_attributes_list.append(args.missing_term)
        text_attribute = ", ".join(text_attributes_list)

        numerical_attributes_list = []
        for attribute in args.numerical_attributes:
            attr_value = getattr(metadata, attribute, None)
            try:
                numerical_attributes_list.append(int(attr_value) if attr_value is not None else -1)
            except (ValueError, TypeError):
                numerical_attributes_list.append(-1)

        final_series_label = ""
        if hasattr(args.classification, 'target'):
            if args.classification.target == 'body_part':
                if args.classification.custom_labeler and body_part_labeler:
                    try:
                        first_slice_for_labeling = next(
                            (str(f) for f in dicom_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.dcm', '']),
                            None)
                        if first_slice_for_labeling:
                            final_series_label = body_part_labeler.get_bodypart(first_slice_for_labeling)
                        else:
                            final_series_label = "NS"
                    except Exception as e:
                        logger.warning(f"Error labeling body part for {dicom_dir}: {e}. Labeling as NS.")
                        final_series_label = "NS"
                else:
                    final_series_label = dicom_dir.parent.parent.name
            elif args.classification.target == 'sequence':
                final_series_label = dicom_dir.parent.name.split("-")[0]
        else:
            final_series_label = dicom_dir.parent.name.split("-")[0]

        dicom_data = dicomData(
            img=np_image,
            text_attributes=text_attribute,
            numerical_attributes=numerical_attributes_list,
            label=final_series_label
        )

        del np_image, text_attributes_list, numerical_attributes_list, metadata
        gc.collect()

        if nifti_output_path.exists():
            try:
                nifti_output_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"[WARNING] Error removing NIfTI file {nifti_output_path}: {e}")

        # Save the dicomData object to a unique pickle file
        safe_dir_name = dicom_dir.name.replace(os.sep, '_').replace(' ', '_')
        unique_filename = f"{safe_dir_name}_{target_orientation}_{uuid.uuid4().hex}.pkl"
        output_pkl_path = output_folder / unique_filename

        with open(output_pkl_path, 'wb') as f:
            pickle.dump(dicom_data, f)

        # Append the successfully processed DICOM directory to the log file
        with open(processed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")

        logger.info(
            f"Processed Series: {dicom_dir.name} | Orientation: {target_orientation} | Modality: {modality_from_metadata} | Body Part: {final_series_label} -> Saved to {output_pkl_path}")

        return output_pkl_path

    except Exception as e:
        logger.error(f"[ERROR] An unhandled exception occurred processing {dicom_dir} for {target_orientation}: {e}",
                     exc_info=True)
        return None


# --- MODIFIED: Function to process and save ALL orientations (single dicomData with dict of images) ---
def _process_and_save_all_dicom_data(dicom_dir_str: str, args: Namespace, body_part_labeler,
                                     output_folder: Path, processed_dirs_log_path: Path) -> Union[Path, None]:
    """
    Processes a single DICOM directory to extract images for ALL specified orientations,
    and saves a single dicomData object (with img as a dict) to a pickle file.
    Appends the processed DICOM directory path to the log file upon successful saving.
    Returns the path to the saved file or None if processing fails.
    """
    dicom_dir = Path(dicom_dir_str)
    nifti_output_path = Path(str(dicom_dir) + ".nii.gz")

    try:
        slices = list(dicom_dir.glob("*"))
        if not slices:
            logger.warning(f"No DICOM slices found in {dicom_dir}. Skipping series.")
            return None

        dicom_slice_path = None
        for s in slices:
            if s.is_file() and s.suffix.lower() in ['.dcm', '']:
                dicom_slice_path = s
                break
        if not dicom_slice_path:
            logger.warning(f"No valid DICOM slice file found in {dicom_dir}. Skipping series.")
            return None

        try:
            DicomConverter(toConvert=str(dicom_dir)).convert_ps()
        except Exception as e:
            logger.warning(f"DICOM conversion failed for {dicom_dir}: {e}. Skipping series.")
            return None

        metadata = pydicom.dcmread(str(dicom_slice_path), stop_before_pixels=True, force=True)
        modality_from_metadata = getattr(metadata, 'Modality', 'UNKNOWN_MODALITY')

        text_attributes_list = []
        for attribute in args.text_attributes:
            attr_value = getattr(metadata, attribute, None)
            if attr_value is not None:
                text_attributes_list.append(f"{attribute} is {attr_value}")
            else:
                text_attributes_list.append(args.missing_term)
        text_attribute = ", ".join(text_attributes_list)

        numerical_attributes_list = []
        for attribute in args.numerical_attributes:
            attr_value = getattr(metadata, attribute, None)
            try:
                numerical_attributes_list.append(int(attr_value) if attr_value is not None else -1)
            except (ValueError, TypeError):
                numerical_attributes_list.append(-1)

        final_series_label = ""
        if hasattr(args.classification, 'target'):
            if args.classification.target == 'body_part':
                if args.classification.custom_labeler and body_part_labeler:
                    try:
                        first_slice_for_labeling = next(
                            (str(f) for f in dicom_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.dcm', '']),
                            None)
                        if first_slice_for_labeling:
                            final_series_label = body_part_labeler.get_bodypart(first_slice_for_labeling)
                        else:
                            final_series_label = "NS"
                    except Exception as e:
                        logger.warning(f"Error labeling body part for {dicom_dir}: {e}. Labeling as NS.")
                        final_series_label = "NS"
                else:
                    final_series_label = dicom_dir.parent.parent.name
            elif args.classification.target == 'sequence':
                final_series_label = dicom_dir.parent.name.split("-")[0]
        else:
            final_series_label = dicom_dir.parent.name.split("-")[0]

        images_per_orientation = {}
        target_orientations_to_extract = ["TRA", "COR", "SAG"]

        # Ensure the output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)

        for orient in target_orientations_to_extract:
            processed_img = None
            try:
                slice_img = get_n_slice(str(nifti_output_path), str(dicom_slice_path), desired_orientation=orient)
                processed_img = resize_or_pad_image(slice_img)
                if processed_img is None or processed_img.size == 0:
                    logger.warning(
                        f"Extracted image is empty or None for {dicom_dir} ({orient}). Skipping this orientation for this series.")
                    continue
                images_per_orientation[orient] = processed_img
            except FileNotFoundError:
                logger.warning(
                    f"NIfTI file not found after conversion for {dicom_dir} ({orient}). Skipping this orientation.")
            except Exception as e:
                logger.warning(
                    f"Could not extract {orient} slice from {dicom_dir}: {e}. Skipping this orientation for this series.")

        if not images_per_orientation:
            logger.error(f"No valid images could be extracted for any orientation from {dicom_dir}. Skipping series.")
            if nifti_output_path.exists():
                try:
                    nifti_output_path.unlink(missing_ok=True)
                except OSError as e_unlink:
                    logger.warning(f"Error cleaning up NIfTI: {e_unlink}")
            return None

        # Create a single dicomData object with the dictionary of images
        dicom_data = dicomData(
            img=images_per_orientation,  # This is the key change: img is Dict[str, np.ndarray]
            text_attributes=text_attribute,
            numerical_attributes=numerical_attributes_list,
            label=final_series_label
        )

        del metadata, text_attributes_list, numerical_attributes_list
        gc.collect()

        if nifti_output_path.exists():
            try:
                nifti_output_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"[WARNING] Error removing NIfTI file {nifti_output_path}: {e}")

        # Save the dicomData object to a unique pickle file
        safe_dir_name = dicom_dir.name.replace(os.sep, '_').replace(' ', '_')
        # Use UUID to ensure absolute uniqueness for the *series* (not per orientation anymore)
        unique_filename = f"{safe_dir_name}_ALL_ORIENTATIONS_{uuid.uuid4().hex}.pkl"
        output_pkl_path = output_folder / unique_filename

        with open(output_pkl_path, 'wb') as f:
            pickle.dump(dicom_data, f)

        # Append the successfully processed DICOM directory to the log file
        with open(processed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")

        logger.info(
            f"Processed Series: {dicom_dir.name} | Modality: {modality_from_metadata} | Body Part: {final_series_label} | Orientations Extracted: {list(images_per_orientation.keys())} -> Saved to {output_pkl_path}")

        return output_pkl_path  # Return the path to the single saved file for the series

    except Exception as e:
        logger.error(f"[ERROR] An unhandled exception occurred processing {dicom_dir}: {e}", exc_info=True)
        return None


def fix_sequence_lengths(input_ids_list: List[List[int]], attention_mask_list: List[List[int]],
                         max_length: int = 512, pad_token_id: int = 0) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Fix sequence lengths to ensure all sequences have the same length.
    """
    logger.info(f"Fixing sequence lengths to {max_length} tokens...")

    lengths = [len(seq) for seq in input_ids_list]
    current_max = max(lengths)
    current_min = min(lengths)

    logger.info(f"Current sequence lengths: min={current_min}, max={current_max}")

    if max_length is None:
        sorted_lengths = sorted(lengths)
        percentile_95_idx = int(len(sorted_lengths) * 0.95)
        max_length = sorted_lengths[percentile_95_idx]
        logger.info(f"Auto-selected max_length={max_length} (95th percentile)")

    fixed_input_ids = []
    fixed_attention_masks = []
    truncated_count = 0
    padded_count = 0
    unchanged_count = 0

    for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        original_length = len(input_ids)

        fixed_input_ids_seq = list(input_ids)
        fixed_attention_mask_seq = list(attention_mask)

        if original_length > max_length:
            fixed_input_ids_seq = fixed_input_ids_seq[:max_length]
            fixed_attention_mask_seq = fixed_attention_mask_seq[:max_length]
            truncated_count += 1
        elif original_length < max_length:
            pad_length = max_length - original_length
            fixed_input_ids_seq.extend([pad_token_id] * pad_length)
            fixed_attention_mask_seq.extend([0] * pad_length)
            padded_count += 1
        else:
            unchanged_count += 1

        assert len(fixed_input_ids_seq) == max_length, f"Length mismatch at sample {i}"
        assert len(fixed_attention_mask_seq) == max_length, f"Attention mask length mismatch at sample {i}"

        fixed_input_ids.append(fixed_input_ids_seq)
        fixed_attention_masks.append(fixed_attention_mask_seq)

    logger.info(f"Sequence length fixing complete:")
    logger.info(f"  - Truncated: {truncated_count} sequences")
    logger.info(f"  - Padded: {padded_count} sequences")
    logger.info(f"  - Unchanged: {unchanged_count} sequences")
    logger.info(f"  - All sequences now have length: {max_length}")

    return fixed_input_ids, fixed_attention_masks


# --- New function to save sample images (modified to use label_dict for folder name) ---
def save_sample_images(samples: List[dicomData], output_base_dir: Path, current_orientation: str,
                       label_mapping_dict: Dict[str, int], max_images_per_label: int = 5):
    """
    Saves a maximum of `max_images_per_label` sample images as PNGs for each label and orientation.
    Also saves text attributes alongside the images.

    Args:
        samples: A list of dicomData objects (already filtered and cleaned).
        output_base_dir: The root output directory (e.g., /media/e210/portable_hdd/d_bodypart_2).
        current_orientation: The orientation being processed (e.g., "TRA", "COR").
        label_mapping_dict: The dictionary mapping string labels to integer IDs for this orientation.
        max_images_per_label: Maximum number of sample images to save per label.
    """
    observation_check_output_root = output_base_dir / "observation_check" / current_orientation
    observation_check_output_root.mkdir(parents=True, exist_ok=True)

    saved_counts_per_label = Counter()

    logger.info(
        f"Saving up to {max_images_per_label} sample images and text attributes per class for orientation '{current_orientation}' to {observation_check_output_root}...")

    random.shuffle(samples)  # Shuffle samples for a random selection

    for i, sample in enumerate(samples):
        if saved_counts_per_label[sample.label] < max_images_per_label:
            # Get integer ID and combine with safe label name
            label_int_id = label_mapping_dict.get(sample.label, 'UNKNOWN_ID')
            label_safe_name = sample.label.replace("/", "_").replace("\\", "_").replace(":",
                                                                                        "_")  # Sanitize string label

            # Construct folder name as "LABEL_STRING_ID"
            folder_name = f"{label_safe_name}_{label_int_id}"
            label_output_dir = observation_check_output_root / folder_name
            label_output_dir.mkdir(parents=True, exist_ok=True)

            # Filenames for image and text attribute
            base_filename = f"{label_safe_name}_{saved_counts_per_label[sample.label]}"
            image_output_path = label_output_dir / f"{base_filename}.png"
            text_output_path = label_output_dir / f"{base_filename}.txt"

            try:
                img_data = sample.img
                # Handle cases where img might be a dict for 'ALL' orientations, pick one
                if isinstance(img_data, dict):
                    # Prefer TRA, then COR, then SAG for sample image if multiple are available
                    if "TRA" in img_data:
                        img_data = img_data["TRA"]
                    elif "COR" in img_data:
                        img_data = img_data["COR"]
                    elif "SAG" in img_data:
                        img_data = img_data["SAG"]
                    else:
                        # If no standard orientation, pick the first one
                        img_data = next(iter(img_data.values()))

                if img_data.ndim > 2:
                    img_data = img_data[:, :, img_data.shape[2] // 2]  # Take middle slice for 3D

                data_min = img_data.min()
                data_max = img_data.max()
                if data_max > data_min:
                    img_data = (img_data - data_min) / (data_max - data_min)
                else:
                    img_data = np.zeros_like(img_data, dtype=np.uint8)

                img_data = (img_data * 255).astype(np.uint8)

                plt.imsave(image_output_path, img_data, cmap='gray')

                # Save Text Attributes
                with open(text_output_path, 'w') as f:
                    f.write(sample.text_attributes)

                saved_counts_per_label[sample.label] += 1
            except Exception as e:
                # Corrected variable name from output_path.parent to label_output_dir
                logger.warning(f"Failed to save sample for label '{sample.label}' ({label_output_dir}): {e}")

        all_labels_covered = True
        for unique_label in set(s.label for s in samples):
            if saved_counts_per_label[unique_label] < max_images_per_label:
                all_labels_covered = False
                break

        if all_labels_covered:
            break

    logger.info(f"Finished saving sample images and text attributes for orientation '{current_orientation}'.")
    logger.info(f"Sample images and text attributes saved per label: {dict(saved_counts_per_label)}")


def get_dataset(args: Namespace) -> None:
    """
    Process DICOM directories to create a dataset for body part or sequence classification.
    Supports 'ALL' orientations or a single specified orientation.
    Applies a maximum sample cap per label.
    Includes a test mode to process only a limited number of directories.
    Integrates label cleaning and normalization steps, including early filtering by min_class_size.
    """
    logger.info("Starting dataset creation...")

    pycurt_dicom_dirs = []
    if hasattr(args.paths, 'pycurt_dir'):
        pycurt_dicom_dirs = [Path(args.paths.pycurt_dir)]
    elif hasattr(args.paths, 'pycurt_dirs'):
        pycurt_dicom_dirs = [Path(dir_path) for dir_path in args.paths.pycurt_dirs]
    else:
        logger.error("No pycurt directory specified in config.")
        return

    dataset_save_dir = Path(args.paths.dataset_dir)
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    # --- NEW: Create directory for individual processed samples ---
    individual_samples_dir = dataset_save_dir / "individual_processed_samples"
    individual_samples_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Individual processed samples will be saved to: {individual_samples_dir}")

    # --- NEW: Processed Directories Tracking File ---
    processed_dirs_log_path = dataset_save_dir / "processed_dicom_dirs.csv"
    already_processed_dirs = set()
    if processed_dirs_log_path.exists():
        with open(processed_dirs_log_path, 'r') as f:
            for line in f:
                already_processed_dirs.add(line.strip())
        logger.info(
            f"Loaded {len(already_processed_dirs)} previously processed DICOM directories from {processed_dirs_log_path}")
    else:
        logger.info(f"No existing processed DICOM directories log found at {processed_dirs_log_path}.")

    test_mode_max_dirs = getattr(args.processing_params, 'test_mode_max_dirs', -1)
    if test_mode_max_dirs > 0:
        logger.warning(
            f"!!! TEST MODE ACTIVE: Limiting processing to first {test_mode_max_dirs} DICOM directories. !!!")
    min_class_size = getattr(args.processing_params, 'min_class_size', 20)

    process_all_orientations = (hasattr(args, 'orientation') and args.orientation.upper() == "ALL")
    single_target_orientation = None
    if not process_all_orientations:
        if hasattr(args, 'orientation') and args.orientation:
            single_target_orientation = args.orientation.upper()
            logger.info(f"Processing single orientation: {single_target_orientation}")
        else:
            logger.warning("No specific 'orientation' or 'ALL' specified in config. Defaulting to 'TRA'.")
            single_target_orientation = "TRA"

    ignored_modalities_set = set(args.ignored_modalities)

    use_axes_filter = False
    accepted_axes_set = None
    if hasattr(args.filtering, 'axes') and args.filtering.axes:
        accepted_axes_set = set(args.filtering.axes)
        use_axes_filter = True
    elif hasattr(args, 'accepted_axes') and args.accepted_axes:
        accepted_axes_set = set(args.accepted_axes)
        use_axes_filter = True

    use_anatomies_filter = False
    accepted_anatomies_set = None
    if hasattr(args.filtering, 'anatomies') and args.filtering.anatomies:
        accepted_anatomies_set = set(args.filtering.anatomies)
        use_anatomies_filter = True
    elif hasattr(args, 'accepted_anatomies') and args.accepted_anatomies:
        accepted_anatomies_set = set(args.accepted_anatomies)
        use_anatomies_filter = True

    use_sequences_filter = False
    accepted_sequences_set = None
    if hasattr(args.filtering, 'sequences') and args.filtering.sequences:
        accepted_sequences_set = set(args.filtering.sequences)
        use_sequences_filter = True
    elif hasattr(args, 'accepted_sequences') and args.accepted_sequences:
        accepted_sequences_set = set(args.accepted_sequences)
        use_sequences_filter = True

    logger.info(f"Ignoring modalities: {ignored_modalities_set}")
    if use_axes_filter:
        logger.info(f"Filtering by axes: {accepted_axes_set}")
    else:
        logger.info("Including all axes")
    if use_anatomies_filter:
        logger.info(f"Filtering by anatomies: {accepted_anatomies_set}")
    else:
        logger.info("Including all anatomies")
    if use_sequences_filter:
        logger.info(f"Filtering by sequences: {accepted_sequences_set}")
    else:
        logger.info("Including all sequences")

    body_part_labeler = None
    if (hasattr(args.classification, 'target') and
            args.classification.target == 'body_part' and
            hasattr(args.classification, 'custom_labeler') and
            args.classification.custom_labeler):
        from paste import BPLabeler
        body_part_labeler = BPLabeler()
        logger.info("Using custom BPLabeler for body part classification")

    # --- Initial Scan using glob for speed ---
    glob_pattern = "*/*/*/*/*/*/"  # 6 wildcards for 6 levels of directories (patient/study/axis/anatomy/seq_type/series_desc)

    all_raw_dicom_dirs_paths_and_estimated_labels = []

    logger.info("\n--- Scanning directories with glob and applying initial filters ---")
    for pycurt_dir in pycurt_dicom_dirs:
        logger.info(f"Scanning source directory: {pycurt_dir}")
        found_dirs_in_current_root = 0

        for sequence_dir_path in pycurt_dir.glob(glob_pattern):
            if not sequence_dir_path.is_dir() or "patches" in sequence_dir_path.name:
                continue

            try:
                relative_path_parts = sequence_dir_path.relative_to(pycurt_dir).parts
                if len(relative_path_parts) < 6:  # patient_id/study_id/axis_name/anatomy_name/sequence_type_folder/series_description_folder
                    logger.warning(
                        f"Path {sequence_dir_path} relative to {pycurt_dir} is too short ({len(relative_path_parts)} parts). Expected 6 levels. Skipping.")
                    continue

                sequence_type_folder_name = relative_path_parts[-2]
                anatomy_name_from_path = relative_path_parts[-3]
                axis_name_from_path = relative_path_parts[-4]

                sequence_name_for_filter = sequence_type_folder_name.split("-")[0]

            except ValueError:
                logger.warning(f"Path {sequence_dir_path} is not a proper descendant of {pycurt_dir}. Skipping.")
                continue
            except IndexError:
                logger.warning(f"Unexpected path structure or indexing error for {sequence_dir_path}. Skipping.")
                continue

            # Apply filters using the correctly extracted names
            if sequence_name_for_filter in ignored_modalities_set: continue
            if anatomy_name_from_path in ignored_modalities_set: continue
            if use_axes_filter and axis_name_from_path not in accepted_axes_set: continue
            if use_anatomies_filter and anatomy_name_from_path not in accepted_anatomies_set: continue
            if use_sequences_filter and sequence_name_for_filter not in accepted_sequences_set: continue
            if "RGB" in anatomy_name_from_path: continue

            estimated_label_str = _get_estimated_label_from_path(sequence_dir_path, args)

            if estimated_label_str:
                all_raw_dicom_dirs_paths_and_estimated_labels.append((sequence_dir_path, estimated_label_str))
                found_dirs_in_current_root += 1

        logger.info(f"Found {found_dirs_in_current_root} directories in {pycurt_dir} matching initial filters.")

    logger.info(
        f"Total DICOM directories found after initial glob and filters: {len(all_raw_dicom_dirs_paths_and_estimated_labels)}")

    # --- Apply test mode limit here, BEFORE class size counting ---
    if test_mode_max_dirs > 0:
        original_total_dirs_before_test_mode = len(all_raw_dicom_dirs_paths_and_estimated_labels)
        all_raw_dicom_dirs_paths_and_estimated_labels = all_raw_dicom_dirs_paths_and_estimated_labels[
                                                        :test_mode_max_dirs]
        logger.warning(
            f"Test mode applied: Reduced from {original_total_dirs_before_test_mode} to {len(all_raw_dicom_dirs_paths_and_estimated_labels)} directories.")

    if not all_raw_dicom_dirs_paths_and_estimated_labels:
        logger.error("No DICOM directories left after initial filters or test mode limit. Exiting.")
        return

    # --- Build counts for early min_class_size filtering from the (potentially truncated) list ---
    temp_label_counts_for_early_filter = Counter(item[1] for item in all_raw_dicom_dirs_paths_and_estimated_labels)
    dicom_dir_path_to_estimated_label = {str(item[0]): item[1] for item in
                                         all_raw_dicom_dirs_paths_and_estimated_labels}

    logger.info("Estimated class distribution before early class size filtering:")
    for label, count in sorted(temp_label_counts_for_early_filter.items()):
        logger.info(f"  {label}: {count} samples")

    # --- Early Filtering by min_class_size ---
    dicom_dirs_to_process = []
    classes_removed_by_early_filter = set()
    total_samples_removed_early = 0

    logger.info(f"\n--- Applying early filter: classes with fewer than {min_class_size} samples removed ---")
    for dir_path_obj, estimated_label in all_raw_dicom_dirs_paths_and_estimated_labels:
        # Check against already processed dirs here
        if str(dir_path_obj) in already_processed_dirs:
            logger.debug(f"Skipping already processed directory: {dir_path_obj}")
            continue

        if temp_label_counts_for_early_filter[estimated_label] >= min_class_size:
            dicom_dirs_to_process.append(str(dir_path_obj))
        else:
            classes_removed_by_early_filter.add(estimated_label)
            total_samples_removed_early += 1

    if classes_removed_by_early_filter:
        for cls in sorted(list(classes_removed_by_early_filter)):
            logger.info(f"  Removed class '{cls}' with {temp_label_counts_for_early_filter[cls]} samples.")
        logger.info(f"Total samples removed by early class size filter: {total_samples_removed_early}")
    else:
        logger.info("No classes found below minimum size. No samples removed by early filter.")

    logger.info(
        f"Final {len(dicom_dirs_to_process)} directories selected for full processing (excluding already processed).")

    if not dicom_dirs_to_process:
        logger.info(
            "No new DICOM directories left to process after all filtering steps and skipping processed ones. Exiting.")
        return  # Exit gracefully if nothing new to process

    # --- Initial Label Dictionary Creation (based on classes that passed all early filters) ---
    potential_labels_after_early_filter: Set[str] = set()
    for dir_path_str in dicom_dirs_to_process:
        label = dicom_dir_path_to_estimated_label.get(dir_path_str)
        if label:
            potential_labels_after_early_filter.add(label)

    sorted_unique_labels = sorted(list(potential_labels_after_early_filter))
    initial_label_dict = {label_name: i for i, label_name in enumerate(sorted(sorted_unique_labels))}

    initial_label_dict_path = dataset_save_dir / "label_dict_initial.json"
    with open(initial_label_dict_path, "w") as f:
        json.dump(initial_label_dict, f, indent=4)
    logger.info(f"Initial label dictionary (after early filtering) saved to: {initial_label_dict_path}")
    logger.info("\n--- Classes selected for full processing (Initial `label_dict_initial.json`) ---")
    for label_name, label_id in initial_label_dict.items():
        logger.info(f"  ID: {label_id:<5} | Class: {label_name}")
    logger.info("--------------------------------------------------------------------------------\n")

    # --- PHASE 1: Parallel DICOM Processing and Saving Individual Files ---
    logger.info("Starting parallel DICOM processing to extract data and save individual files to disk...")

    # Choose the appropriate processing function based on 'ALL' vs single orientation
    process_func_for_saving = _process_and_save_all_dicom_data if process_all_orientations else _process_and_save_single_dicom_data

    # List to store paths to the saved individual dicomData pickle files
    newly_saved_dicom_data_paths: List[Path] = []  # Only collect newly processed files
    if args.processing_params.workers == "max":
        w =os.cpu_count() - 1
    else:
        w = args.processing_params.workers
    with ProcessPoolExecutor(max_workers=w) as executor:
        futures = []
        if process_all_orientations:
            futures = [executor.submit(process_func_for_saving, d_str, args, body_part_labeler, individual_samples_dir,
                                       processed_dirs_log_path)
                       for d_str in dicom_dirs_to_process]
        else:
            futures = [
                executor.submit(process_func_for_saving, d_str, args, single_target_orientation, body_part_labeler,
                                individual_samples_dir, processed_dirs_log_path) for d_str in dicom_dirs_to_process]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting & Saving Individual DICOM Data",
                      ncols=100):
            result_path = f.result()  # result_path will be a Path object (or None)
            if result_path is not None:
                newly_saved_dicom_data_paths.append(result_path)

    if not newly_saved_dicom_data_paths:
        logger.warning(
            "No *new* valid DICOM data extracted or saved to individual files in this run. Exiting if no old data to process.")
        # If there were already processed files, we might still proceed.
        # But if `all_saved_dicom_data_paths` (see below) is empty, then exit.

    logger.info(
        f"Successfully processed and saved {len(newly_saved_dicom_data_paths)} new individual DICOM data samples.")
    gc.collect()  # Clear memory after initial processing phase

    # --- PHASE 2: Load All Individual Files (new + old), Apply Label Cleaning/Normalization, and Create Batches ---
    logger.info(
        "\n--- Loading ALL individual processed samples (new and existing) for final cleaning, bootstrapping, and batching ---")

    # Collect all existing individual files, including those from previous runs if any.
    all_individual_pkl_files = list(individual_samples_dir.glob("*.pkl"))
    if not all_individual_pkl_files:
        logger.error("No individual processed sample files found (new or old). Cannot proceed with final steps.")
        return

    raw_dicom_data_for_postprocessing: List[dicomData] = []
    failed_to_load_count = 0
    for pkl_path in tqdm(all_individual_pkl_files, desc="Loading individual samples from disk", ncols=100):
        try:
            with open(pkl_path, 'rb') as f:
                data_item = pickle.load(f)
                raw_dicom_data_for_postprocessing.append(data_item)
            # We don't delete here; deletion happens at the very end of a full successful run.
        except Exception as e:
            logger.warning(f"Could not load individual sample file {pkl_path}: {e}. Skipping.")
            failed_to_load_count += 1
            continue

    if failed_to_load_count > 0:
        logger.warning(f"Skipped {failed_to_load_count} individual samples due to loading errors.")

    if not raw_dicom_data_for_postprocessing:
        logger.error("No samples left after loading individual processed files. Cannot proceed with final steps.")
        return

    # --- Start of Label Cleaning/Normalization Logic from PostProcessing_Script (UNCHANGED) ---
    raw_dicom_data_with_string_labels = raw_dicom_data_for_postprocessing

    original_class_counts_raw = Counter(d.label for d in raw_dicom_data_with_string_labels)

    manual_remapping_rules = {
        "PLV-EG": "PLV-LEG",
        "SPINE": "WS",
        "ANPLS": "LS"
    }
    labels_to_remove_completely = {"HNCANP", "NS"}

    logger.info("\n" + "=" * 60)
    logger.info("APPLYING SPECIFIC LABEL MERGES AND DELETIONS (Stage 0)")
    logger.info("=" * 60)

    pre_normalized_data_after_manual_fix = []
    deleted_sample_count = 0
    remapped_summary = Counter()

    for data_item in tqdm(raw_dicom_data_with_string_labels, desc="Applying manual label adjustments"):
        current_label_str = data_item.label

        if current_label_str in labels_to_remove_completely:
            deleted_sample_count += 1
            continue

        if current_label_str in manual_remapping_rules:
            new_label_str = manual_remapping_rules[current_label_str]
            if new_label_str != current_label_str:
                remapped_summary[f"'{current_label_str}' -> '{new_label_str}'"] += 1
            current_label_str = new_label_str

        data_item.label = current_label_str
        pre_normalized_data_after_manual_fix.append(data_item)

    logger.info(f"Total samples before manual adjustments: {len(raw_dicom_data_with_string_labels)}")
    logger.info(f"Samples deleted due to explicit removal: {deleted_sample_count}")
    logger.info("Specific label remappings performed:")
    if remapped_summary:
        for k, v in remapped_summary.items():
            logger.info(f"  - {k}: {v} samples")
    else:
        logger.info("  - No specific remappings were applied.")

    if not pre_normalized_data_after_manual_fix:
        logger.error("[CRITICAL ERROR] No samples left after manual merges/deletions. Cannot proceed.")
        return

    # Pattern filtering (Step 1)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: FILTERING OUT SPECIFIED PATTERNS")
    logger.info("=" * 60)

    filter_patterns = getattr(args.processing_params, 'filter_patterns', [])

    pattern_filtered_classes = []
    current_unique_labels_after_manual = set(d.label for d in pre_normalized_data_after_manual_fix)
    for pattern in filter_patterns:
        matching_classes = [cls_name for cls_name in current_unique_labels_after_manual if pattern in cls_name]
        if matching_classes:
            pattern_filtered_classes.extend(matching_classes)
            logger.info(f"Found {len(matching_classes)} classes matching pattern '{pattern}'")

    pattern_filtered_classes = list(set(pattern_filtered_classes))

    logger.info(f"Filtering out {len(pattern_filtered_classes)} classes matching specified patterns:")
    if pattern_filtered_classes:
        for cls in pattern_filtered_classes[:10]:
            logger.info(f"  - {cls}")
        if len(pattern_filtered_classes) > 10:
            logger.info(f"  - ... and {len(pattern_filtered_classes) - 10} more")
    else:
        logger.info("  - No classes matched specified patterns for filtering.")

    data_after_pattern_filtering = []
    for data_item in pre_normalized_data_after_manual_fix:
        if data_item.label not in pattern_filtered_classes:
            data_after_pattern_filtering.append(data_item)

    logger.info(
        f"Removed {len(pre_normalized_data_after_manual_fix) - len(data_after_pattern_filtering)} samples from pattern-filtered classes")

    counts_after_pattern_filtering = Counter(d.label for d in data_after_pattern_filtering)

    # General normalization (Step 2)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: NORMALIZING CLASS NAMES (General Rules)")
    logger.info("=" * 60)

    if not data_after_pattern_filtering:
        logger.info("No samples left after pattern filtering to normalize.")
        normalized_data = []
        normalized_counts = Counter()
        normalization_mapping = {}
    else:
        remaining_class_string_names = set(d.label for d in data_after_pattern_filtering)
        normalized_labels_map = {name: normalize_class_name(name) for name in remaining_class_string_names}

        normalized_data = []
        normalized_counts = Counter()
        for data_item in tqdm(data_after_pattern_filtering, desc="Applying general normalization"):
            new_normalized_label_str = normalized_labels_map[data_item.label]
            data_item.label = new_normalized_label_str
            normalized_data.append(data_item)
            normalized_counts[new_normalized_label_str] += 1

        logger.info(f"Classes after pattern filtering (before general norm): {len(counts_after_pattern_filtering)}")
        logger.info(f"Classes after general normalization: {len(normalized_counts)}")
        logger.info(
            f"Reduced by: {len(counts_after_pattern_filtering) - len(normalized_counts)} classes through general normalization")

        normalization_mapping = {}
        for original_name, norm_name in normalized_labels_map.items():
            if norm_name not in normalization_mapping:
                normalization_mapping[norm_name] = []
            normalization_mapping[norm_name].append(original_name)

        logger.info("\nExamples of general normalization:")
        merged_examples = [(norm, originals) for norm, originals in normalization_mapping.items() if len(originals) > 1]
        merged_examples.sort(key=lambda x: len(x[1]), reverse=True)
        if merged_examples:
            for normalized, originals in merged_examples[:10]:
                logger.info(f"  - {normalized}: merged from {len(originals)} classes")
                orig_examples = originals[:3]
                if len(originals) > 3:
                    orig_examples.append("...")
                logger.info(f"    Examples: {orig_examples}")
        else:
            logger.info("  - No classes were merged through general normalization.")

    # Minimum class size filtering (Step 3 - Final one before capping for bootstrapping)
    logger.info(f"\n" + "=" * 60)
    logger.info(f"STEP 3: FILTERING OUT CLASSES WITH FEWER THAN {min_class_size} SAMPLES (Post-Normalization)")
    logger.info("=" * 60)

    small_classes_names_str_post_norm = [cls for cls, count in normalized_counts.items() if count < min_class_size]

    logger.info(f"Filtering out {len(small_classes_names_str_post_norm)} small classes:")
    if small_classes_names_str_post_norm:
        for cls in small_classes_names_str_post_norm[:20]:
            logger.info(f"  - {cls} ({normalized_counts[cls]} samples)")
        if len(small_classes_names_str_post_norm) > 20:
            logger.info(f"  - ... and {len(small_classes_names_str_post_norm) - 20} more")
    else:
        logger.info("  - No classes below minimum size found.")

    if not normalized_data:
        logger.info("No samples left after general normalization for final size filter.")
        data_after_final_size_filtering = []
        final_counts_before_bootstrap = Counter()
    else:
        data_after_final_size_filtering = [data_item for data_item in normalized_data if
                                           data_item.label not in small_classes_names_str_post_norm]

        logger.info(
            f"Removed {len(normalized_data) - len(data_after_pattern_filtering)} samples by final size filter.")

        if not data_after_final_size_filtering:
            logger.info("No samples left after final size filtering.")
            final_counts_before_bootstrap = Counter()
        else:
            final_counts_before_bootstrap = Counter(d.label for d in data_after_final_size_filtering)

    # --- End of Label Cleaning/Normalization Logic ---

    # --- Conditional Processing Branch (ALL vs Single Orientation) ---
    if process_all_orientations:
        logger.info("Configured to process ALL orientations (TRA/COR/SAG using `get_n_slice`).")

        final_cleaned_data_per_orientation: Dict[str, List[dicomData]] = {
            orient: [] for orient in ["TRA", "COR", "SAG"]
        }

        # The `data_after_final_size_filtering` now correctly contains dicomData objects
        # where `img` is a dictionary of orientations if it came from `_process_and_save_all_dicom_data`.
        for data_item_series in data_after_final_size_filtering:
            if isinstance(data_item_series.img, dict) and data_item_series.img:
                for orient_extracted, img_data in data_item_series.img.items():
                    if orient_extracted in ["TRA", "COR", "SAG"]:
                        final_cleaned_data_per_orientation[orient_extracted].append(
                            dicomData(
                                img=img_data,  # This is already the single orientation image
                                text_attributes=data_item_series.text_attributes,
                                numerical_attributes=data_item_series.numerical_attributes,
                                label=data_item_series.label
                            )
                        )
            else:
                logger.warning(
                    f"Unexpected img type or empty image dict for series after cleaning: {type(data_item_series.img)}. Skipping series. (This implies an issue in _process_and_save_all_dicom_data or inconsistent data)")

        # Now proceed with per-orientation capping, tokenization, and saving
        tokenizer = AutoTokenizer.from_pretrained(args.training_params.tokenizer_name)
        max_length = getattr(args.training_params, 'max_sequence_length', None)
        do_bootstrap = getattr(args.processing_params, 'do_bootstrap', True)
        min_samples = getattr(args.processing_params, 'min_samples', 50)
        samples_per_batch_file = getattr(args.processing_params, 'samples_per_batch_file', 1500)
        CAP_PER_LABEL_RUNTIME = getattr(args.processing_params, 'cap_per_label', 2500)

        for orient, dataset_for_orient in final_cleaned_data_per_orientation.items():
            # Initialize these for each orientation, even if dataset_for_orient might be empty later.
            readable_dist_orient = Counter()
            final_counts_for_orient_after_bootstrap = Counter()

            if not dataset_for_orient:
                logger.info(f"No samples for orientation: {orient}. Skipping further processing.")
                # We still proceed to write empty stats/label_dict if no data, for consistency.
            else:
                # Bootstrapping (Step 4, per orientation after all cleaning)
                logger.info(f"\n" + "=" * 60)
                logger.info(f"STEP 4: BOOTSTRAPPING CLASSES FOR {orient} WITH FEWER THAN {min_samples} SAMPLES")
                logger.info("=" * 60)

                counts_before_bootstrap_for_orient = Counter(d.label for d in dataset_for_orient)

                bootstrapped_samples = []
                final_dataset_for_orient_with_bootstrap = []

                if not dataset_for_orient:
                    logger.info(f"No samples left for {orient} to bootstrap (inner check).")
                else:
                    classes_to_bootstrap = [cls for cls, count in counts_before_bootstrap_for_orient.items() if
                                            count < min_samples]

                    logger.info(
                        f"Bootstrapping {len(classes_to_bootstrap)} classes for {orient} with fewer than {min_samples} samples:")
                    if classes_to_bootstrap:
                        for cls in classes_to_bootstrap[:20]:
                            logger.info(f"  - {cls} ({counts_before_bootstrap_for_orient[cls]} samples)")
                        if len(classes_to_bootstrap) > 20:
                            logger.info(f"  - ... and {len(classes_to_bootstrap) - 20} more")
                    else:
                        logger.info("  - All classes meet minimum sample requirement, no bootstrapping needed.")

                    class_indices = {}
                    for i, sample in enumerate(dataset_for_orient):
                        if sample.label not in class_indices:
                            class_indices[sample.label] = []
                        class_indices[sample.label].append(i)

                    if do_bootstrap:
                        for class_name in tqdm(classes_to_bootstrap, desc=f"Bootstrapping {orient} classes"):
                            count = counts_before_bootstrap_for_orient[class_name]
                            if count >= min_samples: continue

                            indices = class_indices.get(class_name, [])
                            if not indices:
                                logger.warning(
                                    f"[WARNING] No samples found for class '{class_name}' in {orient} despite being in counts. Skipping bootstrap.")
                                continue

                            samples_needed = min_samples - count
                            for _ in range(samples_needed):
                                original_sample_idx = random.choice(indices)
                                original_sample_data = dataset_for_orient[original_sample_idx]

                                new_sample_data = dicomData(
                                    img=np.copy(original_sample_data.img),
                                    text_attributes=original_sample_data.text_attributes,
                                    numerical_attributes=list(
                                        original_sample_data.numerical_attributes) if original_sample_data.numerical_attributes is not None else None,
                                    label=original_sample_data.label
                                )
                                if new_sample_data.numerical_attributes is not None:
                                    noise = np.random.normal(0, 0.01, len(new_sample_data.numerical_attributes))
                                    new_sample_data.numerical_attributes = [max(0, float(val) + n) for val, n in
                                                                            zip(new_sample_data.numerical_attributes,
                                                                                noise)]
                                bootstrapped_samples.append(new_sample_data)
                    else:
                        logger.info("Bootstrapping is disabled (`do_bootstrap=False`).")

                    final_dataset_for_orient_with_bootstrap = dataset_for_orient + bootstrapped_samples
                    final_counts_for_orient_after_bootstrap = Counter(
                        d.label for d in final_dataset_for_orient_with_bootstrap)

                if not final_dataset_for_orient_with_bootstrap:
                    logger.warning(
                        f"No samples left for {orient} after bootstrapping. Skipping saving for this orientation.")
                else:
                    # Tokenization & Saving (Step 5)
                    logger.info(f"\n" + "=" * 60)
                    logger.info(f"STEP 5: TOKENIZING AND SAVING RESULTS FOR {orient}")
                    logger.info("=" * 60)

                    text_attrs_for_orient = [d.text_attributes for d in final_dataset_for_orient_with_bootstrap]

                    labels_in_final_orient_dataset = set(d.label for d in final_dataset_for_orient_with_bootstrap)
                    label_dict_for_orient = {label_name: i for i, label_name in
                                             enumerate(sorted(labels_in_final_orient_dataset))}

                    logger.info(
                        f"Found {len(label_dict_for_orient)} unique classes in final {orient} dataset: {', '.join(sorted(labels_in_final_orient_dataset))}")

                    logger.info(f"Tokenizing text data for {orient}...")
                    text_encoded = tokenizer(
                        text_attrs_for_orient,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        padding="longest",
                        truncation=True
                    )

                    current_max_length_for_orient = max_length
                    if current_max_length_for_orient is None:
                        lengths = [len(seq) for seq in text_encoded["input_ids"]]
                        sorted_lengths = sorted(lengths)
                        percentile_95_idx = int(len(sorted_lengths) * 0.95)
                        current_max_length_for_orient = min(512, sorted_lengths[percentile_95_idx])
                        logger.info(
                            f"Auto-selected max_length={current_max_length_for_orient} for {orient} (95th percentile, capped at 512)")

                    fixed_input_ids, fixed_attention_masks = fix_sequence_lengths(
                        text_encoded["input_ids"],
                        text_encoded["attention_mask"],
                        max_length=current_max_length_for_orient,
                        pad_token_id=tokenizer.pad_token_id or 0
                    )

                    orient_batch_dir = dataset_save_dir / orient / "batches"
                    orient_batch_dir.mkdir(parents=True, exist_ok=True)

                    buffer = []
                    total_samples_saved_in_batches = 0
                    for i, d in enumerate(
                            tqdm(final_dataset_for_orient_with_bootstrap, desc=f"Encoding & saving {orient} batches",
                                 ncols=100)):
                        sample = encodedSample(
                            img=d.img,
                            input_ids=fixed_input_ids[i],  # Use fixed_input_ids here
                            attention_mask=fixed_attention_masks[i],  # Use fixed_attention_masks here
                            numerical_attributes=d.numerical_attributes,
                            label=label_dict_for_orient[d.label],
                            orientation=orient
                        )
                        buffer.append(sample)

                        if len(buffer) == samples_per_batch_file:
                            batch_file = orient_batch_dir / f"dataset_batch_{i // samples_per_batch_file}.pkl"
                            with open(batch_file, "wb") as f:
                                pickle.dump(buffer, f)
                            total_samples_saved_in_batches += len(buffer)
                            buffer = []
                            gc.collect()

                    if buffer:  # Save any remaining items in the buffer
                        batch_file = orient_batch_dir / f"dataset_batch_final.pkl"
                        with open(batch_file, "wb") as f:
                            pickle.dump(buffer, f)
                        total_samples_saved_in_batches += len(buffer)
                        buffer = []

                    # --- Save sample images AFTER label_dict_for_orient is created and data is finalized ---
                    if final_dataset_for_orient_with_bootstrap:
                        save_sample_images(final_dataset_for_orient_with_bootstrap, dataset_save_dir, orient,
                                           label_mapping_dict=label_dict_for_orient)
                    # --- End Save sample images ---

                    inv_label_dict_for_stats = {v: k for k, v in label_dict_for_orient.items()}
                    readable_dist_orient = {label_str: count for label_str, count in
                                            final_counts_for_orient_after_bootstrap.items()}
                    for k_str, v_id in label_dict_for_orient.items():
                        if k_str not in readable_dist_orient:
                            readable_dist_orient[k_str] = 0

                    stats_orient = {
                        "total_samples": total_samples_saved_in_batches,  # Use the count from batching
                        "num_classes": len(label_dict_for_orient),
                        "class_distribution": readable_dist_orient,
                        "source_directories": [str(dir_path) for dir_path in pycurt_dicom_dirs],
                        "classification_target": getattr(args.classification, 'target', 'sequence') if hasattr(args,
                                                                                                               'classification') else 'sequence',
                        "orientation_processed": orient,
                        "filtering": {
                            "axes": list(accepted_axes_set) if use_axes_filter else [],
                            "anatomies": list(accepted_anatomies_set) if use_anatomies_filter else [],
                            "sequences": list(accepted_sequences_set) if use_sequences_filter else []
                        },
                        "sequence_length": current_max_length_for_orient,
                        "label_cap_per_class": CAP_PER_LABEL_RUNTIME,  # Read from config
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }

                    with open(dataset_save_dir / orient / f"dataset_stats_{orient}.json", "w") as f:
                        json.dump(stats_orient, f, indent=4)

                    with open(dataset_save_dir / orient / f"label_dict_{orient}.json", "w") as f:
                        json.dump(label_dict_for_orient, f, indent=4)  # This is saved separately

                    logger.info(
                        f"Successfully created {orient} dataset batches with {total_samples_saved_in_batches} samples.")
                    logger.info(f"Number of classes for {orient}: {len(label_dict_for_orient)}")
                    logger.info(f"Sequence length for {orient}: {current_max_length_for_orient}")
                    logger.info(f"Cap per label for {orient}: {CAP_PER_LABEL_RUNTIME}")
                    logger.info(f"Class distribution for {orient} (after bootstrapping):")
                    for label_name, count in sorted(readable_dist_orient.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  {label_name}: {count} samples")
                    logger.info(f"Dataset batches for {orient} saved to {dataset_save_dir / orient / 'batches'}")

    else:  # Process only a single orientation (TRA, COR, SAG etc.)
        logger.info(f"Configured to process single orientation: {single_target_orientation}")

        # The data_after_final_size_filtering already contains single-orientation dicomData objects
        final_cleaned_data_single_orientation = data_after_final_size_filtering

        # Initialize for single orientation branch
        readable_dist = Counter()
        final_counts_for_single_orient_after_bootstrap = Counter()

        # Bootstrapping (Step 4, for single orientation after all cleaning)
        logger.info(f"\n" + "=" * 60)
        logger.info(
            f"STEP 4: BOOTSTRAPPING CLASSES FOR {single_target_orientation} WITH FEWER THAN {min_samples} SAMPLES")
        logger.info("=" * 60)

        counts_before_bootstrap_for_single_orient = Counter(d.label for d in final_cleaned_data_single_orientation)

        bootstrapped_samples = []
        final_dataset_single_orient_with_bootstrap = []

        if not final_cleaned_data_single_orientation:
            logger.info(f"No samples left for {single_target_orientation} to bootstrap.")
        else:
            classes_to_bootstrap = [cls for cls, count in counts_before_bootstrap_for_single_orient.items() if
                                    count < min_samples]

            logger.info(
                f"Bootstrapping {len(classes_to_bootstrap)} classes for {single_target_orientation} with fewer than {min_samples} samples:")
            if classes_to_bootstrap:
                for cls in classes_to_bootstrap[:20]:
                    logger.info(f"  - {cls} ({counts_before_bootstrap_for_single_orient[cls]} samples)")
                if len(classes_to_bootstrap) > 20:
                    logger.info(f"  - ... and {len(classes_to_bootstrap) - 20} more")
            else:
                logger.info("  - All classes meet minimum sample requirement, no bootstrapping needed.")

            class_indices = {}
            for i, sample in enumerate(final_cleaned_data_single_orientation):
                if sample.label not in class_indices:
                    class_indices[sample.label] = []
                class_indices[sample.label].append(i)

            if do_bootstrap:
                for class_name in tqdm(classes_to_bootstrap, desc=f"Bootstrapping {single_target_orientation} classes"):
                    count = counts_before_bootstrap_for_single_orient[class_name]
                    if count >= min_samples: continue

                    indices = class_indices.get(class_name, [])
                    if not indices:
                        logger.warning(
                            f"[WARNING] No samples found for class '{class_name}' in {single_target_orientation} despite being in counts. Skipping bootstrap.")
                        continue

                    samples_needed = min_samples - count
                    for _ in range(samples_needed):
                        original_sample_idx = random.choice(indices)
                        original_sample_data = final_cleaned_data_single_orientation[original_sample_idx]

                        new_sample_data = dicomData(
                            img=np.copy(original_sample_data.img),  # Deep copy image data
                            text_attributes=original_sample_data.text_attributes,
                            numerical_attributes=list(
                                original_sample_data.numerical_attributes) if original_sample_data.numerical_attributes is not None else None,
                            label=original_sample_data.label
                        )
                        if new_sample_data.numerical_attributes is not None:
                            noise = np.random.normal(0, 0.01, len(new_sample_data.numerical_attributes))
                            new_sample_data.numerical_attributes = [max(0, float(val) + n) for val, n in
                                                                    zip(new_sample_data.numerical_attributes, noise)]
                        bootstrapped_samples.append(new_sample_data)
            else:
                logger.info("Bootstrapping is disabled (`do_bootstrap=False`).")

            final_dataset_single_orient_with_bootstrap = final_cleaned_data_single_orientation + bootstrapped_samples
            final_counts_for_single_orient_after_bootstrap = Counter(
                d.label for d in final_dataset_single_orient_with_bootstrap)

        if not final_dataset_single_orient_with_bootstrap:
            logger.error(f"No samples left for {single_target_orientation} after bootstrapping. Exiting.")
            return

        # Tokenization & Saving (Step 5)
        logger.info(f"\n" + "=" * 60)
        logger.info(f"STEP 5: TOKENIZING AND SAVING RESULTS FOR {single_target_orientation}")
        logger.info("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(args.training_params.tokenizer_name)

        text_attrs = [d.text_attributes for d in final_dataset_single_orient_with_bootstrap]

        labels_in_final_dataset = set(d.label for d in final_dataset_single_orient_with_bootstrap)
        final_label_dict_for_encoded_samples = {label_name: i for i, label_name in
                                                enumerate(sorted(labels_in_final_dataset))}

        logger.info(
            f"Found {len(final_label_dict_for_encoded_samples)} unique classes in final {single_target_orientation} dataset: {', '.join(sorted(labels_in_final_dataset))}")

        logger.info("Tokenizing text data...")
        text_encoded = tokenizer(
            text_attrs,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="longest",
            truncation=True
        )

        max_length_single_orient = getattr(args.training_params, 'max_sequence_length', None)
        if max_length_single_orient is None:
            lengths = [len(seq) for seq in text_encoded["input_ids"]]
            sorted_lengths = sorted(lengths)
            percentile_95_idx = int(len(sorted_lengths) * 0.95)
            max_length_single_orient = min(512, sorted_lengths[percentile_95_idx])
            logger.info(
                f"Auto-selected max_length={max_length_single_orient} for {single_target_orientation} (95th percentile, capped at 512)")

        fixed_input_ids, fixed_attention_masks = fix_sequence_lengths(
            text_encoded["input_ids"],
            text_encoded["attention_mask"],
            max_length=max_length_single_orient,
            pad_token_id=tokenizer.pad_token_id or 0
        )

        batch_dir = dataset_save_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)

        buffer = []
        total_samples_saved_in_batches = 0
        for i, d in enumerate(
                tqdm(final_dataset_single_orient_with_bootstrap, desc="Encoding & saving in batches", ncols=100)):
            sample = encodedSample(
                img=d.img,
                input_ids=fixed_input_ids[i],
                attention_mask=fixed_attention_masks[i],
                numerical_attributes=d.numerical_attributes,
                label=final_label_dict_for_encoded_samples[d.label],
                orientation=single_target_orientation
            )
            buffer.append(sample)

            if len(buffer) == samples_per_batch_file:
                batch_file = batch_dir / f"dataset_batch_{i // samples_per_batch_file}.pkl"
                with open(batch_file, "wb") as f:
                    pickle.dump(buffer, f)
                total_samples_saved_in_batches += len(buffer)
                buffer = []
                gc.collect()

        if buffer:
            batch_file = batch_dir / f"dataset_batch_final.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(buffer, f)
            total_samples_saved_in_batches += len(buffer)
            buffer = []

        # --- Save sample images AFTER label_dict is created and data is finalized ---
        if final_dataset_single_orient_with_bootstrap:
            save_sample_images(final_dataset_single_orient_with_bootstrap, dataset_save_dir, single_target_orientation,
                               label_mapping_dict=final_label_dict_for_encoded_samples)
        # --- End Save sample images ---

        # Update total_samples in stats to reflect total samples in batches
        stats = {
            "total_samples": total_samples_saved_in_batches,
            "num_classes": len(final_label_dict_for_encoded_samples),
            "class_distribution": final_counts_for_single_orient_after_bootstrap,
            "source_directories": [str(dir_path) for dir_path in pycurt_dicom_dirs],
            "classification_target": getattr(args.classification, 'target', 'sequence') if hasattr(args,
                                                                                                   'classification') else 'sequence',
            "orientation_processed": single_target_orientation,
            "filtering": {
                "axes": list(accepted_axes_set) if use_axes_filter else [],
                "anatomies": list(accepted_anatomies_set) if use_anatomies_filter else [],
                "sequences": list(accepted_sequences_set) if use_sequences_filter else []
            },
            "sequence_length": max_length_single_orient,
            "label_cap_per_class": CAP_PER_LABEL_RUNTIME,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        with open(dataset_save_dir / "label_dict.json", "w") as f:
            json.dump(final_label_dict_for_encoded_samples, f, indent=4)

        with open(dataset_save_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        logger.info(
            f"Successfully created {single_target_orientation} dataset batches with {total_samples_saved_in_batches} samples.")
        logger.info(f"Number of classes: {len(final_label_dict_for_encoded_samples)}")
        logger.info(f"Sequence length: {max_length_single_orient}")
        logger.info(f"Cap per label: {CAP_PER_LABEL_RUNTIME}")
        logger.info(f"Class distribution (after bootstrapping):")
        for label_name, count in sorted(final_counts_for_single_orient_after_bootstrap.items(), key=lambda x: x[1],
                                        reverse=True):
            logger.info(f"  {label_name}: {count} samples")
        logger.info(f"Dataset batches for {single_target_orientation} saved to {dataset_save_dir / 'batches'}")

    # --- FINAL CLEANUP OF INDIVIDUAL PROCESSED SAMPLES ---
    logger.info(f"\n--- Cleaning up individual processed samples from {individual_samples_dir} ---")
    deleted_count = 0
    # Use rglob to find files in subdirectories too if any were created
    for pkl_file in individual_samples_dir.rglob("*.pkl"):
        try:
            pkl_file.unlink()
            deleted_count += 1
        except OSError as e:
            logger.warning(f"Error deleting temporary file {pkl_file}: {e}")
    logger.info(f"Deleted {deleted_count} individual processed sample files.")

    # Attempt to remove all directories, starting from deepest
    for path in sorted(individual_samples_dir.rglob('*'), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
                # logger.debug(f"Removed empty directory: {path}") # Uncomment for verbose cleanup logs
            except OSError as e:
                logger.warning(f"Could not remove directory {path}: {e}")
    try:
        if individual_samples_dir.exists() and not any(individual_samples_dir.iterdir()):
            individual_samples_dir.rmdir()
            logger.info(f"Removed top-level empty directory: {individual_samples_dir}")
    except OSError as e:
        logger.warning(f"Could not remove top-level directory {individual_samples_dir}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str,
                        default="/home/e210/git/mrclass2/config/preproc_config_bp.json",
                        help="Path to config file.")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    get_dataset(args)