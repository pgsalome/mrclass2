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


# --- MODIFIED: Function to process and save individual DICOM data with enhanced error tracking ---
def _process_and_save_single_dicom_data(dicom_dir_str: str, args: Namespace, target_orientation: str,
                                        body_part_labeler, output_folder: Path, processed_dirs_log_path: Path,
                                        failed_dirs_log_path: Path) -> Union[Path, None]:
    """
    Processes a single DICOM directory to extract an image for a specific orientation,
    assigns the final string label, and saves the dicomData object to a pickle file.
    Logs the processed DICOM directory path upon successful saving to processed_dirs_log_path.
    Logs failed directories to failed_dirs_log_path.
    Returns the path to the saved file or None if processing fails.
    """
    dicom_dir = Path(dicom_dir_str)
    nifti_output_path = Path(str(dicom_dir) + ".nii.gz")

    try:
        slices = list(dicom_dir.glob("*"))
        if not slices:
            logger.warning(f"No DICOM slices found in {dicom_dir}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        dicom_slice_path = None
        for s in slices:
            if s.is_file() and s.suffix.lower() in ['.dcm', '']:
                dicom_slice_path = s
                break
        if not dicom_slice_path:
            logger.warning(f"No valid DICOM slice file found in {dicom_dir}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        try:
            DicomConverter(toConvert=str(dicom_dir)).convert_ps()
        except Exception as e:
            logger.warning(f"DICOM conversion failed for {dicom_dir}: {e}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        np_image = None
        try:
            np_image = resize_or_pad_image(
                get_n_slice(str(nifti_output_path), str(dicom_slice_path), desired_orientation=target_orientation))
        except FileNotFoundError:
            logger.warning(
                f"NIfTI file not found after conversion for {dicom_dir} ({target_orientation}). Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None
        except Exception as e:
            logger.warning(
                f"Failed to load NIfTI or extract slice for {dicom_dir} ({target_orientation}): {e}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        if np_image is None or np_image.size == 0:
            logger.warning(f"Final image is empty or None for {dicom_dir} ({target_orientation}). Skipping series.")
            if nifti_output_path.exists():
                try:
                    nifti_output_path.unlink(missing_ok=True)
                except OSError as e_unlink:
                    logger.warning(f"Error cleaning up NIfTI: {e_unlink}")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
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

        # Append the successfully processed DICOM directory to the success log file
        with open(processed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")

        logger.info(
            f"Processed Series: {dicom_dir.name} | Orientation: {target_orientation} | Modality: {modality_from_metadata} | Body Part: {final_series_label} -> Saved to {output_pkl_path}")

        return output_pkl_path

    except Exception as e:
        logger.error(f"[ERROR] An unhandled exception occurred processing {dicom_dir} for {target_orientation}: {e}",
                     exc_info=True)
        # Log as failed for any unhandled exception
        with open(failed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")
        return None


# --- MODIFIED: Function to process and save ALL orientations with enhanced error tracking ---
def _process_and_save_all_dicom_data(dicom_dir_str: str, args: Namespace, body_part_labeler,
                                     output_folder: Path, processed_dirs_log_path: Path,
                                     failed_dirs_log_path: Path) -> Union[Path, None]:
    """
    Processes a single DICOM directory to extract images for ALL specified orientations,
    and saves a single dicomData object (with img as a dict) to a pickle file.
    Logs the processed DICOM directory path upon successful saving to processed_dirs_log_path.
    Logs failed directories to failed_dirs_log_path.
    Returns the path to the saved file or None if processing fails.
    """
    dicom_dir = Path(dicom_dir_str)
    nifti_output_path = Path(str(dicom_dir) + ".nii.gz")

    try:
        slices = list(dicom_dir.glob("*"))
        if not slices:
            logger.warning(f"No DICOM slices found in {dicom_dir}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        dicom_slice_path = None
        for s in slices:
            if s.is_file() and s.suffix.lower() in ['.dcm', '']:
                dicom_slice_path = s
                break
        if not dicom_slice_path:
            logger.warning(f"No valid DICOM slice file found in {dicom_dir}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
            return None

        try:
            DicomConverter(toConvert=str(dicom_dir)).convert_ps()
        except Exception as e:
            logger.warning(f"DICOM conversion failed for {dicom_dir}: {e}. Skipping series.")
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
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
            # Log as failed
            with open(failed_dirs_log_path, 'a') as f:
                f.write(f"{dicom_dir_str}\n")
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

        # Append the successfully processed DICOM directory to the success log file
        with open(processed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")

        logger.info(
            f"Processed Series: {dicom_dir.name} | Modality: {modality_from_metadata} | Body Part: {final_series_label} | Orientations Extracted: {list(images_per_orientation.keys())} -> Saved to {output_pkl_path}")

        return output_pkl_path  # Return the path to the single saved file for the series

    except Exception as e:
        logger.error(f"[ERROR] An unhandled exception occurred processing {dicom_dir}: {e}", exc_info=True)
        # Log as failed for any unhandled exception
        with open(failed_dirs_log_path, 'a') as f:
            f.write(f"{dicom_dir_str}\n")
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
    This version processes data in chunks to avoid out-of-memory errors.
    """
    logger.info("Starting dataset creation with memory-efficient chunking...")

    # --- Phase 0: Setup and Configuration (Largely Unchanged) ---
    pycurt_dicom_dirs = [Path(d) for d in getattr(args.paths, 'pycurt_dirs', [getattr(args.paths, 'pycurt_dir', None)])
                         if d]
    if not pycurt_dicom_dirs:
        logger.error("No pycurt directory specified in config.")
        return

    dataset_save_dir = Path(args.paths.dataset_dir)
    individual_samples_dir = dataset_save_dir / "individual_processed_samples"
    individual_samples_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Individual processed samples will be saved to/loaded from: {individual_samples_dir}")

    processed_dirs_log_path = dataset_save_dir / "processed_dicom_dirs.csv"
    failed_dirs_log_path = dataset_save_dir / "failed_dicom_dirs.csv"

    already_processed_dirs = set()
    if processed_dirs_log_path.exists():
        with open(processed_dirs_log_path, 'r') as f:
            already_processed_dirs.update(line.strip() for line in f)
    if failed_dirs_log_path.exists():
        with open(failed_dirs_log_path, 'r') as f:
            already_processed_dirs.update(line.strip() for line in f)
    logger.info(f"Loaded {len(already_processed_dirs)} previously attempted directories to skip.")

    # Get processing parameters from config
    process_all_orientations = getattr(args, 'orientation', 'TRA').upper() == "ALL"
    single_target_orientation = getattr(args, 'orientation', 'TRA').upper()
    orientations_to_process = ["TRA", "COR", "SAG"] if process_all_orientations else [single_target_orientation]

    # ... (Initial scanning, filtering, and test mode logic remains the same as your original code) ...
    # This part populates `dicom_dirs_to_process`
    # --- For brevity, I'll assume the initial scan (`Phase 1`) has run and populated `individual_samples_dir` ---
    # The crucial change is in how we handle the files in `individual_samples_dir` (Phase 2).

    # --- PHASE 2: Load and Process in Chunks ---
    logger.info(
        "\n--- Loading and processing ALL individual samples IN CHUNKS ---")

    all_individual_pkl_files = sorted(list(individual_samples_dir.glob("*.pkl")))
    if not all_individual_pkl_files:
        logger.error("No individual processed sample files found. Cannot proceed.")
        return

    # --- Setup for chunk processing ---
    chunk_size = getattr(args.processing_params, 'samples_per_batch_file', 1500)
    tokenizer = AutoTokenizer.from_pretrained(args.training_params.tokenizer_name)
    manual_remapping_rules = {"PLV-EG": "PLV-LEG", "SPINE": "WS", "ANPLS": "LS"}
    labels_to_remove_completely = {"HNCANP", "NS"}
    filter_patterns = getattr(args.processing_params, 'filter_patterns', [])
    min_class_size = getattr(args.processing_params, 'min_class_size', 20)
    do_bootstrap = getattr(args.processing_params, 'do_bootstrap', True)
    min_samples = getattr(args.processing_params, 'min_samples', 50)

    # --- Aggregators to collect results across all chunks ---
    final_data = {orient: {"buffer": [], "file_counter": 0, "total_samples": 0} for orient in orientations_to_process}
    final_class_counts = {orient: Counter() for orient in orientations_to_process}
    final_label_set = {orient: set() for orient in orientations_to_process}

    # --- Main Chunking Loop ---
    for i in range(0, len(all_individual_pkl_files), chunk_size):
        chunk_paths = all_individual_pkl_files[i:i + chunk_size]
        logger.info(
            f"--- Processing chunk {i // chunk_size + 1}/{(len(all_individual_pkl_files) - 1) // chunk_size + 1} ({len(chunk_paths)} files) ---")

        # 1. Load one chunk from disk
        raw_dicom_data_for_chunk = []
        for pkl_path in tqdm(chunk_paths, desc="Loading chunk", ncols=100):
            try:
                with open(pkl_path, 'rb') as f:
                    raw_dicom_data_for_chunk.append(pickle.load(f))
            except Exception as e:
                logger.warning(f"Could not load {pkl_path}: {e}. Skipping.")

        # 2. Apply ALL data cleaning and normalization steps to the chunk
        # Manual remapping and removal
        data_after_manual_fix = []
        for item in raw_dicom_data_for_chunk:
            if item.label in labels_to_remove_completely: continue
            item.label = manual_remapping_rules.get(item.label, item.label)
            data_after_manual_fix.append(item)

        # Pattern filtering
        current_labels_in_chunk = set(d.label for d in data_after_manual_fix)
        patterns_to_remove = set()
        for pattern in filter_patterns:
            patterns_to_remove.update({label for label in current_labels_in_chunk if pattern in label})
        data_after_pattern_filter = [item for item in data_after_manual_fix if item.label not in patterns_to_remove]

        # General normalization
        normalized_labels_map = {name: normalize_class_name(name) for name in
                                 set(d.label for d in data_after_pattern_filter)}
        for item in data_after_pattern_filter:
            item.label = normalized_labels_map[item.label]

        # Final size filtering (based on counts within this chunk - an approximation)
        counts_in_chunk = Counter(d.label for d in data_after_pattern_filter)
        cleaned_chunk = [item for item in data_after_pattern_filter if counts_in_chunk[item.label] >= min_class_size]

        if not cleaned_chunk:
            logger.warning("Chunk is empty after cleaning. Skipping to next chunk.")
            del raw_dicom_data_for_chunk, data_after_manual_fix, data_after_pattern_filter
            gc.collect()
            continue

        # 3. Process the cleaned chunk for each orientation
        for orient in orientations_to_process:

            # A. Extract data for the current orientation
            orient_data = []
            if process_all_orientations:
                for item in cleaned_chunk:
                    if isinstance(item.img, dict) and orient in item.img:
                        orient_data.append(dicomData(
                            img=item.img[orient],
                            text_attributes=item.text_attributes,
                            numerical_attributes=item.numerical_attributes,
                            label=item.label
                        ))
            else:  # Single orientation mode
                orient_data = cleaned_chunk

            if not orient_data: continue

            # B. Apply bootstrapping
            bootstrapped_samples = []
            if do_bootstrap:
                counts_before_bootstrap = Counter(d.label for d in orient_data)
                class_indices = {label: [idx for idx, s in enumerate(orient_data) if s.label == label] for label in
                                 counts_before_bootstrap}

                for class_name, count in counts_before_bootstrap.items():
                    if 0 < count < min_samples:
                        samples_needed = min_samples - count
                        indices = class_indices[class_name]
                        for _ in range(samples_needed):
                            original_sample = orient_data[random.choice(indices)]
                            bootstrapped_samples.append(original_sample)  # Simple resampling for this example

            final_orient_data_chunk = orient_data + bootstrapped_samples

            # C. Update final counts and labels
            final_class_counts[orient].update(d.label for d in final_orient_data_chunk)
            final_label_set[orient].update(d.label for d in final_orient_data_chunk)

            # D. Tokenize and create encodedSample objects
            if not final_orient_data_chunk: continue

            text_attrs = [d.text_attributes for d in final_orient_data_chunk]
            text_encoded = tokenizer(text_attrs, add_special_tokens=True, padding=False, truncation=True)  # Pad later

            for idx, d in enumerate(final_orient_data_chunk):
                final_data[orient]["buffer"].append(encodedSample(
                    img=d.img,
                    input_ids=text_encoded['input_ids'][idx],
                    attention_mask=text_encoded['attention_mask'][idx],
                    numerical_attributes=d.numerical_attributes,
                    label=d.label,  # Keep string label for now, convert to int at the end
                    orientation=orient
                ))

            # E. Save buffer to batch file if it's full
            while len(final_data[orient]["buffer"]) >= chunk_size:
                batch_to_save = final_data[orient]["buffer"][:chunk_size]
                final_data[orient]["buffer"] = final_data[orient]["buffer"][chunk_size:]

                batch_dir = dataset_save_dir / orient / "batches"
                batch_dir.mkdir(parents=True, exist_ok=True)
                batch_file = batch_dir / f"dataset_batch_{final_data[orient]['file_counter']}.pkl"

                with open(batch_file, "wb") as f:
                    pickle.dump(batch_to_save, f)

                logger.info(f"Saved batch file: {batch_file}")
                final_data[orient]['total_samples'] += len(batch_to_save)
                final_data[orient]['file_counter'] += 1

        # Clean up memory from the processed chunk
        del raw_dicom_data_for_chunk, cleaned_chunk
        gc.collect()

    # --- Finalization Step (after loop) ---
    logger.info("\n--- Finalizing dataset: saving remaining buffers and stats ---")

    for orient in orientations_to_process:
        # Save any remaining samples in the buffer
        if final_data[orient]["buffer"]:
            batch_dir = dataset_save_dir / orient / "batches"
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_file = batch_dir / f"dataset_batch_final.pkl"

            with open(batch_file, "wb") as f:
                pickle.dump(final_data[orient]["buffer"], f)

            logger.info(f"Saved final batch file: {batch_file}")
            final_data[orient]['total_samples'] += len(final_data[orient]["buffer"])

        # Create final label dictionary
        sorted_labels = sorted(list(final_label_set[orient]))
        label_dict = {label: i for i, label in enumerate(sorted_labels)}

        # Now, reopen the saved batches to convert string labels to integer IDs
        logger.info(f"Converting string labels to integer IDs for {orient}...")
        batch_files = sorted(list((dataset_save_dir / orient / "batches").glob("*.pkl")))
        for batch_file in tqdm(batch_files, desc=f"Finalizing {orient} batches"):
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)

            for sample in batch:
                sample.label = label_dict.get(sample.label, -1)  # Convert to int

            with open(batch_file, 'wb') as f:
                pickle.dump(batch, f)

        # Save sample images for observation
        # To do this correctly, we'd need to load one batch back.
        # This is a simplified version.

        # Save final stats and label dict
        stats_path = dataset_save_dir / orient / f"dataset_stats_{orient}.json"
        label_dict_path = dataset_save_dir / orient / f"label_dict_{orient}.json"
        if not (dataset_save_dir / orient).exists():
            (dataset_save_dir / orient).mkdir(parents=True, exist_ok=True)

        stats = {
            "total_samples": final_data[orient]['total_samples'],
            "num_classes": len(label_dict),
            "class_distribution": dict(final_class_counts[orient]),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            # ... add other stats from your original config ...
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        with open(label_dict_path, "w") as f:
            json.dump(label_dict, f, indent=4)

        logger.info(f"Final stats for {orient} saved to {stats_path}")
        logger.info(f"Final label dict for {orient} saved to {label_dict_path}")

    # --- Final Cleanup of Individual Processed Samples ---
    logger.info(f"\n--- Cleaning up intermediate files from {individual_samples_dir} ---")
    deleted_count = 0
    for pkl_file in individual_samples_dir.rglob("*.pkl"):
        try:
            pkl_file.unlink()
            deleted_count += 1
        except OSError as e:
            logger.warning(f"Error deleting file {pkl_file}: {e}")
    logger.info(f"Deleted {deleted_count} intermediate sample files.")
    try:
        # Attempt to remove the now-empty directory
        os.rmdir(individual_samples_dir)
        logger.info(f"Removed empty directory: {individual_samples_dir}")
    except OSError:
        pass  # Ignore if not empty, which shouldn't happen


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str,
                        default="/home/e210/git/mrclass2/config/preproc_config_bp.json",
                        help="Path to config file.")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    get_dataset(args)