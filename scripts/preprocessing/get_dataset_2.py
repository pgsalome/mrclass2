import dicom2nifti
import nibabel as nib
import pydicom
import os
import pickle
import json
import gc
from pathlib import Path
# Changed to ProcessPoolExecutor for CPU-bound image processing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from transformers import AutoTokenizer
from loguru import logger
from typing import Union, List, Tuple, Dict, Set
from utils.io import read_yaml_config
from utils.dataclass import dicomData, encodedSample # Assuming these are defined elsewhere
from argparse import ArgumentParser, Namespace
from pycurtv2.processing.common import get_n_slice, resize_or_pad_image
from pycurtv2.converters.dicom import DicomConverter
from tqdm import tqdm
import glob # Keep glob for BPLabeler if it's called with string paths
from datetime import datetime

# Define the maximum samples per label
CAP_PER_LABEL = 2500

# Added for the early label extraction
# This dataclass is expected by the previous script (PostProcessing_Script)
@dataclass
class encodedSample:
    """Class for holding encoded DICOM data after preprocessing."""
    img: any  # np.ndarray
    input_ids: List[int]
    attention_mask: List[int]
    numerical_attributes: List[float]
    label: int # Kept as int as this script will output integer labels

@dataclass
class dicomData:
    """Class for holding raw DICOM data before encoding."""
    img: any
    text_attributes: str
    numerical_attributes: List[float]
    label: str # This is a string label before converting to int for encodedSample


def process_dicom_dir(dicom_dir_str: str, args) -> Union[dicomData, None]:
    """
    Processes a single DICOM directory: converts to NIfTI, reads metadata,
    resizes image, and extracts attributes.
    This function should be as efficient as possible.
    """
    dicom_dir = Path(dicom_dir_str) # Work with Path objects for consistency
    nifti_output_path = Path(str(dicom_dir) + ".nii.gz") # Construct Path object for nifti

    try:
        # Get slices once, use Path.glob for efficiency and robustness
        slices = list(dicom_dir.glob("*"))
        if not slices:
            logger.error(f"[ERROR] No DICOM slices found in {dicom_dir}. Skipping.")
            return None

        # Choose a representative slice, ensuring it's a file
        dicom_slice_path = None
        for s in slices:
            if s.is_file() and s.suffix.lower() in ['.dcm', '']: # Assuming typical DICOM extensions or no extension
                dicom_slice_path = s
                break
        if not dicom_slice_path:
            logger.error(f"[ERROR] No valid DICOM slice files found in {dicom_dir}. Skipping.")
            return None

        # Log at info level for progress, not tqdm.write in worker processes
        logger.info(f"Started image format conversion for {dicom_dir}")

        # Conversion to NIfTI (most likely bottleneck)
        DicomConverter(toConvert=str(dicom_dir)).convert_ps()

        # Image processing
        np_image = resize_or_pad_image(get_n_slice(str(nifti_output_path), str(dicom_slice_path)))

        # Read DICOM metadata
        metadata = pydicom.dcmread(str(dicom_slice_path), stop_before_pixels=True, force=True)

        # Pre-allocate lists and use f-strings for efficiency
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
                numerical_attributes_list.append(-1) # Handle cases where conversion to int fails

        # Label extraction - This will be *overwritten* later in get_dataset based on classification target
        # but a placeholder is needed for the dicomData object.
        # This initial label is mostly for the early label collection phase
        label = dicom_slice_path.parts[-3].replace('-DER', '') # Default, will be refined in wrapper

        dicom_data = dicomData(
            img=np_image,
            text_attributes=text_attribute,
            numerical_attributes=numerical_attributes_list,
            label=label # This is a preliminary label, refined in the main process
        )

        # Explicitly delete large objects if memory is an issue
        del np_image, text_attributes_list, numerical_attributes_list, metadata
        gc.collect() # Trigger garbage collection

        # Clean up temporary NIfTI file
        try:
            nifti_output_path.unlink(missing_ok=True) # Use unlink and missing_ok for cleaner deletion
        except OSError as e:
            logger.warning(f"[WARNING] Error removing NIfTI file {nifti_output_path}: {e}")

        return dicom_data

    except Exception as e:
        logger.error(f"[ERROR] Exception processing {dicom_dir}: {e}", exc_info=True) # Log full traceback
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

        fixed_input_ids_seq = list(input_ids) # Ensure lists are mutable copies
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


def get_dataset(args: Namespace) -> None:
    """
    Process DICOM directories to create a dataset for body part or sequence classification.
    Supports multiple input directories and flexible filtering criteria.
    Applies a maximum sample cap per label.
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

    ignored_modalities_set = set(args.ignored_modalities)

    use_axes_filter = False
    use_anatomies_filter = False
    use_sequences_filter = False

    accepted_axes_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'axes') and args.filtering.axes:
        accepted_axes_set = set(args.filtering.axes)
        use_axes_filter = True
    elif hasattr(args, 'accepted_axes') and args.accepted_axes:
        accepted_axes_set = set(args.accepted_axes)
        use_axes_filter = True

    accepted_anatomies_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'anatomies') and args.filtering.anatomies:
        accepted_anatomies_set = set(args.filtering.anatomies)
        use_anatomies_filter = True
    elif hasattr(args, 'accepted_anatomies') and args.accepted_anatomies:
        accepted_anatomies_set = set(args.accepted_anatomies)
        use_anatomies_filter = True

    accepted_sequences_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'sequences') and args.filtering.sequences:
        accepted_sequences_set = set(args.filtering.sequences)
        use_sequences_filter = True

    logger.info(f"Filtering with orientation: {args.orientation}")
    logger.info(f"Ignoring modalities: {ignored_modalities_set}")
    if use_axes_filter: logger.info(f"Filtering by axes: {accepted_axes_set}")
    else: logger.info("Including all axes")
    if use_anatomies_filter: logger.info(f"Filtering by anatomies: {accepted_anatomies_set}")
    else: logger.info("Including all anatomies")
    if use_sequences_filter: logger.info(f"Filtering by sequences: {accepted_sequences_set}")
    else: logger.info("Including all sequences")

    body_part_labeler = None
    use_custom_labeler = False
    if (hasattr(args, 'classification') and
                hasattr(args.classification, 'target') and
                args.classification.target == 'body_part' and
                hasattr(args.classification, 'custom_labeler') and
                args.classification.custom_labeler):
        from paste import BPLabeler # Import only if needed
        body_part_labeler = BPLabeler()
        use_custom_labeler = True
        logger.info("Using custom BPLabeler for body part classification")

    dicom_dirs_to_process = []
    # Collect all potential labels for early label_dict creation
    potential_labels: Set[str] = set()

    for pycurt_dir in pycurt_dicom_dirs:
        logger.info(f"Scanning directory: {pycurt_dir}")

        # Efficiently find DICOM directories by iterating through known structure levels
        for patient_dir in pycurt_dir.iterdir():
            if not patient_dir.is_dir(): continue
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir(): continue
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir(): continue
                    for axis_dir in series_dir.iterdir():
                        if not axis_dir.is_dir(): continue
                        for anatomy_dir in axis_dir.iterdir():
                            if not anatomy_dir.is_dir(): continue
                            for sequence_dir in anatomy_dir.iterdir():
                                # Only process directories, skip files and 'patches' subdirectories
                                if not sequence_dir.is_dir() or "patches" in sequence_dir.name:
                                    continue

                                # Extract parts for filtering
                                sequence_name = str(sequence_dir.name).split("-")[0]
                                anatomy_name = anatomy_dir.name
                                axis_name = axis_dir.name

                                # Apply filters
                                if sequence_name in ignored_modalities_set: continue
                                if use_axes_filter and axis_name not in accepted_axes_set: continue
                                if use_anatomies_filter and anatomy_name not in accepted_anatomies_set: continue
                                if use_sequences_filter and sequence_name not in accepted_sequences_set: continue
                                if "RGB" in anatomy_name: continue

                                dicom_dirs_to_process.append(str(sequence_dir)) # Pass as string for multiprocessing

                                # Early label extraction for label_dict.json
                                current_label_str_for_dict = ""
                                if hasattr(args, 'classification') and hasattr(args.classification, 'target'):
                                    if args.classification.target == 'body_part':
                                        # If custom_labeler is true, we cannot determine label early without processing DICOM
                                        # so we'll fall back to anatomy_name as an estimate for early label_dict
                                        if use_custom_labeler:
                                            # This is a heuristic: assume BPLabeler often uses anatomy or a derivation
                                            # A more robust solution might require a separate pre-scan step.
                                            # For now, we'll use anatomy_dir.name as a proxy if custom_labeler is enabled.
                                            # The actual label will be set in process_and_label_wrapper.
                                            current_label_str_for_dict = anatomy_name
                                        else:
                                            current_label_str_for_dict = anatomy_name
                                    elif args.classification.target == 'sequence':
                                        current_label_str_for_dict = sequence_name # Use sequence_name for early label
                                else:
                                    current_label_str_for_dict = sequence_name # Default to sequence name

                                if current_label_str_for_dict:
                                    potential_labels.add(current_label_str_for_dict)


        logger.info(f"Found {len(dicom_dirs_to_process)} valid directories in {pycurt_dir}")

    logger.info(f"{len(dicom_dirs_to_process)} total DICOM directories found after initial filtering.")

    if len(dicom_dirs_to_process) == 0:
        logger.error("No DICOM directories found that match the filtering criteria. Check your configuration.")
        return

    # --- NEW: Create and save label_dict.json and print classes early ---
    # Convert potential_labels to a sorted list to ensure reproducible ID assignment
    sorted_unique_labels = sorted(list(potential_labels))
    initial_label_dict = {label_name: i for i, label_name in enumerate(sorted_unique_labels)}

    initial_label_dict_path = dataset_save_dir / "label_dict.json"
    with open(initial_label_dict_path, "w") as f:
        json.dump(initial_label_dict, f, indent=4)
    logger.info(f"Initial label dictionary saved to: {initial_label_dict_path}")

    logger.info("\n--- Discovered Classes (Initial `label_dict.json`) ---")
    for label_name, label_id in initial_label_dict.items():
        logger.info(f"  ID: {label_id:<5} | Class: {label_name}")
    logger.info("--------------------------------------------------\n")
    # --- END NEW ---

    tqdm.write(f"Starting parallel DICOM processing with {len(dicom_dirs_to_process)} directories...")

    raw_processed_data = [] # Collect all results before capping
    # Use ProcessPoolExecutor for CPU-bound tasks like image processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor: # Use all available CPU cores
        # Wrapper to correctly assign the label based on classification target
        def process_and_label_wrapper(dicom_dir_str):
            result = process_dicom_dir(dicom_dir_str, args)
            if result is not None:
                d_path = Path(dicom_dir_str)

                # Determine the *final* label here (after dicom_dir processing)
                if hasattr(args, 'classification') and hasattr(args.classification, 'target'):
                    if args.classification.target == 'body_part':
                        if use_custom_labeler:
                            try:
                                # For BPLabeler, find a single DICOM file.
                                first_slice = [f for f in d_path.iterdir() if f.is_file() and f.suffix.lower() in ['.dcm', '']][0]
                                result.label = body_part_labeler.get_bodypart(str(first_slice))
                            except IndexError:
                                logger.warning(f"No .dcm file found for BPLabeler in {dicom_dir_str}. Labeling as NS.")
                                result.label = "NS"
                            except Exception as e:
                                logger.error(f"Error labeling body part for {dicom_dir_str}: {e}", exc_info=True)
                                result.label = "NS"
                        else:
                            result.label = str(d_path.parts[-3]) # Anatomy name
                    elif args.classification.target == 'sequence':
                        result.label = str(d_path.parts[-2]) # Sequence name
                else:
                    result.label = str(d_path.parts[-2]) # Default to sequence name (sequence folder name)

            return result

        futures = [executor.submit(process_and_label_wrapper, d_str) for d_str in dicom_dirs_to_process]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM dirs", ncols=100):
            result = f.result()
            if result is not None:
                raw_processed_data.append(result)

    if not raw_processed_data:
        logger.error("No valid DICOM data found after parallel processing. Check your data and filters.")
        return

    logger.info(f"Successfully processed {len(raw_processed_data)} raw DICOM directories.")

    # Apply the CAP_PER_LABEL limit per label AFTER all parallel processing is complete
    final_dataset_after_cap = []
    label_counts_after_cap = Counter()

    # Sort data by label to ensure a consistent (reproducible) selection when capping
    raw_processed_data.sort(key=lambda x: x.label)

    for data_item in tqdm(raw_processed_data, desc=f"Applying {CAP_PER_LABEL} samples/label cap", ncols=100):
        if label_counts_after_cap[data_item.label] < CAP_PER_LABEL:
            final_dataset_after_cap.append(data_item)
            label_counts_after_cap[data_item.label] += 1
        # else:
            # logger.debug(f"Skipping sample for label '{data_item.label}' as cap of {CAP_PER_LABEL} reached.")
            # Uncomment for verbose logging on skipped samples, but can be noisy for large datasets

    dataset = final_dataset_after_cap
    logger.info(f"Dataset size after capping: {len(dataset)} samples.")

    if not dataset:
        logger.error("No samples remaining after applying label cap. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.training_params.tokenizer_name)

    text_attrs = [d.text_attributes for d in dataset]

    # Re-collect all unique labels from the *capped* dataset
    # This ensures that the final label_dict only contains classes that actually exist in the final dataset
    labels_in_final_dataset = set(d.label for d in dataset)
    # The final label_dict might be smaller than initial_label_dict if some classes were capped to 0 or too few.
    # We should re-create the label_dict to map only existing labels to new sequential integers.
    final_label_dict_for_encoded_samples = {modality: i for i, modality in enumerate(sorted(labels_in_final_dataset))}


    logger.info(f"Found {len(final_label_dict_for_encoded_samples)} unique classes in final capped dataset: {', '.join(sorted(labels_in_final_dataset))}")

    logger.info("Tokenizing text data...")
    text_encoded = tokenizer(
        text_attrs,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="longest",
        truncation=True
    )

    max_length = getattr(args.training_params, 'max_sequence_length', None)
    if max_length is None:
        lengths = [len(seq) for seq in text_encoded["input_ids"]]
        sorted_lengths = sorted(lengths)
        percentile_95_idx = int(len(sorted_lengths) * 0.95)
        max_length = min(512, sorted_lengths[percentile_95_idx])
        logger.info(f"Auto-selected max_length={max_length} (95th percentile, capped at 512)")

    fixed_input_ids, fixed_attention_masks = fix_sequence_lengths(
        text_encoded["input_ids"],
        text_encoded["attention_mask"],
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id or 0
    )

    text_encoded["input_ids"] = fixed_input_ids
    text_encoded["attention_mask"] = fixed_attention_masks

    lengths_after_fix = [len(seq) for seq in text_encoded["input_ids"]]
    assert len(set(lengths_after_fix)) == 1, f"Not all sequences have the same length after fixing: {set(lengths_after_fix)}"
    logger.info(f"✅ All sequences now have uniform length: {lengths_after_fix[0]}")

    batch_size = 1500
    batch_dir = dataset_save_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    buffer = []
    for i, d in enumerate(tqdm(dataset, desc="Encoding & saving in batches", ncols=100)):
        sample = encodedSample(
            img=d.img,
            input_ids=text_encoded["input_ids"][i],
            attention_mask=text_encoded["attention_mask"][i],
            numerical_attributes=d.numerical_attributes,
            label=final_label_dict_for_encoded_samples[d.label], # Use the newly created label_dict for integer IDs
        )
        buffer.append(sample)

        if len(buffer) == batch_size:
            batch_file = batch_dir / f"dataset_batch_{i // batch_size}.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(buffer, f)
            buffer = []
            gc.collect() # Aggressive GC after saving a batch

    if buffer: # Save any remaining items in the buffer
        batch_file = batch_dir / f"dataset_batch_final.pkl"
        with open(batch_file, "wb") as f:
            pickle.dump(buffer, f)

    all_batches = sorted(batch_dir.glob("dataset_batch_*.pkl"))
    full_dataset = []
    for batch_file in tqdm(all_batches, desc="Merging batches", ncols=100):
        with open(batch_file, "rb") as f:
            full_dataset.extend(pickle.load(f))
        batch_file.unlink(missing_ok=True) # Delete batch file after merging to save disk space
        gc.collect()

    sample_lengths = [len(sample.input_ids) for sample in full_dataset[:100]]
    if len(set(sample_lengths)) == 1:
        logger.info(f"✅ Final verification passed: All saved sequences have length {sample_lengths[0]}")
    else:
        logger.error(f"❌ Final verification failed: Found different lengths {set(sample_lengths)}")

    with open(dataset_save_dir / "dataset.pkl", "wb") as f:
        pickle.dump(full_dataset, f)

    # Overwrite the initial label_dict with the final one that reflects capping
    # This is important for consistency with the generated dataset.pkl
    with open(dataset_save_dir / "label_dict.json", "w") as f:
        json.dump(final_label_dict_for_encoded_samples, f, indent=4) # Use the final_label_dict


    # Use the label_counts_after_cap for statistics, as it reflects the capped distribution
    inv_label_dict_for_stats = {v: k for k, v in final_label_dict_for_encoded_samples.items()}
    readable_dist = {inv_label_dict_for_stats[label_id]: count for label_id, count in label_counts_after_cap.items()}
    # Ensure all labels in final_label_dict_for_encoded_samples are represented in readable_dist, even if their count is 0 after capping
    for k_str, v_id in final_label_dict_for_encoded_samples.items():
        if k_str not in readable_dist:
            readable_dist[k_str] = 0

    stats = {
        "total_samples": len(full_dataset),
        "num_classes": len(final_label_dict_for_encoded_samples),
        "class_distribution": readable_dist,
        "source_directories": [str(dir_path) for dir_path in pycurt_dicom_dirs],
        "classification_target": getattr(args.classification, 'target', 'sequence') if hasattr(args, 'classification') else 'sequence',
        "filtering": {
            "axes": list(accepted_axes_set) if use_axes_filter else [],
            "anatomies": list(accepted_anatomies_set) if use_anatomies_filter else [],
            "sequences": list(accepted_sequences_set) if use_sequences_filter else []
        },
        "sequence_length": max_length,
        "label_cap_per_class": CAP_PER_LABEL, # Add the new cap information to stats
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open(dataset_save_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Successfully created dataset with {len(full_dataset)} samples.")
    logger.info(f"Number of classes: {len(final_label_dict_for_encoded_samples)}")
    logger.info(f"Sequence length: {max_length}")
    logger.info(f"Cap per label: {CAP_PER_LABEL}")

    logger.info("Class distribution (after capping):")
    for label_name, count in sorted(readable_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label_name}: {count} samples")

    logger.info(f"Dataset saved to {dataset_save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="/home/e210/git/mrclass2/config/preproc_config_bp.json",
                        help="Path to config file.")
    # You can add an argument for the label cap if you want to make it configurable
    # parser.add_argument("--label_cap", type=int, default=2500, help="Maximum number of samples per label.")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    get_dataset(args)