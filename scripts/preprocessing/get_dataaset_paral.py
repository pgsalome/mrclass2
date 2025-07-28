import dicom2nifti
import nibabel as nib
import pydicom
import os
import pickle
import json
import gc
import psutil
import multiprocessing
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter
from transformers import AutoTokenizer
from loguru import logger
from typing import Union, List, Tuple, Dict, Any
from utils.io import read_yaml_config
from utils.dataclass import dicomData, encodedSample
from argparse import ArgumentParser, Namespace
from pycurtv2.processing.common import get_n_slice, resize_or_pad_image
from pycurtv2.converters.dicom import DicomConverter
from tqdm import tqdm
import glob
from datetime import datetime


# Configuration for parallel processing
def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    cpu_count = multiprocessing.cpu_count()
    mem_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB

    # Reduced memory per worker estimate to allow more workers (1.0 GB instead of 1.5)
    mem_workers = max(1, int(mem_available / 1.0))

    # Increased CPU utilization from 75% to 90%
    workers = min(max(1, int(cpu_count * 0.9)), mem_workers)

    logger.info(f"Optimal workers calculation: CPU count={cpu_count}, available memory={mem_available:.2f}GB")
    logger.info(f"Workers from memory: {mem_workers}, From CPU: {int(cpu_count * 0.9)}, Final: {workers}")

    return workers


# Memory monitoring
def log_memory_usage(tag):
    """Log current memory usage with a tag"""
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 * 1024)  # MB
    logger.info(f"Memory usage ({tag}): {memory:.2f} MB")

    # Output extra diagnostics if memory usage is high
    if memory > 4000:  # If using more than 4GB
        # Log the size of the largest objects
        objects = []
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, type):
                    size = sys.getsizeof(obj)
                    objects.append((size, type(obj).__name__, obj))
            except:
                pass

        # Log the 5 largest objects by size
        top_objects = sorted(objects, key=lambda x: x[0], reverse=True)[:5]
        for size, type_name, obj in top_objects:
            logger.info(f"Large object: {type_name} - {size / 1024 / 1024:.2f} MB")


def configure_memory_settings():
    """Configure memory settings for optimal performance"""
    # Set lower GC threshold to clean up more aggressively
    gc.set_threshold(700, 10, 5)  # Default is (700, 10, 10)

    # Disable automatic garbage collection and manage it manually
    gc.disable()

    # Return initial memory usage for tracking
    return psutil.Process().memory_info().rss / (1024 * 1024)  # MB


# Batch processing helpers
def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def scan_directory(pycurt_dir, ignored_modalities_set, use_axes_filter,
                   accepted_axes_set, use_anatomies_filter, accepted_anatomies_set,
                   use_sequences_filter, accepted_sequences_set):
    """Scan a single directory for DICOM directories with filtering"""
    logger.info(f"Scanning directory: {pycurt_dir}")

    # Build glob pattern based on directory structure
    glob_pattern = str(pycurt_dir) + "/*/*/*/*/*/*"

    # Function to filter directories based on criteria
    def filter_dir(d):
        d_path = Path(d)
        if not d_path.is_dir() or "patches" in d_path.name:
            return False

        # Parts[-2] is typically the sequence name
        # Parts[-3] is typically the anatomy name
        # Parts[-4] is typically the axis/orientation

        # Filter by ignored modalities
        sequence_name = str(d_path.parts[-2]).split("-")[0]
        if sequence_name in ignored_modalities_set:
            return False

        # Filter by axes if filter is active
        if use_axes_filter and d_path.parts[-4] not in accepted_axes_set:
            return False

        # Filter by anatomy if filter is active
        if use_anatomies_filter and d_path.parts[-3] not in accepted_anatomies_set:
            return False

        # Filter by sequence if filter is active
        if use_sequences_filter and sequence_name not in accepted_sequences_set:
            return False

        # Filter out RGB directories
        if "RGB" in d_path.parts[-3]:
            return False

        return True

    dirs = [d for d in glob.glob(glob_pattern) if filter_dir(d)]
    logger.info(f"Found {len(dirs)} valid directories in {pycurt_dir}")

    return dirs


def process_dicom_dir(dicom_dir: Path, args_dict: Dict) -> Union[dicomData, None]:
    """Process a single DICOM directory"""
    # Check if cache exists
    cache_file = Path(str(dicom_dir) + "_processed_cache.pkl")

    # Check if cache exists and is newer than source files
    if cache_file.exists() and cache_file.stat().st_mtime > Path(dicom_dir).stat().st_mtime:
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # We need to check if the cached data structure matches what we expect
                if isinstance(cached_data, dicomData):
                    return cached_data
        except Exception as e:
            tqdm.write(f"[WARNING] Failed to load cache for {dicom_dir}: {e}")
            # Continue with normal processing

    try:
        slices = glob.glob(str(dicom_dir) + "/*")
        if not slices:
            tqdm.write(f"[ERROR] No DICOM slices found in {dicom_dir}. Skipping.")
            return None

        dicom_slice = str(slices[len(slices) // 2])
        tqdm.write(f"Started image format conversion for {dicom_dir}")
        DicomConverter(toConvert=dicom_dir).convert_ps()
        nifti_output_path = str(dicom_dir) + ".nii.gz"

        np_image = resize_or_pad_image(get_n_slice(nifti_output_path, dicom_slice))

        metadata = pydicom.dcmread(dicom_slice, stop_before_pixels=True, force=True)

        text_attributes_list = [
            f"{attribute} is {getattr(metadata, attribute)}"
            if hasattr(metadata, attribute) and getattr(metadata, attribute) is not None
            else args_dict['missing_term']
            for attribute in args_dict['text_attributes']
        ]
        text_attribute = ", ".join(text_attributes_list)

        numerical_attributes_list = [
            int(getattr(metadata, attribute))
            if hasattr(metadata, attribute) and getattr(metadata, attribute) is not None
            else -1
            for attribute in args_dict['numerical_attributes']
        ]

        label = Path(dicom_slice).parts[-3].replace('-DER', '')

        dicom_data = dicomData(
            img=np_image,
            text_attributes=text_attribute,
            numerical_attributes=numerical_attributes_list,
            label=label
        )

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dicom_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            tqdm.write(f"[WARNING] Failed to cache {dicom_dir}: {e}")

        del np_image, text_attributes_list, numerical_attributes_list, metadata
        try:
            os.remove(nifti_output_path)
        except FileNotFoundError:
            tqdm.write(f"[WARNING] No NIfTI file found at {nifti_output_path}.")

        # Manually trigger garbage collection
        gc.collect()

        return dicom_data

    except Exception as e:
        tqdm.write(f"[ERROR] Exception processing {dicom_dir}: {e}")
        return None


def process_dicom_batch(dicom_dirs_batch: List[str], args_dict: Dict, use_custom_labeler: bool = False,
                        body_part_labeler: Any = None) -> List[dicomData]:
    """Process a batch of DICOM directories"""
    results = []

    # Process each directory in the batch
    for dicom_dir in dicom_dirs_batch:
        result = process_dicom_dir(dicom_dir, args_dict)
        if result is not None:
            # Determine the label based on the classification target
            d_path = Path(dicom_dir)

            classification_target = args_dict.get('classification_target', 'sequence')

            if classification_target == 'body_part':
                if use_custom_labeler and body_part_labeler is not None:
                    # Use BPLabeler to determine the body part
                    try:
                        first_slice = glob.glob(dicom_dir + "/*.dcm")[0]
                        result.label = body_part_labeler.get_bodypart(first_slice)
                    except Exception as e:
                        logger.error(f"Error labeling body part: {str(e)}")
                        result.label = "NS"
                else:
                    # Use folder name as body part label
                    result.label = str(d_path.parts[-3])
            elif classification_target == 'sequence':
                # Use sequence (folder name) as label
                sequence_name = str(d_path.parts[-2])
                result.label = sequence_name
            else:
                # Default: use sequence (folder name) as label
                sequence_name = str(d_path.parts[-2])
                result.label = sequence_name

            results.append(result)

    return results


def optimize_tokenization(text_attrs, tokenizer_name, batch_size=1000):
    """Tokenize text data in batches to reduce memory usage"""
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Process in batches
    all_input_ids = []
    all_attention_masks = []

    for i in range(0, len(text_attrs), batch_size):
        batch = text_attrs[i:i + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="longest",
            truncation=True
        )

        all_input_ids.extend(encoded["input_ids"])
        all_attention_masks.extend(encoded["attention_mask"])

        # Clear batch from memory
        del batch, encoded
        gc.collect()

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}


def save_batch(batch_data, batch_index, batch_dir):
    """Save a batch of data to disk with improved error handling"""
    batch_file = batch_dir / f"dataset_batch_{batch_index}.pkl"

    # Use a temporary file first to prevent corruption if interrupted
    temp_file = batch_dir / f"temp_{batch_index}.pkl"
    try:
        with open(temp_file, "wb") as f:
            pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Rename to final name when complete
        temp_file.rename(batch_file)
    except Exception as e:
        logger.error(f"Error saving batch {batch_index}: {e}")
        # Try to clean up temp file if it exists
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except:
                pass
        raise

    # Force garbage collection after saving a batch
    del batch_data
    gc.collect()

    return batch_file


def log_progress(processed, total, start_time, process=None):
    """Log progress with ETA and memory usage"""
    if process is None:
        process = psutil.Process()

    elapsed = time.time() - start_time
    progress = processed / total if total > 0 else 0

    if progress > 0:
        eta = elapsed / progress * (1 - progress)
        eta_str = f"{eta:.1f}s"
        if eta > 60:
            eta_str = f"{eta / 60:.1f}m"
        if eta > 3600:
            eta_str = f"{eta / 3600:.1f}h"
    else:
        eta_str = "Unknown"

    mem_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    logger.info(f"Progress: {processed}/{total} ({progress * 100:.1f}%) - ETA: {eta_str} - Memory: {mem_usage:.1f} GB")


def get_dataset(args: Namespace) -> None:
    """
    Process DICOM directories to create a dataset for body part or sequence classification.
    Supports multiple input directories and flexible filtering criteria.

    Args:
        args: Configuration namespace containing dataset parameters
    """
    # Configure memory settings for optimal performance
    initial_memory = configure_memory_settings()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    start_time = time.time()

    logger.info("Starting dataset creation with optimized parallel processing...")

    # Handle multiple input directories
    pycurt_dicom_dirs = []
    if hasattr(args.paths, 'pycurt_dir'):
        pycurt_dicom_dirs = [Path(args.paths.pycurt_dir)]
    elif hasattr(args.paths, 'pycurt_dirs'):
        pycurt_dicom_dirs = [Path(dir_path) for dir_path in args.paths.pycurt_dirs]
    else:
        logger.error("No pycurt directory specified in config.")
        return

    # Create output directory
    dataset_save_dir = Path(args.paths.dataset_dir)
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    # Get filtering criteria
    ignored_modalities_set = set(args.ignored_modalities)

    # Handle filtering based on new configuration structure
    # If filtering lists are empty, do not filter on those criteria
    use_axes_filter = False
    use_anatomies_filter = False
    use_sequences_filter = False

    accepted_axes_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'axes') and args.filtering.axes:
        accepted_axes_set = set(args.filtering.axes)
        use_axes_filter = True
    elif hasattr(args, 'accepted_axes') and args.accepted_axes:
        # Backward compatibility
        accepted_axes_set = set(args.accepted_axes)
        use_axes_filter = True

    accepted_anatomies_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'anatomies') and args.filtering.anatomies:
        accepted_anatomies_set = set(args.filtering.anatomies)
        use_anatomies_filter = True
    elif hasattr(args, 'accepted_anatomies') and args.accepted_anatomies:
        # Backward compatibility
        accepted_anatomies_set = set(args.accepted_anatomies)
        use_anatomies_filter = True

    accepted_sequences_set = None
    if hasattr(args, 'filtering') and hasattr(args.filtering, 'sequences') and args.filtering.sequences:
        accepted_sequences_set = set(args.filtering.sequences)
        use_sequences_filter = True

    # Log filtering criteria
    logger.info(f"Filtering with orientation: {args.orientation}")
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

    # Initialize BPLabeler if using custom labeling
    body_part_labeler = None
    use_custom_labeler = False
    if (hasattr(args, 'classification') and
            hasattr(args.classification, 'target') and
            args.classification.target == 'body_part' and
            hasattr(args.classification, 'custom_labeler') and
            args.classification.custom_labeler):
        # Import only if needed
        from paste import BPLabeler
        body_part_labeler = BPLabeler()
        use_custom_labeler = True
        logger.info("Using custom BPLabeler for body part classification")

    # Collect DICOM directories from all source directories - using parallel processing
    dicom_dirs = []
    logger.info(f"Starting parallel directory scanning for {len(pycurt_dicom_dirs)} source directories")

    # Use multiprocessing for directory scanning
    with ProcessPoolExecutor(max_workers=min(len(pycurt_dicom_dirs), multiprocessing.cpu_count())) as executor:
        dir_scan_futures = []
        for pycurt_dir in pycurt_dicom_dirs:
            future = executor.submit(
                scan_directory,
                pycurt_dir,
                ignored_modalities_set,
                use_axes_filter,
                accepted_axes_set,
                use_anatomies_filter,
                accepted_anatomies_set,
                use_sequences_filter,
                accepted_sequences_set
            )
            dir_scan_futures.append(future)

        # Collect directory scan results
        for future in as_completed(dir_scan_futures):
            dicom_dirs.extend(future.result())

    logger.info(f"{len(dicom_dirs)} total DICOM directories found after filtering.")
    log_memory_usage("After directory scanning")

    if len(dicom_dirs) == 0:
        logger.error("No DICOM directories found that match the filtering criteria. Check your configuration.")
        return

    # Determine optimal batch size and workers based on system resources
    num_workers = getattr(args, 'num_workers', get_optimal_workers())
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Calculate batch size - each worker gets at least 5 items but not more than 50
    items_per_worker = max(5, min(50, len(dicom_dirs) // num_workers))
    batch_size = items_per_worker * num_workers // 4  # Divide by 4 to create more batches than workers
    batch_size = max(5, min(batch_size, 100))  # Keep batch size reasonable

    logger.info(f"Processing with batch size of {batch_size} directories per batch")

    # Batch the directories
    batches = chunk_list(dicom_dirs, batch_size)

    # Process batches in parallel
    dataset = []

    tqdm.write(f"Starting parallel DICOM processing with {len(dicom_dirs)} directories in {len(batches)} batches...")

    # Extract necessary parameters from args as a dictionary for pickling
    args_dict = {
        'text_attributes': args.text_attributes,
        'numerical_attributes': args.numerical_attributes,
        'missing_term': args.missing_term,
        'orientation': args.orientation
    }

    # Add classification_target if available
    if hasattr(args, 'classification') and hasattr(args.classification, 'target'):
        args_dict['classification_target'] = args.classification.target
    else:
        args_dict['classification_target'] = 'sequence'

    # Process batches in parallel - using ThreadPoolExecutor instead of ProcessPoolExecutor
    # to avoid pickling issues with complex objects
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches for processing
        futures = []
        for batch in batches:
            future = executor.submit(
                process_dicom_batch,
                batch,
                args_dict,
                use_custom_labeler,
                body_part_labeler
            )
            futures.append(future)

        # Collect results as they complete with better progress tracking
        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches", ncols=100):
            try:
                batch_results = future.result()
                if batch_results:
                    dataset.extend(batch_results)

                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    log_progress(completed, len(futures), start_time)

                # Force garbage collection to free memory
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

    log_memory_usage("After DICOM processing")

    if not dataset:
        logger.error("No valid DICOM data found after processing. Check your data and filters.")
        return

    logger.info(f"Successfully processed {len(dataset)} DICOM directories.")

    # Prepare data for tokenization
    text_attrs = []
    for d in dataset:
        # text_attributes is already a string, use it directly
        text_attrs.append(d.text_attributes)

    # Collect all unique labels
    labels = set(d.label for d in dataset)
    label_dict = {modality: i for i, modality in enumerate(sorted(labels))}

    logger.info(f"Found {len(label_dict)} unique classes: {', '.join(sorted(labels))}")

    # Tokenize text data using optimized batched tokenization
    logger.info("Starting optimized text tokenization...")
    text_encoded = optimize_tokenization(text_attrs, args.training_params.tokenizer_name, batch_size=1000)
    log_memory_usage("After tokenization")

    # Optimize batch size for saving
    batch_size = min(1500, max(500, len(dataset) // 10))  # More appropriate batch sizing
    batch_dir = dataset_save_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Batch save in parallel with improved error handling
    buffer = []
    batch_index = 0
    save_futures = []
    batch_files = []

    logger.info(f"Saving dataset in batches of {batch_size} samples...")
    with ThreadPoolExecutor(max_workers=4) as save_executor:
        for i, d in enumerate(tqdm(dataset, desc="Encoding & preparing batches", ncols=100)):
            sample = encodedSample(
                img=d.img,
                input_ids=text_encoded["input_ids"][i],
                attention_mask=text_encoded["attention_mask"][i],
                numerical_attributes=d.numerical_attributes,
                label=label_dict[d.label],
            )
            buffer.append(sample)

            if len(buffer) == batch_size:
                # Submit this batch for saving
                future = save_executor.submit(save_batch, buffer.copy(), batch_index, batch_dir)
                save_futures.append(future)
                batch_index += 1
                buffer = []

                # Track progress periodically
                if batch_index % 5 == 0:
                    log_progress(i, len(dataset), start_time)

        # Handle any remaining items
        if buffer:
            future = save_executor.submit(save_batch, buffer, batch_index, batch_dir)
            save_futures.append(future)
            buffer = []

        # Wait for all saves to complete and collect batch files
        for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving batches", ncols=100):
            try:
                batch_file = future.result()
                batch_files.append(batch_file)
            except Exception as e:
                logger.error(f"Error saving batch: {e}")

    log_memory_usage("After batch saving")

    # More efficient batch merging by streaming through the batch files
    logger.info("Starting efficient batch merging...")
    all_batches = sorted(batch_dir.glob("dataset_batch_*.pkl"))

    # First count total samples in all batches
    total_samples = 0
    for batch_file in tqdm(all_batches, desc="Counting samples", ncols=100):
        try:
            with open(batch_file, "rb") as f:
                batch_data = pickle.load(f)
                total_samples += len(batch_data)
        except Exception as e:
            logger.error(f"Error reading batch file {batch_file}: {e}")

    logger.info(f"Total samples to merge: {total_samples}")

    # More memory-efficient approach to merge batches
    full_dataset_file = dataset_save_dir / "dataset.pkl"
    try:
        # Create a new file to hold the dataset
        with open(full_dataset_file, "wb") as f:
            # Initialize an empty list in the file
            full_dataset = []
            pickle.dump(full_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Now process each batch and append to the full dataset
        full_dataset = []
        for batch_file in tqdm(all_batches, desc="Merging batches", ncols=100):
            try:
                with open(batch_file, "rb") as f:
                    batch_data = pickle.load(f)
                    full_dataset.extend(batch_data)

                # Remove the batch file after processing to save disk space
                os.remove(batch_file)
            except Exception as e:
                logger.error(f"Error processing batch file {batch_file}: {e}")

        # Save the full dataset
        with open(full_dataset_file, "wb") as f:
            pickle.dump(full_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        logger.error(f"Error merging batches: {e}")
        # Try a simpler approach if the optimized one fails
        logger.info("Trying simpler batch merging approach...")

        full_dataset = []
        for batch_file in tqdm(all_batches, desc="Merging batches (fallback)", ncols=100):
            try:
                with open(batch_file, "rb") as f:
                    batch_data = pickle.load(f)
                    full_dataset.extend(batch_data)
            except Exception as e:
                logger.error(f"Error reading batch file {batch_file}: {e}")

        # Save the full dataset
        with open(full_dataset_file, "wb") as f:
            pickle.dump(full_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    log_memory_usage("After batch merging")

    # Clean up batch directory if it's empty
    if len(list(batch_dir.glob("*"))) == 0:
        try:
            batch_dir.rmdir()
        except:
            pass

    # Save the label dictionary
    with open(dataset_save_dir / "label_dict.json", "w") as f:
        json.dump(label_dict, f)

    # Generate statistics about the dataset
    label_counts = Counter([sample.label for sample in full_dataset])

    # Map numeric labels back to string labels for readability
    inv_label_dict = {v: k for k, v in label_dict.items()}
    readable_dist = {inv_label_dict[label_id]: count for label_id, count in label_counts.items()}

    stats = {
        "total_samples": len(full_dataset),
        "num_classes": len(label_dict),
        "class_distribution": readable_dist,
        "source_directories": [str(dir_path) for dir_path in pycurt_dicom_dirs],
        "classification_target": getattr(args.classification, 'target', 'sequence') if hasattr(args,
                                                                                               'classification') else 'sequence',
        "filtering": {
            "axes": list(accepted_axes_set) if use_axes_filter else [],
            "anatomies": list(accepted_anatomies_set) if use_anatomies_filter else [],
            "sequences": list(accepted_sequences_set) if use_sequences_filter else []
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "processing_time_seconds": time.time() - start_time
    }

    with open(dataset_save_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    total_time = time.time() - start_time
    logger.info(f"Successfully created dataset with {len(full_dataset)} samples in {total_time:.2f} seconds.")
    logger.info(f"Number of classes: {len(label_dict)}")

    # Print class distribution
    logger.info("Class distribution:")
    for label_id, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {inv_label_dict[label_id]}: {count} samples")

    logger.info(f"Dataset saved to {dataset_save_dir}")
    log_memory_usage("Final")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file.")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers (default: auto)")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    # Add num_workers if specified in command line
    if config_args.num_workers:
        args.num_workers = config_args.num_workers

    # Set up logging with timing information
    logger.remove()  # Remove default handler
    logger.add(sys.stderr,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    # Output system information
    logger.info(
        f"System information: {multiprocessing.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024 ** 3):.1f} GB RAM")

    # Run the dataset creation with optimized performance
    get_dataset(args)