import dicom2nifti
import nibabel as nib
import pydicom
import os
import pickle
import json
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from transformers import AutoTokenizer
from loguru import logger
from typing import Union, List, Tuple, Dict
from utils.io import read_yaml_config
from utils.dataclass import dicomData, encodedSample
from argparse import ArgumentParser, Namespace
from pycurtv2.processing.common import get_n_slice, resize_or_pad_image
from pycurtv2.converters.dicom import DicomConverter
from tqdm import tqdm
import glob

def process_dicom_dir(dicom_dir: Path, args) -> Union[dicomData, None]:
    try:
        slices = glob.glob(dicom_dir + "/*")
        if not slices:
            tqdm.write(f"[ERROR] No DICOM slices found in {dicom_dir}. Skipping.")
            return None

        dicom_slice = str(slices[len(slices) // 2])
        tqdm.write(f"Started image format conversion for {dicom_dir}")
        DicomConverter(toConvert=dicom_dir).convert_ps()
        nifti_output_path = dicom_dir + ".nii.gz"

        np_image = resize_or_pad_image(get_n_slice(nifti_output_path, dicom_slice))

        metadata = pydicom.dcmread(dicom_slice, stop_before_pixels=True, force=True)

        text_attributes_list = [
            f"{attribute} is {getattr(metadata, attribute)}"
            if hasattr(metadata, attribute) and getattr(metadata, attribute) is not None
            else args.missing_term
            for attribute in args.text_attributes
        ]
        text_attribute = ", ".join(text_attributes_list)

        numerical_attributes_list = [
            int(getattr(metadata, attribute))
            if hasattr(metadata, attribute) and getattr(metadata, attribute) is not None
            else -1
            for attribute in args.numerical_attributes
        ]

        label = Path(dicom_slice).parts[-3].replace('-DER', '')

        dicom_data = dicomData(
            img=np_image,
            text_attributes=text_attribute,
            numerical_attributes=numerical_attributes_list,
            label=label
        )

        del np_image, text_attributes_list, numerical_attributes_list, metadata
        try:
            os.remove(nifti_output_path)
        except FileNotFoundError:
            tqdm.write(f"[WARNING] No NIfTI file found at {nifti_output_path}.")
        gc.collect()

        return dicom_data

    except Exception as e:
        tqdm.write(f"[ERROR] Exception processing {dicom_dir}: {e}")
        return None

def get_dataset_old(args: Namespace) -> None:
    logger.warning("Label extraction not implemented. Labels set to 'NA'.")
    logger.warning("Image extraction not implemented. Images set to 'NAN'.")

    pycurt_dicom_dir =Path(args.paths.pycurt_dir)

    dataset_save_dir = Path(args.paths.dataset_dir)


    ignored_modalities_set = set(args.ignored_modalities)
    accepted_anatomies_set = set(args.accepted_anatomies)
    accepted_axes_set = set(args.accepted_axes)

    logger.debug(f"Filtering with orientation={args.orientation}, anatomy={args.accepted_anatomies}, modality={args.ignored_modalities}")

    dicom_dirs = [
        d for d in glob.glob(str(pycurt_dicom_dir) + "/*/*/*/*/*/*")
        if Path(d).is_dir()
           and str(Path(d).parts[-2]).split("-")[0] not in ignored_modalities_set
           and Path(d).parts[-3] in accepted_anatomies_set
           and Path(d).parts[-4] in accepted_axes_set
           and "RGB" not in Path(d).parts[-3]
    ]

    logger.info(f"{len(dicom_dirs)} total DICOM directories found after filtering.")
    tqdm.write(f"Starting parallel DICOM processing with {len(dicom_dirs)} directories...")

    dataset = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_dicom_dir, d, args) for d in dicom_dirs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM dirs", ncols=100):
            result = f.result()
            if result is not None:
                dataset.append(result)

    if not dataset:
        logger.error("No valid DICOM data found. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    text_attrs = [d.text_attributes for d in dataset]
    labels = set(d.label for d in dataset)
    label_dict = {modality: i for i, modality in enumerate(labels)}

    text_encoded = tokenizer(
        text_attrs, add_special_tokens=True, return_attention_mask=True, padding="longest", truncation=True
    )

    # Split into batches and save
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
            label=label_dict[d.label],
        )
        buffer.append(sample)

        if len(buffer) == batch_size:
            batch_file = batch_dir / f"dataset_batch_{i // batch_size}.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(buffer, f)
            buffer = []

    if buffer:
        batch_file = batch_dir / f"dataset_batch_final.pkl"
        with open(batch_file, "wb") as f:
            pickle.dump(buffer, f)

    # Merge all batches
    all_batches = sorted(batch_dir.glob("dataset_batch_*.pkl"))
    full_dataset = []
    for batch_file in tqdm(all_batches, desc="Merging batches", ncols=100):
        with open(batch_file, "rb") as f:
            full_dataset.extend(pickle.load(f))

    with open(dataset_save_dir / "dataset.pkl", "wb") as f:
        pickle.dump(full_dataset, f)

    with open(dataset_save_dir / "label_dict.json", "w") as f:
        json.dump(label_dict, f)

    logger.info(f"Successfully created dataset with {len(full_dataset)} samples.")


def get_dataset(args: Namespace) -> None:
    """
    Process DICOM directories to create a dataset for body part or sequence classification.
    Supports multiple input directories and flexible filtering criteria.

    Args:
        args: Configuration namespace containing dataset parameters
    """
    logger.info("Starting dataset creation...")

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

    # Collect DICOM directories from all source directories
    dicom_dirs = []
    for pycurt_dir in pycurt_dicom_dirs:
        logger.info(f"Scanning directory: {pycurt_dir}")

        # Build glob pattern based on directory structure
        glob_pattern = str(pycurt_dir) + "/*/*/*/*/*/*"

        # Function to filter directories based on criteria
        def filter_dir(d):
            d_path = Path(d)
            if not d_path.is_dir():
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
        dicom_dirs.extend(dirs)
        logger.info(f"Found {len(dirs)} valid directories in {pycurt_dir}")

    logger.info(f"{len(dicom_dirs)} total DICOM directories found after filtering.")

    if len(dicom_dirs) == 0:
        logger.error("No DICOM directories found that match the filtering criteria. Check your configuration.")
        return

    tqdm.write(f"Starting parallel DICOM processing with {len(dicom_dirs)} directories...")

    # Process directories
    dataset = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Define a wrapper function to handle the labeling
        def process_with_labeling(dicom_dir):
            result = process_dicom_dir(dicom_dir, args)
            if result is not None:
                # Determine the label based on the classification target
                d_path = Path(dicom_dir)

                if hasattr(args, 'classification') and hasattr(args.classification, 'target'):
                    if args.classification.target == 'body_part':
                        if use_custom_labeler:
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
                    elif args.classification.target == 'sequence':
                        # Use sequence (folder name) as label
                        sequence_name = str(d_path.parts[-2])
                        result.label = sequence_name
                else:
                    # Default: use sequence (folder name) as label
                    sequence_name = str(d_path.parts[-2])
                    result.label = sequence_name

            return result

        futures = [executor.submit(process_with_labeling, d) for d in dicom_dirs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM dirs", ncols=100):
            result = f.result()
            if result is not None:
                dataset.append(result)

    if not dataset:
        logger.error("No valid DICOM data found after processing. Check your data and filters.")
        return

    logger.info(f"Successfully processed {len(dataset)} DICOM directories.")

    # Initialize tokenizer for text encoding
    tokenizer = AutoTokenizer.from_pretrained(args.training_params.tokenizer_name)

    # Prepare data for tokenization
    text_attrs = []
    for d in dataset:
        # Combine all text attributes into a single string
        combined_text = " ".join([f"{key}: {value}" for key, value in d.text_attributes.items()])
        text_attrs.append(combined_text)

    # Collect all unique labels
    labels = set(d.label for d in dataset)
    label_dict = {modality: i for i, modality in enumerate(sorted(labels))}

    logger.info(f"Found {len(label_dict)} unique classes: {', '.join(sorted(labels))}")

    # Tokenize text data
    text_encoded = tokenizer(
        text_attrs,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="longest",
        truncation=True
    )

    # Split into batches and save
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
            label=label_dict[d.label],
        )
        buffer.append(sample)

        if len(buffer) == batch_size:
            batch_file = batch_dir / f"dataset_batch_{i // batch_size}.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(buffer, f)
            buffer = []

    if buffer:
        batch_file = batch_dir / f"dataset_batch_final.pkl"
        with open(batch_file, "wb") as f:
            pickle.dump(buffer, f)

    # Merge all batches
    all_batches = sorted(batch_dir.glob("dataset_batch_*.pkl"))
    full_dataset = []
    for batch_file in tqdm(all_batches, desc="Merging batches", ncols=100):
        with open(batch_file, "rb") as f:
            full_dataset.extend(pickle.load(f))

    # Save the full dataset
    with open(dataset_save_dir / "dataset.pkl", "wb") as f:
        pickle.dump(full_dataset, f)

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
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open(dataset_save_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Successfully created dataset with {-len(full_dataset)} samples.")
    logger.info(f"Number of classes: {len(label_dict)}")

    # Print class distribution
    logger.info("Class distribution:")
    for label_id, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {inv_label_dict[label_id]}: {count} samples")

    logger.info(f"Dataset saved to {dataset_save_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file.")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    get_dataset(args)
