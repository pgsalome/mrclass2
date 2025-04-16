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

def get_dataset(args: Namespace) -> None:
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file.")
    config_args = parser.parse_args()
    args = read_yaml_config(config_args.config_path)

    get_dataset(args)
