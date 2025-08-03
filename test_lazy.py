#!/usr/bin/env python3
"""
Test script to verify lazy loading works correctly.
Save as test_lazy.py and run from project root.
"""

import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Direct imports to avoid circular dependencies
from utils.lazy_dataset import LazyMRIDataset, LazyMRIDatasetWrapper


def prepare_transforms_simple(img_size=224):
    """Simple transform preparation to avoid circular import."""
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Validation/test transforms without augmentation
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return {
        "train": train_transforms,
        "val": val_transforms,
        "test": val_transforms
    }


def test_lazy_loading():
    """Test the lazy loading implementation."""

    # Create a minimal config
    config = {
        "data": {
            "dataset_dir": "/media/e210/portable_hdd/d_bodypart_final_cleaned",
            "orientation": "SAG",  # or "TRA", "COR"
            "img_size": 224,
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": False,
            "shuffle": True,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        },
        "seed": 42
    }

    print("Testing lazy dataset loading...")
    print(f"Dataset directory: {config['data']['dataset_dir']}")
    print(f"Orientation: {config['data']['orientation']}")

    # Prepare transforms
    transforms_dict = prepare_transforms_simple(config['data']['img_size'])

    # Create wrapper and datasets
    wrapper = LazyMRIDatasetWrapper(config)
    datasets, label_dict = wrapper.create_datasets(transforms_dict)

    print(f"\nLabel dictionary ({len(label_dict)} classes):")
    for i, (label, idx) in enumerate(list(label_dict.items())[:5]):
        print(f"  {label}: {idx}")
    if len(label_dict) > 5:
        print(f"  ... and {len(label_dict) - 5} more classes")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(datasets['train'])}")
    print(f"  Val: {len(datasets['val'])}")
    print(f"  Test: {len(datasets['test'])}")

    # Test loading a few samples
    print("\nTesting sample loading...")
    train_dataset = datasets['train']

    # Load first 5 samples
    for i in range(min(5, len(train_dataset))):
        try:
            sample = train_dataset[i]
            print(f"\nSample {i}:")
            print(f"  Image shape: {sample['img'].shape}")
            print(f"  Input IDs length: {len(sample['input_ids'])}")
            print(f"  Attention mask length: {len(sample['attention_mask'])}")
            print(f"  Numerical attributes: {len(sample['numerical_attributes'])} features")
            print(f"  Label: {sample['label'].item()}")
        except Exception as e:
            print(f"\nError loading sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Test with DataLoader
    print("\nTesting DataLoader with batch loading...")
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )

    # Get one batch
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx} shapes:")
            print(f"  Images: {batch['img'].shape}")
            print(f"  Input IDs: {batch['input_ids'].shape}")
            print(f"  Attention masks: {batch['attention_mask'].shape}")
            print(f"  Numerical attributes: {batch['numerical_attributes'].shape}")
            print(f"  Labels: {batch['label'].shape}")

            # Print memory usage
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"  Current memory usage: {mem_mb:.1f} MB")
            except:
                pass

            if batch_idx >= 2:  # Test a few batches
                break

    except Exception as e:
        print(f"\nError during batch loading: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Lazy loading test completed!")
    print("=" * 50)


if __name__ == "__main__":
    # Add the encodedSample import at module level
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import encodedSample before running test
    from utils.dataclass import encodedSample

    test_lazy_loading()