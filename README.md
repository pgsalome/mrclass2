# MR Sequence Classifier

A deep learning repository for classifying MRI sequences using multimodal data (image, text metadata, and numerical parameters).

## Overview

This repository implements a multimodal classifier for MRI sequence classification. The model combines three types of input:

1. **Image data**: 2D mid-slice images from MRI series
2. **Text metadata**: Series descriptions and protocol names
3. **Numerical parameters**: Acquisition parameters like TE, TR, flip angle, etc.

The architecture features modular encoders for each modality and configurable fusion strategies.

## Features

- Multimodal deep learning (image, text, numerical features)
- Multiple image encoder options (ResNet, DenseNet, EfficientNet)
- Support for RadImageNet pre-trained weights
- Transformer-based text encoding (BERT, DistilBERT)
- Various fusion strategies (concatenation, attention, gating)
- Advanced medical image transforms via MONAI
- MRI-specific intensity normalization with class-specific options
- MRI foreground-specific standardization
- One-vs-Rest (OVR) loss with class balancing
- Classification threshold support
- Support for hierarchical classification
- Comprehensive evaluation metrics and visualizations
- Experiment management with Weights & Biases
- Configurable hyperparameter search

## Installation

```bash
# Clone the repository
git clone https://github.com/username/mr-sequence-classifier.git
cd mr-sequence-classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- transformers
- torchvision
- scikit-learn
- pandas
- matplotlib
- seaborn
- wandb
- tqdm
- MONAI (for medical imaging transforms)
- intensity-normalization (optional, for advanced MRI normalization)

## Usage

### Data Preparation

The repository expects preprocessed data in the format produced by the preprocessing scripts. The data should be organized as follows:

```
data/
├── processed/
│   ├── bootstrapped_dataset.pkl
│   └── bootstrapped_label_dict.json
```

### Training

To train a model with the default configuration:

```bash
python train.py --config config/default.json
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_dir saved_models/run_YYYYMMDD_HHMMSS_config0
```

### Running Multiple Experiments

To run experiments with different configurations:

```bash
# Generate experiment configurations
python run_experiments.py --base_config config/default.json --output_dir config/experiments --generate_only

# Run all experiments
python run_experiments.py --base_config config/default.json --output_dir config/experiments
```

To use a custom parameter grid:

```bash
python run_experiments.py --base_config config/default.json --custom_grid config/custom_grid.json
```

## Configuration

The model and training parameters are controlled through a JSON configuration file. The main configuration sections are:

- **data**: Dataset paths, preprocessing, and augmentation
- **model**: Model architecture (image encoder, text encoder, numeric encoder, fusion)
- **training**: Training parameters (optimizer, scheduler, loss function)
- **logging**: Logging and visualization options

See `config/default.json` for a complete example.

### Medical Image Transforms

The repository supports specialized medical image transformations via MONAI. To enable them:

```json
"data": {
  "use_medical_transforms": true
}
```

### Intensity Normalization

The repository supports MRI-specific intensity normalization methods through the `intensity-normalization` package. To enable it:

```json
"intensity_normalization": {
  "enabled": true,
  "method": "zscore",
  "class_specific": {
    "enabled": false
  }
}
```

For class-specific normalization (different methods for different MR sequences):

```json
"intensity_normalization": {
  "enabled": true,
  "method": null,
  "class_specific": {
    "enabled": true,
    "T1": "zscore",
    "T2": "whitestripe",
    "FLAIR": "kde"
  }
}
```

### One-vs-Rest (OVR) Loss and Thresholding

The repository supports One-vs-Rest loss with classification thresholding:

```json
"training": {
  "loss": {
    "type": "ovr",
    "threshold": 0.5
  }
}
```

Setting a threshold value allows the model to assign samples to an "unclassified" category when confidence is below the threshold.

### RadImageNet Pre-training

To use RadImageNet pre-trained weights for the image encoder:

```json
"model": {
  "img_encoder": {
    "name": "radimagenet",
    "weights_path": "path/to/radimagenet_weights.pth"
  }
}
```

## Weights & Biases Integration

The repository integrates with Weights & Biases for experiment tracking. To use it:

1. Set `logging.wandb.enabled` to `true` in the configuration.
2. Configure your project and entity in the configuration or set them up with `wandb login`.

```json
"wandb": {
  "enabled": true,
  "project": "mr-sequence-classification",
  "entity": "your-username",
  "name": "experiment-name",
  "tags": ["v1", "multimodal"],
  "notes": "Testing different image encoders"
}
```

## Project Structure

```
mr_sequence_classifier/
├── config/
│   ├── default.json             # Default configuration
│   └── experiments/             # Experiment-specific configs
├── data/
│   ├── processed/               # Processed datasets
│   └── raw/                     # Raw DICOM data
├── logs/                        # Training logs
├── models/                      # Model definitions
│   ├── img_encoder.py           # Image encoders (CNN)
│   ├── txt_encoder.py           # Text encoders (BERT, etc.)
│   ├── num_encoder.py           # Numerical feature encoders
│   ├── fusion.py                # Fusion strategies
│   └── classifier.py            # Full multimodal classifier model
├── saved_models/                # Saved model checkpoints
├── utils/
│   ├── dataclass.py             # Data structures
│   ├── io.py                    # I/O utilities
│   ├── metrics.py               # Evaluation metrics
│   ├── visualize.py             # Visualization utilities
│   ├── loss.py                  # Loss functions
│   ├── rad_imagenet.py          # RadImageNet utilities
│   ├── medical_transforms.py    # Medical image transforms
│   └── intensity_normalization.py # MRI normalization
├── data_loader.py               # Data loading and preprocessing
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Prediction script for new data
└── run_experiments.py           # Script to run multiple experiments
```

## Extending the Repository

### Adding New Image Encoders

To add a new image encoder, extend the `_get_backbone` method in `models/img_encoder.py`.

### Adding New Fusion Methods

To add a new fusion strategy, create a new module in `models/fusion.py` and update the `get_fusion_module` function.

### Adding New Loss Functions

To add a new loss function, extend the `utils/loss.py` module and update the `get_loss_function` function.

### Adding New Normalization Methods

To add a new intensity normalization method, update the `IntensityNormalizer` class in `utils/intensity_normalization.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon methods and architectures from the following works:

- Deep multi-task learning and random forest for series classification by pulse sequence type and orientation (Helm et al., Neuroradiology 2022)
- Automatic classification of prostate MR series type using image content and metadata (Krishnaswamy et al., MIDL 2024)
- MRISeqClassifier: A Deep Learning Toolkit for Precise MRI Sequence Classification (Pan et al., medRxiv 2024)
- RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning (Mei et al., Radiology: Artificial Intelligence 2022)
- MONAI: Medical Open Network for AI (Cardoso et al., 2022)
- Intensity Normalization of MR Images (Reinhold et al., SPIE Medical Imaging 2019)
# MR Sequence Classifier

A deep learning repository for classifying MRI sequences using multimodal data (image, text metadata, and numerical parameters).

## Overview

This repository implements a multimodal classifier for MRI sequence classification. The model combines three types of input:

1. **Image data**: 2D mid-slice images from MRI series
2. **Text metadata**: Series descriptions and protocol names
3. **Numerical parameters**: Acquisition parameters like TE, TR, flip angle, etc.

The architecture features modular encoders for each modality and configurable fusion strategies.

## Features

- Multimodal deep learning (image, text, numerical features)
- Multiple image encoder options (ResNet, DenseNet, EfficientNet)
- Transformer-based text encoding (BERT, DistilBERT)
- Various fusion strategies (concatenation, attention, gating)
- MRI-specific intensity normalization with class-specific options
- Support for hierarchical classification
- Comprehensive evaluation metrics and visualizations
- Experiment management with Weights & Biases
- Configurable hyperparameter search

## Installation

```bash
# Clone the repository
git clone https://github.com/username/mr-sequence-classifier.git
cd mr-sequence-classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- transformers
- torchvision
- scikit-learn
- pandas
- matplotlib
- seaborn
- wandb
- tqdm
- intensity-normalization (optional, for advanced MRI normalization)

## Usage

### Data Preparation

The repository expects preprocessed data in the format produced by the preprocessing scripts. The data should be organized as follows:

```
data/
├── processed/
│   ├── bootstrapped_dataset.pkl
│   └── bootstrapped_label_dict.json
```

### Training

To train a model with the default configuration:

```bash
python train.py --config config/default.json
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_dir saved_models/run_YYYYMMDD_HHMMSS_config0
```

### Running Multiple Experiments

To run experiments with different configurations:

```bash
# Generate experiment configurations
python run_experiments.py --base_config config/default.json --output_dir config/experiments --generate_only

# Run all experiments
python run_experiments.py --base_config config/default.json --output_dir config/experiments
```

To use a custom parameter grid:

```bash
python run_experiments.py --base_config config/default.json --custom_grid config/custom_grid.json
```

## Configuration

The model and training parameters are controlled through a JSON configuration file. The main configuration sections are:

- **data**: Dataset paths, preprocessing, and augmentation
- **model**: Model architecture (image encoder, text encoder, numeric encoder, fusion)
- **training**: Training parameters (optimizer, scheduler, loss function)
- **logging**: Logging and visualization options

See `config/default.json` for a complete example.

### Intensity Normalization

The repository supports MRI-specific intensity normalization methods through the `intensity-normalization` package. To enable it:

```json
"intensity_normalization": {
  "enabled": true,
  "method": "zscore",
  "class_specific": {
    "enabled": false
  }
}
```

For class-specific normalization (different methods for different MR sequences):

```json
"intensity_normalization": {
  "enabled": true,
  "method": null,
  "class_specific": {
    "enabled": true,
    "T1": "zscore",
    "T2": "whitestripe",
    "FLAIR": "kde"
  }
}
```

## Weights & Biases Integration

The repository integrates with Weights & Biases for experiment tracking. To use it:

1. Set `logging.wandb.enabled` to `true` in the configuration.
2. Configure your project and entity in the configuration or set them up with `wandb login`.

```json
"wandb": {
  "enabled": true,
  "project": "mr-sequence-classification",
  "entity": "your-username",
  "name": "experiment-name",
  "tags": ["v1", "multimodal"],
  "notes": "Testing different image encoders"
}
```

## Project Structure

```
mr_sequence_classifier/
├── config/
│   ├── default.json             # Default configuration
│   └── experiments/             # Experiment-specific configs
├── data/
│   ├── processed/               # Processed datasets
│   └── raw/                     # Raw DICOM data
├── logs/                        # Training logs
├── models/                      # Model definitions
│   ├── img_encoder.py           # Image encoders (CNN)
│   ├── txt_encoder.py           # Text encoders (BERT, etc.)
│   ├── num_encoder.py           # Numerical feature encoders
│   ├── fusion.py                # Fusion strategies
│   └── classifier.py            # Full multimodal classifier model
├── saved_models/                # Saved model checkpoints
├── utils/
│   ├── dataclass.py             # Data structures
│   ├── io.py                    # I/O utilities
│   ├── metrics.py               # Evaluation metrics
│   ├── visualize.py             # Visualization utilities
│   └── intensity_normalization.py # MRI normalization
├── data_loader.py               # Data loading and preprocessing
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Prediction script for new data
└── run_experiments.py           # Script to run multiple experiments
```

## Extending the Repository

### Adding New Image Encoders

To add a new image encoder, extend the `_get_backbone` method in `models/img_encoder.py`.

### Adding New Fusion Methods

To add a new fusion strategy, create a new module in `models/fusion.py` and update the `get_fusion_module` function.

### Adding New Normalization Methods

To add a new intensity normalization method, update the `IntensityNormalizer` class in `utils/intensity_normalization.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon methods and architectures from the following works:

- Deep multi-task learning and random forest for series classification by pulse sequence type and orientation (Helm et al., Neuroradiology 2022)
- Automatic classification of prostate MR series type using image content and metadata (Krishnaswamy et al., MIDL 2024)
- MRISeqClassifier: A Deep Learning Toolkit for Precise MRI Sequence Classification (Pan et al., medRxiv 2024)