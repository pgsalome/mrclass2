#!/bin/bash

# Create main directory structure
mkdir -p mrclass2/config/experiments
mkdir -p mrclass2/data/processed
mkdir -p mrclass2/data/raw
mkdir -p mrclass2/logs
mkdir -p mrclass2/models
mkdir -p mrclass2/saved_models
mkdir -p mrclass2/utils

# Create model files
touch mrclass2/models/__init__.py
touch mrclass2/models/img_encoder.py
touch mrclass2/models/txt_encoder.py
touch mrclass2/models/num_encoder.py
touch mrclass2/models/fusion.py
touch mrclass2/models/classifier.py

# Create utility files
touch mrclass2/utils/__init__.py
touch mrclass2/utils/dataclass.py
touch mrclass2/utils/io.py
touch mrclass2/utils/metrics.py
touch mrclass2/utils/visualize.py
touch mrclass2/utils/loss.py
touch mrclass2/utils/rad_imagenet.py
touch mrclass2/utils/medical_transforms.py
touch mrclass2/utils/intensity_normalization.py

# Create main application files
touch mrclass2/__init__.py
touch mrclass2/data_loader.py
touch mrclass2/train.py
touch mrclass2/evaluate.py
touch mrclass2/predict.py
touch mrclass2/run_experiments.py

# Create configuration files
touch mrclass2/config/default.json
touch mrclass2/config/custom_grid.json

# Create requirements and README
touch mrclass2/requirements.txt
touch mrclass2/README.md

echo "Repository structure created successfully!"