# Capstone Project: CyclotekAI

## Overview
This project is part of the INFO 6156 Capstone Project at Fanshawe College. The goal is to develop an AI application using the COCO dataset.

## Directory Structure
```
main_project/
│
├── datasets/
│   └── coco/
│       ├── train2017.zip
│       └── annotations_trainval2017.zip
│
├── notebooks/
│   └── INFO6156_CapstoneProject_CyclotekAI_COLAB_r0.ipynb
│
├── src/
│   ├── data/
│   │   └── download.py
│   ├── models/
│   │   └── model.py
│   ├── utils/
│   │   └── utils.py
│   └── main.py
│
├── tests/
│   └── test_download.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
## Model Files
The trained models are not included in this repository due to size limitations (>200MB). 

### Training New Models
1. Create the required directory:
   ```bash
   mkdir -p models/saved_models
   ```

2. Train the model:
   ```bash
   python src/models/training.py
   ```

### Accessing Pre-trained Models
Contact project maintainers for access to pre-trained models:
- waste_classifier.weights.h5 (214MB)
- waste_classifier_full.keras (214MB)

Place these files in the `models/saved_models/` directory after obtaining them.

## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
To download and extract the dataset, run:
```sh
python src/data/download.py
```

## License
This project is licensed under the MIT License.
