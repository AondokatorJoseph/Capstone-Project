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
