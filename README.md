
# AAsU-Net: Adaptive Anisotropic Convolutional Network for Renal Neoplasm Segmentation

AAsU-Net is a 3D medical image segmentation framework for kidney and renal tumor segmentation, based on the paper *AAsU-Net: Adaptive anisotropic convolutional network for renal neoplasm segmentation*.

## Datasets

AAsU-Net is designed for experiments on publicly available kidney tumor segmentation datasets:

- **KiTS19** – Kidney and Kidney Tumor Segmentation Challenge 2019  
  Dataset link: https://kits19.grand-challenge.org/

- **KiTS21** – Kidney and Kidney Tumor Segmentation Challenge 2021  
  Dataset link: https://kits-challenge.org/kits21/

Before running the training scripts, please organize the datasets in the following structure:

```text
data/
├── kits19_raw/
│   ├── case_00000/
│   │   ├── imaging.nii.gz
│   │   └── segmentation.nii.gz
│   ├── case_00001/
│   │   ├── imaging.nii.gz
│   │   └── segmentation.nii.gz
│   └── ...
├── kits21_raw/
│   ├── case_00000/
│   │   ├── imaging.nii.gz
│   │   └── segmentation.nii.gz
│   └── ...
````

## Project Structure

The repository is organized as follows:

```text
AAsU-Net/
├── AAsU-Net/
├── configs/
├── docs/
├── scripts/
├── tests/
├── README.md
└── requirements.txt
```

## Environment Setup

To run the project, first create a Python environment and install the required dependencies:

```bash
git clone https://github.com/coyld/AAsU-Net.git
cd AAsU-Net
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Prepare the dataset using the provided preprocessing script:

```bash
python scripts/prepare_kits.py --input-dir data/kits19_raw --output-dir data/kits19_preprocessed --config configs/kits19_train.yaml
```

Create the train/validation split:

```bash
python scripts/make_splits.py --manifest data/kits19_preprocessed/manifest.jsonl --output-dir data/kits19_preprocessed/splits --val-ratio 0.2 --seed 42
```

## Training

Run the training script with the KiTS19 configuration:

```bash
python scripts/train.py --config configs/kits19_train.yaml --train-manifest data/kits19_preprocessed/splits/train.jsonl --val-manifest data/kits19_preprocessed/splits/val.jsonl
```



## Notes

Please adjust the dataset paths and configuration files according to your local environment before training or testing.

## Citation

If you use this project, please cite the original paper:

**AAsU-Net: Adaptive anisotropic convolutional network for renal neoplasm segmentation**

```

