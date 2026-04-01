# AAsU-Net

AAsU-Net is a 3D medical image segmentation framework for kidney and renal tumor segmentation, based on the paper *AAsU-Net ：Adaptive anisotropic convolutional net-work for renal neoplasm segmentation*.

## 1.Features

- Adaptive anisotropic convolution (AAs-conv)
- 6-stage 3D U-Net backbone
- Cross-scale feature fusion (CSFF)
- Deep supervision
- Training and inference pipeline for KiTS datasets

## 2.Project Structure

```text
AAsU-Net/
├─ aasunet/
├─ configs/
├─ docs/
├─ scripts/
├─ tests/
├─ requirements.txt
└─ README.md

## 3. Installation

```bash
git clone <your-repo-url>
cd AAsU-Net
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Data Preparation

### KiTS19 
```text
data/kits19_raw/
├─ case_00000/
│  ├─ imaging.nii.gz
│  └─ segmentation.nii.gz
├─ case_00001/
│  ├─ imaging.nii.gz
│  └─ segmentation.nii.gz
└─ ...
```
### KiTS21 
```text
data/kits21_raw/
├─ case_00000/
│  ├─ imaging.nii.gz
│  └─ segmentation.nii.gz
└─ ...
```

Prepare the dataset：

```bash
python scripts/prepare_kits.py --input-dir data/kits19_raw --output-dir data/kits19_preprocessed --config configs/kits19_train.yaml
```

Create train/validation splits:

```bash
python scripts/make_splits.py --manifest data/kits19_preprocessed/manifest.jsonl --output-dir data/kits19_preprocessed/splits --val-ratio 0.2 --seed 42
```

---

## 5. Training

```bash
python scripts/train.py --config configs/kits19_train.yaml --train-manifest data/kits19_preprocessed/splits/train.jsonl --val-manifest data/kits19_preprocessed/splits/val.jsonl
```




##  Citation

If you use this project, please cite the original paper:

AAsU-Net: Adaptive anisotropic convolutional network for renal neoplasm segmentation


