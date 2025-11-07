# Enhanced Brain Tumor Segmentation with 3D U-Net and Laplacian-of-Gaussian (LoG) Preprocessing

This project implements an **enhanced 3D brain tumor segmentation pipeline** using:

✅ **3D U-Net** for volumetric segmentation  
✅ **Laplacian of Gaussian (LoG) preprocessing** to enhance tumor edges  
✅ **Patch-wise training** for memory efficiency  
✅ **MRI multi-modal inputs** (T1, T1ce, T2, FLAIR)  
✅ **Dice + CrossEntropy loss**  

It is designed to work with datasets such as **BraTS** and includes preprocessing, training, and inference steps.

---

## 🚀 Features

- LoG preprocessing to highlight tumor borders  
- 3D U-Net with skip connections  
- Works with multimodal MRI scans  
- Patch extraction with augmentation  
- Sliding-window inference  
- Dice evaluation metrics  
- Easy to configure  

---

## 📁 Project Structure

```
Enhanced-Brain-Tumor-Segmentation/
├── data/
│   ├── raw/            # original .nii.gz files
│   ├── processed/      # preprocessed numpy volumes
│   └── splits/         # train.txt, val.txt, test.txt
├── notebooks/
│   └── Brain tumor segmentation project.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── datasets.py
│   ├── train.py
│   ├── infer.py
│   └── losses.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
├── configs/
│   └── default.yaml
└── README.md
```

---

## 📦 Installation

```
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image scikit-learn nibabel simpleitk tqdm pyyaml monai matplotlib
```

---

## 🧪 Dataset Format

Place MRI scans as:

```
data/raw/<patient_id>/
    ├── T1.nii.gz
    ├── T1ce.nii.gz
    ├── T2.nii.gz
    ├── FLAIR.nii.gz
    └── seg.nii.gz
```

Create:

```
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

Each file contains patient folder names.

---

## ⚙️ Configuration File (`configs/default.yaml`)

```
seed: 42
data:
  raw_dir: data/raw
  proc_dir: data/processed
  splits_dir: data/splits
  modalities: [T1, T1ce, T2, FLAIR]
  use_log: true
  log_sigma: 1.2
  log_mix_alpha: 0.5
train:
  patch_size: [128, 128, 128]
  batch_size: 2
  epochs: 300
  lr: 3e-4
model:
  in_channels: 4
  out_channels: 4
```

---

## 🧼 Preprocessing (LoG + Normalization)

- Resampling to standard spacing  
- Intensity clipping  
- Z-score normalization  
- Compute **Laplacian of Gaussian** edge map  
- Blend or concatenate LoG channels

---

## 🧠 Model: 3D U-Net

Architecture includes:

- Encoder with downsampling  
- Bottleneck  
- Decoder with skip connections  
- Final segmentation head  

---

## 🔥 Training

Command:

```
python src/train.py --config configs/default.yaml
```

Training includes:

- Dice + CE loss  
- Mixed precision (AMP)  
- Best checkpoint saving  

---

## 🔎 Inference

```
python src/infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --out_dir outputs/predictions
```

Performs:

- preprocessing  
- sliding window inference  
- post-processing  

---

## 📏 Metrics

- Dice Score  
- Per-class Dice for:
  - Whole Tumor (WT)
  - Tumor Core (TC)
  - Enhancing Tumor (ET)

---

## ✅ Results (placeholder)

| Class | Dice |
|-------|------|
| WT    | 0.90 |
| TC    | 0.86 |
| ET    | 0.83 |
| **Mean** | **0.86** |

---

## 🗒️ Tips

- Reduce patch size if GPU runs out of memory  
- Modify LoG sigma for clearer edges  
- Foreground sampling improves tumor detection  
- Use TensorBoard to monitor training  

---

## 📜 Citation

```
@article{unet2015,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger et al.},
  booktitle={MICCAI},
  year={2015}
}
```

---

