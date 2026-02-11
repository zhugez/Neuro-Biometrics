# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Updates:**
> - [2026-02-11] **v1 Major Fix:** Proper 2D reshape for ResNet embedder (was `unsqueeze(-1)` → width=1, now `reshape(B,C,H,W)`)
> - [2026-02-11] Fixed data split: stratified sample-split (was subject-split causing 0% centroid accuracy)
> - [2026-02-11] Added training augmentation (noise jitter + amplitude scaling), deeper projection head
> - [2026-02-11] Integrated **Mamba Selective Scan** into WaveNet denoiser (v2) for linear-time sequence modeling
> - [2026-02-11] Backup script: removed gdrive dependency, zip-only output for Kaggle

## 📖 Introduction

This repository contains the official implementation for the paper **"Enhancing EEG-based Biometrics with Mamba-augmented Denoising Autoencoders"**.

We propose a novel two-stage architecture:
1.  **Denoising Stage:** A WaveNet-based autoencoder augmented with a **Mamba Block** (State Space Model) to capture long-range temporal dependencies in EEG signals efficiently.
2.  **Verification Stage:** A ResNet-based embedder trained with **ArcFace / Multi-Similarity Loss** to extract identity-robust features.

## 🏗️ Architecture

```
Input EEG (B, C=4, T=800)
        │
        ▼
┌──────────────────────┐
│  WaveNet Denoiser    │  Stage 1: SI-SNR loss
│  (Dilated Conv1D)    │  30 epochs, CosineAnnealing
│  [+Mamba Block v2]   │
└──────────┬───────────┘
           │ denoised (B, C, T)
           ▼
┌──────────────────────┐
│  Reshape to 2D       │  (B, 4, 25, 32) for T=800
│  ResNet18/34 Backbone │  Conv2d 3×3 stride=1, no maxpool
│  Projection Head     │  Linear→ReLU→Dropout→Linear→BN
│  L2 Normalize        │
└──────────┬───────────┘
           │ embedding (B, 128)
           ▼
   ArcFace / MultiSimilarity Loss   Stage 2: metric learning
```

- **Denoiser:** WaveNet (Dilated Conv) + optional Mamba SSM (v2)
- **Embedder:** ResNet with proper 2D spatial input, deeper projection head
- **Loss:** SI-SNR (denoising) + ArcFace/MultiSimilarity (identity verification)
- **Augmentation:** Noise jitter + random amplitude scaling during Stage 2

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/zhugez/Neuro-Biometrics.git
cd Neuro-Biometrics

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset & Usage

### 1. Download Data
Dataset is hosted privately. Use the provided script to download (requires access):
```bash
python download_dataset.py
```

### 2. Training

```bash
# v1: Two-stage pipeline (WaveNet + ResNet)
python experiments/v1_two_stage_snr_0_5_10_20/main.py --epochs 30 --seeds 3

# v2: Mamba-augmented denoiser
python experiments/v2_mamba_denoiser/main.py

# Quick one-sample smoke test
python experiments/v1_two_stage_snr_0_5_10_20/main.py --one-sample
```

### 3. Backup Weights
Zip all checkpoints (saves to `/kaggle/working/` for Kaggle output):
```bash
python backup_full.py
```

## � Project Structure

```
Neuro-Biometrics/
├── experiments/
│   ├── v1_two_stage_snr_0_5_10_20/   # Baseline: WaveNet + ResNet
│   │   ├── main.py                    # Entry point
│   │   ├── model.py                   # WaveNet denoiser + ResNet embedder
│   │   ├── trainer.py                 # Two-stage training loop
│   │   ├── datapreprocessor.py        # EEG loading, preprocessing, noise gen
│   │   └── weights/                   # Saved checkpoints
│   └── v2_mamba_denoiser/             # Mamba-augmented variant
├── dataset/                           # EEG data (Filtered_Data, Segmented_Data)
├── backup_full.py                     # Zip & save weights
├── requirements.txt
└── README.md
```

## �📈 Results

<!-- RESULTS_TABLE_START -->
### v1: Two-Stage WaveNet + ResNet (3-seed mean ± std)

**Gaussian Noise (SNR 0/5/10/20 dB)**
| Model | P@1 | P@5 | SI-SNR (dB) | EER | AUROC | AUPR |
|---|---|---|---|---|---|---|
| ResNet34 + MultiSim | **0.9314 ± 0.007** | **0.9652** | 12.58 | **0.0379** | 0.8532 | 0.8558 |
| ResNet18 + MultiSim | 0.9281 ± 0.004 | 0.9641 | 12.57 | 0.0419 | 0.8483 | 0.8512 |
| ResNet34 + ArcFace | 0.9265 ± 0.003 | **0.9730** | 12.58 | 0.0717 | **0.8620** | **0.8647** |

**Powerline Noise (50 Hz)**
| Model | P@1 | P@5 | SI-SNR (dB) | EER | AUROC | AUPR |
|---|---|---|---|---|---|---|
| ResNet34 + MultiSim | **0.9686 ± 0.003** | 0.9828 | 37.89 | **0.0189** | **0.9081** | **0.9104** |
| ResNet18 + MultiSim | 0.9608 ± 0.004 | 0.9798 | 37.73 | 0.0225 | 0.8691 | 0.8794 |
| ResNet34 + ArcFace | 0.9667 ± 0.003 | **0.9887** | 37.89 | 0.0372 | 0.8946 | 0.8976 |

**EMG Noise (20–80 Hz)**
| Model | P@1 | P@5 | SI-SNR (dB) | EER | AUROC | AUPR |
|---|---|---|---|---|---|---|
| ResNet34 + MultiSim | **0.9529 ± 0.002** | 0.9770 | 14.37 | **0.0238** | 0.8728 | 0.8819 |
| ResNet18 + MultiSim | 0.9449 ± 0.006 | 0.9742 | 14.37 | 0.0277 | 0.8570 | 0.8695 |
| ResNet34 + ArcFace | 0.9454 ± 0.007 | **0.9801** | 14.37 | 0.0515 | **0.8827** | **0.8895** |

> **Key findings:**
> - **ResNet34 + MultiSimilarity** gives best P@1 across all noise types
> - **Powerline noise** is easiest to denoise (SI-SNR 37.89 dB) → highest P@1 (96.86%)
> - **ArcFace** trades higher EER for better P@5 and AUROC
> - All models evaluated on **stratified sample split** with 3 random seeds
<!-- RESULTS_TABLE_END -->

*(Results based on Subject-Disjoint protocol).*

## 📜 Citation

If you use this code, please cite our paper:

```bibtex
@article{zhugez2026neurobiometrics,
  title={Neuro-Biometrics: Efficient EEG Denoising via State Space Models},
  author={Ly Ngoc Vu and Huynh Cong Bang},
  year={2026}
}
```

## 🛡️ License
MIT License. For research purposes only.
