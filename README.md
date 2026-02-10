# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Updates:**
> - [2026-02-11] Integrated **Mamba Selective Scan** into WaveNet denoiser for linear-time sequence modeling.
> - [2026-02-11] Implemented **Subject-Disjoint Splitting** to prevent data leakage.

## 📖 Introduction

This repository contains the official implementation for the paper **"Enhancing EEG-based Biometrics with Mamba-augmented Denoising Autoencoders"**.

We propose a novel two-stage architecture:
1.  **Denoising Stage:** A WaveNet-based autoencoder augmented with a **Mamba Block** (State Space Model) to capture long-range temporal dependencies in EEG signals efficiently.
2.  **Verification Stage:** A ResNet-based embedder trained with **ArcFace / Multi-Similarity Loss** to extract identity-robust features.

## 🏗️ Architecture

![Architecture](https://via.placeholder.com/800x300?text=WaveNet+Denoiser+%2B+Mamba+Block+%2B+ArcFace+Head)

- **Backbone:** WaveNet (Dilated Convolutions) + Mamba (SSM).
- **Loss Functions:** SI-SNR (Signal Quality) + ArcFace (Identity Verification).
- **Optimization:** JIT-compiled Selective Scan for efficient training on consumer GPUs.

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
Run the full evaluation pipeline (multi-seed, cross-subject validation):
```bash
# Train Mamba-augmented model (v2)
python experiments/v2_mamba_denoiser/main.py
```

### 3. Backup Weights
Automatically zip and upload checkpoints to Google Drive:
```bash
python backup_full.py
```

## 📈 Results (Preview)

| Model | Denoising (SI-SNR) | Verification (EER) | Identification (P@1) |
|-------|-------------------|--------------------|----------------------|
| Baseline (CNN) | 12.5 dB | 4.2% | 92.1% |
| **NeuroMamba** | **14.8 dB** | **2.1%** | **96.5%** |

*(Results based on Subject-Disjoint protocol).*

## 📜 Citation

If you use this code, please cite our paper:

```bibtex
@article{zhugez2026neurobiometrics,
  title={Neuro-Biometrics: Efficient EEG Denoising via State Space Models},
  author={Ly Ngoc Vu},
  year={2026}
}
```

## 🛡️ License
MIT License. For research purposes only.
