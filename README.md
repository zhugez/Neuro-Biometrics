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

## 📈 Results

<!-- RESULTS_TABLE_START -->
| Model (Noise) | Params | SI-SNR | P@1 | P@5 | EER | AUROC | AUPR | Latency |
|---|---|---|---|---|---|---|---|---|
| *Baseline (Gaussian)* | *11.2M* | *12.50 dB* | *0.9210* | *0.9500* | *0.0420* | *0.9850* | *0.9700* | *15.20 ms* |
| **NeuroMamba (Gaussian)** | **11.5M** | **14.80 dB** | **0.9650** | **0.9850** | **0.0210** | **0.9950** | **0.9900** | **12.50 ms** |
| *NeuroMamba (Powerline)* | *...* | *Pending...* | *...* | *...* | *...* | *...* | *...* | *...* |
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
