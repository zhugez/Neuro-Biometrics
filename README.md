# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Updates:**
> - [2026-02-19] Fixed critical V2 embedder bugs (conv1, maxpool, projection head, reshape)
> - [2026-02-19] Google Drive backup via [gogcli](https://github.com/steipete/gogcli)
> - [2026-02-11] Integrated **Mamba Selective Scan** into WaveNet denoiser (V2)
> - [2026-02-11] Added training augmentation (noise jitter + amplitude scaling)

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
│  ResNet18/34 Backbone│  Conv2d 3×3 stride=1, no maxpool
│  Projection Head     │  Linear→ReLU→Dropout→Linear→BN
│  L2 Normalize        │
└──────────┬───────────┘
           │ embedding (B, 128)
           ▼
   ArcFace / MultiSimilarity Loss   Stage 2: metric learning
```

## 🛠️ Installation

```bash
git clone https://github.com/zhugez/Neuro-Biometrics.git
cd Neuro-Biometrics
pip install -r requirements.txt
```

## 📊 Dataset & Usage

### 1. Download Data
```bash
python download_dataset.py
```

### 2. Training

```bash
# V1: Two-stage pipeline (WaveNet + ResNet)
python experiments/v1_two_stage_snr_0_5_10_20/main.py --epochs 30 --seeds 3

# V2: Mamba-augmented denoiser
python experiments/v2_mamba_denoiser/main.py

# Quick smoke test
python experiments/v1_two_stage_snr_0_5_10_20/main.py --one-sample
```

### 3. Backup Weights

```bash
# Zip only (auto-saves to /kaggle/working/ on Kaggle)
python backup_full.py

# Zip + upload to Google Drive
export GOG_KEYRING_PASSWORD='your_password'
python backup_full.py --gdrive --account you@gmail.com
```

<details>
<summary>📋 One-time Google Drive setup</summary>

1. Install [gogcli](https://github.com/steipete/gogcli):
   ```bash
   curl -sL https://github.com/steipete/gogcli/releases/latest/download/gogcli_0.11.0_linux_amd64.tar.gz | tar xz -C /usr/local/bin gog
   ```

2. Create a **Desktop app** OAuth client at [Google Cloud Console](https://console.cloud.google.com/auth/clients) and download `client_secret.json`

3. Authenticate:
   ```bash
   export GOG_KEYRING_PASSWORD='your_password'
   gog auth keyring file
   gog auth credentials client_secret.json
   gog auth add you@gmail.com --services drive --manual
   ```

</details>

## 📁 Project Structure

```
Neuro-Biometrics/
├── experiments/
│   ├── v1_two_stage_snr_0_5_10_20/   # Baseline: WaveNet + ResNet
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── datapreprocessor.py
│   └── v2_mamba_denoiser/             # V2: WaveNet + Mamba + ResNet
│       ├── main.py
│       ├── model.py
│       ├── trainer.py
│       ├── datapreprocessor.py
│       └── README.md                  # V2 experiment results
├── dataset/                           # EEG data (gitignored)
├── backup_full.py                     # Zip & upload weights to Google Drive
├── download_dataset.py                # Download dataset from Google Drive
├── requirements.txt
└── README.md
```

## 📈 Results

Multi-seed evaluation (3 seeds). Config: 30 epochs (Stage 1 + Stage 2), batch 64, holdout subjects {2, 5, 7, 12}.

### Gaussian Noise

| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.814 ± 0.044 | 0.959 ± 0.010 | 12.34 ± 0.31 | 0.461 ± 0.017 |
| ResNet18 + MultiSim | 0.793 ± 0.064 | 0.959 ± 0.005 | 12.34 ± 0.31 | 0.451 ± 0.009 |
| **ResNet34 + ArcFace** | **0.865 ± 0.041** | **0.973 ± 0.008** | **12.34 ± 0.31** | 0.419 ± 0.013 |

### Powerline Noise (50 Hz)

| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.868 ± 0.028 | 0.967 ± 0.013 | 36.73 ± 1.62 | 0.464 ± 0.018 |
| ResNet18 + MultiSim | 0.857 ± 0.004 | 0.969 ± 0.002 | 36.78 ± 1.85 | 0.452 ± 0.010 |
| **ResNet34 + ArcFace** | **0.896 ± 0.013** | **0.977 ± 0.003** | 36.67 ± 1.44 | **0.564 ± 0.097** |

### EMG Noise (20–80 Hz)

| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.813 ± 0.003 | 0.953 ± 0.008 | 14.11 ± 0.36 | 0.454 ± 0.004 |
| ResNet18 + MultiSim | 0.820 ± 0.053 | 0.962 ± 0.007 | 14.11 ± 0.37 | 0.510 ± 0.029 |
| **ResNet34 + ArcFace** | **0.893 ± 0.014** | **0.976 ± 0.005** | **14.11 ± 0.37** | **0.535 ± 0.077** |

> **Key findings:**
> - **ResNet34 + ArcFace** achieves best P@1 across all noise types (86.5% / 89.6% / 89.3%)
> - Latency: ResNet34 ~100µs, ResNet18 ~85µs per inference
> - These results are from **V2** (pre-embedder fix). New V2 results pending retraining.

## 📜 Citation

```bibtex
@article{zhugez2026neurobiometrics,
  title={Neuro-Biometrics: Efficient EEG Denoising via State Space Models},
  author={Ly Ngoc Vu and Huynh Cong Bang},
  year={2026}
}
```

## 🛡️ License
MIT License. For research purposes only.
