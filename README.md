# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Updates:**
> - [2026-02-19] **v2 Results:** Multi-seed evaluation of Mamba denoiser (30/30 epochs) — ResNet34+ArcFace achieves best P@1 across all noise types
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
│       ├── main.py                    # Entry point
│       ├── model.py                   # WaveNet + Mamba denoiser + ResNet embedder
│       ├── trainer.py                 # Training pipeline
│       ├── datapreprocessor.py        # Data preprocessing utilities
│       ├── visualize.py               # Visualization tools
│       └── README.md                  # v2 experiment results
├── dataset/                           # EEG data (Filtered_Data, Segmented_Data)
├── backup_full.py                     # Zip & save weights
├── requirements.txt
└── README.md
```

## �📈 Results

<!-- RESULTS_TABLE_START -->
| Model (Noise) | Params | SI-SNR | P@1 | P@5 | EER | AUROC | AUPR | Latency |
|---|---|---|---|---|---|---|---|---|
| ResNet34_MultiSim (gaussian) | 21.74M | 12.34384248 dB | 0.81396315 | 0.95949614 | 0.38447895 | 0.46117393 | 0.55686906 | 0.0997 ms |
| ResNet18_MultiSim (gaussian) | 11.63M | 12.34099215 dB | 0.79324725 | 0.95855165 | 0.40152572 | 0.45103746 | 0.54357453 | 0.0869 ms |
| ResNet34_ArcFace (gaussian) | 21.74M | 12.34295043 dB | 0.86483344 | 0.97339037 | 0.34040222 | 0.41934955 | 0.52488582 | 0.0999 ms |
| ResNet34_MultiSim (powerline) | 21.74M | 36.72627652 dB | 0.86843580 | 0.96749538 | 0.35568511 | 0.46377299 | 0.56149379 | 0.1043 ms |
| ResNet18_MultiSim (powerline) | 11.63M | 36.78491616 dB | 0.85742965 | 0.96944165 | 0.35281986 | 0.45237626 | 0.55622528 | 0.0838 ms |
| ResNet34_ArcFace (powerline) | 21.74M | 36.66684530 dB | 0.89645645 | 0.97725263 | 0.37505239 | 0.56427098 | 0.61031896 | 0.0982 ms |
| ResNet34_MultiSim (emg) | 21.74M | 14.11007698 dB | 0.81323737 | 0.95290083 | 0.35979019 | 0.45378748 | 0.54683240 | 0.0989 ms |
| ResNet18_MultiSim (emg) | 11.63M | 14.11106035 dB | 0.81961975 | 0.96161413 | 0.37129743 | 0.50974680 | 0.58473090 | 0.0847 ms |
| ResNet34_ArcFace (emg) | 21.74M | 14.11258150 dB | 0.89284706 | 0.97645083 | 0.31109297 | 0.53505019 | 0.61671734 | 0.1009 ms |
<!-- RESULTS_TABLE_END -->

### v2: Mamba-Augmented WaveNet + ResNet (3-seed mean ± std)

Experiment config: 30 epochs (Stage 1), 30 epochs (Stage 2), batch 64, holdout subjects {2, 5, 7, 12}.

**Gaussian Noise**
| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.81396315 ± 0.04371831 | 0.95949614 ± 0.01034361 | 12.34384248 ± 0.30732190 | 0.46117393 ± 0.01670227 |
| ResNet18 + MultiSim | 0.79324725 ± 0.06396291 | 0.95855165 ± 0.00516182 | 12.34099215 ± 0.30929428 | 0.45103746 ± 0.00918813 |
| **ResNet34 + ArcFace** | **0.86483344 ± 0.04138711** | **0.97339037 ± 0.00775948** | **12.34295043 ± 0.30899014** | **0.41934955 ± 0.01273907** |

**Powerline Noise (50 Hz)**
| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.86843580 ± 0.02835965 | 0.96749538 ± 0.01271194 | 36.72627652 ± 1.62453497 | 0.46377299 ± 0.01849189 |
| ResNet18 + MultiSim | 0.85742965 ± 0.00355056 | 0.96944165 ± 0.00228316 | 36.78491616 ± 1.84745944 | 0.45237626 ± 0.00970946 |
| **ResNet34 + ArcFace** | **0.89645645 ± 0.01259163** | **0.97725263 ± 0.00295469** | **36.66684530 ± 1.44028252** | **0.56427098 ± 0.09744667** |

**EMG Noise (20–80 Hz)**
| Model | P@1 | P@5 | SI-SNR (dB) | AUROC |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.81323737 ± 0.00297374 | 0.95290083 ± 0.00845641 | 14.11007698 ± 0.36224609 | 0.45378748 ± 0.00423437 |
| ResNet18 + MultiSim | 0.81961975 ± 0.05267057 | 0.96161413 ± 0.00681186 | 14.11106035 ± 0.37231974 | 0.50974680 ± 0.02916976 |
| **ResNet34 + ArcFace** | **0.89284706 ± 0.01431596** | **0.97645083 ± 0.00517026** | **14.11258150 ± 0.36649412** | **0.53505019 ± 0.07676603** |

> **Key findings (v2):**
> - **ResNet34 + ArcFace** achieves best P@1 on all noise types (86.5% / 89.6% / 89.3%)
> - **Latency:** ResNet34 ~100ms, ResNet18 ~85ms inference
> - AUROC varies by noise type (ArcFace best on powerline and EMG)
> - Latest metrics (2026-02-19) drawn from `artifacts/output_v2_mamba.json` multi-seed evaluation (30 epochs per stage, holdout {2,5,7,12}).

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
