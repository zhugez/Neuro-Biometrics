# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Latest:**
> - [2026-02-19] Fixed critical V2 embedder bugs (conv1, reshape, projection head)
> - [2026-02-19] Google Drive backup via [gogcli](https://github.com/steipete/gogcli)
> - [2026-02-11] Integrated **Mamba SSM** into WaveNet denoiser (V2)

---

## 📖 Introduction

This repository implements the paper **"Enhancing EEG-based Biometrics with Mamba-augmented Denoising Autoencoders"**.

We propose a **two-stage architecture**:

| Stage | Component | Objective |
|---|---|---|
| **Stage 1** — Denoising | WaveNet (Dilated Conv1D) + optional **Mamba Block** (SSM) | Reconstruct clean EEG signals from noisy input (SI-SNR loss) |
| **Stage 2** — Embedding | ResNet-18/34 with metric learning head | Extract identity-robust 128-d embeddings (ArcFace / MultiSimilarity loss) |

### Why Mamba?

Standard convolutional denoisers have a fixed receptive field. **Mamba** (Selective State Space Model) provides:
- **Linear-time** sequence modeling (vs quadratic for Transformers)
- **Content-aware** gating — selectively remembers/forgets temporal context
- **Drop-in integration** — placed at the midpoint of the WaveNet block stack as a residual module

---

## 🏗️ Architecture

```
Input EEG (B, 4, 800)                 4 EEG channels, 800 time samples
        │
        ▼
┌────────────────────────┐
│  WaveNet Denoiser      │  3 blocks × 4 layers, dilated Conv1D
│  ├─ WaveNetBlock ×6    │  dilation = 1,2,4,8 per block
│  ├─ [MambaBlock] ×1    │  inserted at layer 6 (midpoint)
│  └─ WaveNetBlock ×6    │
│  Output Conv           │  SI-SNR loss, 30 epochs
└────────┬───────────────┘
         │ denoised (B, 4, 800)
         ▼
┌────────────────────────┐
│  ResNet Embedder       │  Reshape 800 → (25, 32)
│  ├─ Conv2d 3×3 s=1     │  no maxpool (preserve spatial info)
│  ├─ ResNet backbone    │  pretrained ImageNet features
│  └─ FC → ReLU → Drop   │
│       → FC → BN → L2   │  128-d normalized embedding
└────────┬───────────────┘
         │ embedding (B, 128)
         ▼
  ArcFace / MultiSimilarity          30 epochs, metric learning
```

### Key Design Choices

| Decision | Rationale |
|---|---|
| **Conv1 3×3 stride=1** (not 7×7 stride=2) | EEG input is small (25×32) — large kernels destroy spatial info |
| **No maxpool** | Same reason — avoid downsampling too aggressively |
| **Deeper projection head** (FC→ReLU→Dropout→FC→BN) | More capacity for learning discriminative embeddings |
| **2D reshape** via `_find_2d_shape(800)` → (25, 32) | Gives proper spatial structure for 2D convolutions |
| **Data augmentation** in Stage 2 | Gaussian noise jitter + random amplitude scaling for robustness |

---

## 🛠️ Installation

```bash
git clone https://github.com/zhugez/Neuro-Biometrics.git
cd Neuro-Biometrics
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch ≥ 2.0` | Core deep learning framework |
| `mamba-ssm` + `causal-conv1d` | Mamba SSM with CUDA kernels |
| `pytorch-metric-learning` | ArcFace, MultiSimilarity losses |
| `torchvision` | ResNet backbones |
| `mne` | EEG signal processing |

---

## 📊 Usage

### 1. Download Dataset
```bash
python download_dataset.py
```

### 2. Training

```bash
# V1 Baseline: WaveNet + ResNet (no Mamba)
python experiments/v1_baseline/main.py --epochs 30 --seeds 3

# V2 Mamba: WaveNet + Mamba + ResNet
python experiments/v2_mamba/main.py

# Quick smoke test
python experiments/v1_baseline/main.py --one-sample
```

### 3. Backup Weights

```bash
# Zip only (auto-saves to /kaggle/working/ on Kaggle)
python backup_full.py

# Zip + upload to Google Drive
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

---

## 📁 Project Structure

```
Neuro-Biometrics/
├── experiments/
│   ├── v1_baseline/              # V1: WaveNet denoiser + ResNet embedder
│   │   ├── main.py               # Training entry point
│   │   ├── model.py              # WaveNetDenoiser, ResNetMetricEmbedder
│   │   ├── trainer.py            # TwoStageTrainer (SI-SNR → metric learning)
│   │   └── datapreprocessor.py   # EEG loading, noise generation
│   └── v2_mamba/                 # V2: + Mamba block in denoiser
│       ├── main.py               # Training entry point
│       ├── model.py              # WaveNetDenoiser + MambaBlock, ResNetMetricEmbedder
│       ├── trainer.py            # TwoStageTrainer + augmentation
│       ├── datapreprocessor.py   # EEG loading, noise generation
│       └── README.md             # V2 detailed results
├── dataset/                      # EEG data (gitignored)
├── backup_full.py                # Zip & upload weights to Google Drive
├── download_dataset.py           # Download dataset from Google Drive
├── requirements.txt
└── README.md
```

---

## 📈 Results

> Multi-seed evaluation (3 seeds). Config: 30/30 epochs (Stage 1/2), batch 64, holdout subjects {2, 5, 7, 12}.
> Subject-disjoint protocol — holdout subjects are never seen during training.

### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.814 ± 0.044 | 0.959 ± 0.010 | 12.34 ± 0.31 | 0.461 ± 0.017 |
| ResNet18 + MultiSim | 0.793 ± 0.064 | 0.959 ± 0.005 | 12.34 ± 0.31 | 0.451 ± 0.009 |
| **ResNet34 + ArcFace** | **0.865 ± 0.041** | **0.973 ± 0.008** | **12.34 ± 0.31** | 0.419 ± 0.013 |

### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.868 ± 0.028 | 0.967 ± 0.013 | 36.73 ± 1.62 | 0.464 ± 0.018 |
| ResNet18 + MultiSim | 0.857 ± 0.004 | 0.969 ± 0.002 | 36.78 ± 1.85 | 0.452 ± 0.010 |
| **ResNet34 + ArcFace** | **0.896 ± 0.013** | **0.977 ± 0.003** | 36.67 ± 1.44 | **0.564 ± 0.097** |

### EMG Noise (20–80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ |
|---|---|---|---|---|
| ResNet34 + MultiSim | 0.813 ± 0.003 | 0.953 ± 0.008 | 14.11 ± 0.36 | 0.454 ± 0.004 |
| ResNet18 + MultiSim | 0.820 ± 0.053 | 0.962 ± 0.007 | 14.11 ± 0.37 | 0.510 ± 0.029 |
| **ResNet34 + ArcFace** | **0.893 ± 0.014** | **0.976 ± 0.005** | **14.11 ± 0.37** | **0.535 ± 0.077** |

### Metric Definitions

| Metric | Description |
|---|---|
| **P@1** | Precision@1 — fraction of queries whose nearest neighbor shares the same identity |
| **P@5** | Precision@5 — fraction of 5 nearest neighbors that share the same identity |
| **SI-SNR** | Scale-Invariant Signal-to-Noise Ratio — denoising quality (higher = cleaner signal) |
| **AUROC** | Area Under ROC — binary verification performance (same vs different identity) |
| **EER** | Equal Error Rate — threshold where FAR = FRR (lower = better, shown in detailed results) |

### Key Findings

- **ResNet34 + ArcFace** achieves best P@1 across all noise types (**86.5% / 89.6% / 89.3%**)
- ArcFace outperforms MultiSimilarity for verification (higher AUROC on powerline + EMG)
- SI-SNR is similar across models (denoiser converges independently of embedder choice)
- Latency: ResNet34 ~100µs, ResNet18 ~85µs per inference
- ⚠️ These are **V2 pre-fix** results. Updated metrics pending retraining with corrected embedder.

---

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
