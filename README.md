# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Latest:**
> - [2026-02-19] Refactored: extracted `experiments/shared/` module, V1/V2 are now thin wrappers (-2082 lines)
> - [2026-02-19] Fixed: deprecated AMP API, P@5 metric (CMC@5 → true Precision@5), dead code cleanup
> - [2026-02-19] Google Drive backup via [gogcli](https://github.com/steipete/gogcli)
> - [2026-02-19] Updated V1 + V2 training results in README (V2 rerun @ 17:16)
> - [2026-02-19] Added V1 baseline results (30/30 epochs)
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
python experiments/v2_mamba/main.py --epochs 30 --seeds 2
```

> ⚡ **H100 / High-End GPU Optimization:**
> For massive GPUs like NVIDIA H100 (80GB VRAM) paired with high-core CPUs, use the following configuration to fully saturate the hardware:
> ```bash
> # 1. Prevent overlapping CPU workers from fighting over cores:
> export OMP_NUM_THREADS=2
> export MKL_NUM_THREADS=2
> 
> # 2. Run with massive batch size, high workers, and H100 optimizations:
> #    --optimize-h100 enables torch.compile (Triton JIT) + bfloat16 mixed precision
> python experiments/v1_baseline/main.py --batch-size 4096 --num-workers 12 --optimize-h100
> python experiments/v2_mamba/main.py    --batch-size 4096 --num-workers 12 --optimize-h100
> ```

### 3. Quick Tests

```bash
# Ultra-fast forward pass (1 sample, no training)
python experiments/v2_mamba/main.py --one-sample

# Synthetic smoke test (forward + dataloader)
python experiments/v2_mamba/main.py --smoke

# Tiny 1-epoch training sanity check
python experiments/v2_mamba/main.py --mini-train
```

> 💡 All `--smoke`, `--one-sample`, `--mini-train` flags work for both V1 and V2.

### 4. Backup Weights

**Setup (one-time):** Create `.env` in project root:
```bash
echo 'GOG_KEYRING_PASSWORD=neuro2024' > .env
```

```bash
# Zip only (auto-saves to /kaggle/working/ on Kaggle)
python backup_full.py

# Zip + upload to Google Drive
python backup_full.py --gdrive --account you@gmail.com
```

> 💡 `backup_full.py` auto-loads `.env` — no manual `export` needed.

<details>
<summary>📋 One-time Google Drive setup</summary>

1. Install [gogcli](https://github.com/steipete/gogcli):
   ```bash
   curl -sL https://github.com/steipete/gogcli/releases/latest/download/gogcli_0.11.0_linux_amd64.tar.gz | tar xz -C /usr/local/bin gog
   ```

2. Create a **Desktop app** OAuth client at [Google Cloud Console](https://console.cloud.google.com/auth/clients) and download `client_secret.json`

3. Authenticate:
   ```bash
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
│   ├── shared/                   # Shared code (zero duplication)
│   │   ├── model.py              # WaveNetDenoiser, MambaBlock, ResNetMetricEmbedder
│   │   ├── trainer.py            # TwoStageTrainer, SISNRLoss, metrics
│   │   ├── datapreprocessor.py   # Config, EEG loading, noise generation
│   │   └── pipeline.py           # EEGPipeline, smoke/mini/one-sample, CLI
│   ├── v1_baseline/              # V1: WaveNet only (thin wrapper)
│   │   └── main.py               # run_cli(use_mamba=False)
│   └── v2_mamba/                 # V2: WaveNet + Mamba (thin wrapper)
│       ├── main.py               # run_cli(use_mamba=True)
│       └── README.md             # V2 detailed results
├── dataset/                      # EEG data (gitignored)
├── .env                          # Secrets: GOG_KEYRING_PASSWORD (gitignored)
├── backup_full.py                # Zip & upload weights to Google Drive
├── download_dataset.py           # Download dataset from Google Drive
├── requirements.txt
└── README.md
```

> 💡 **V1 and V2 are identical** except the `use_mamba` flag. All model, trainer, data, and pipeline logic lives in `experiments/shared/`.

---

## 📈 Results

> **Protocol:** Subject-disjoint — holdout subjects {2, 5, 7, 12} never seen during training.
> Multi-seed evaluation (3 seeds), best model highlighted per noise type.

### V2: Mamba-Augmented Denoiser (30/30 epochs)

V2 adds a **MambaBlock** at the midpoint of the WaveNet denoiser + training augmentation (noise jitter, amplitude scaling).

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.800 ± 0.043 | 0.774 ± 0.049 | 12.40 ± 0.24 | 0.454 ± 0.058 | — |
| ResNet18 + MultiSim | 0.793 ± 0.073 | 0.764 ± 0.076 | 12.40 ± 0.23 | 0.481 ± 0.062 | — |
| **ResNet34 + ArcFace** | **0.822 ± 0.052** | **0.800 ± 0.058** | 12.40 ± 0.24 | **0.528 ± 0.062** | **39.1%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.871 ± 0.041 | 0.841 ± 0.043 | 37.32 ± 1.73 | 0.481 ± 0.054 | — |
| ResNet18 + MultiSim | 0.837 ± 0.052 | 0.809 ± 0.059 | 37.32 ± 1.85 | 0.531 ± 0.044 | — |
| **ResNet34 + ArcFace** | **0.873 ± 0.042** | **0.854 ± 0.049** | 37.22 ± 1.77 | **0.580 ± 0.063** | **39.6%** |

#### EMG Noise (20–80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.822 ± 0.044 | 0.792 ± 0.037 | 14.16 ± 0.32 | 0.553 ± 0.050 | — |
| ResNet18 + MultiSim | 0.806 ± 0.056 | 0.781 ± 0.063 | 14.16 ± 0.31 | 0.501 ± 0.078 | — |
| **ResNet34 + ArcFace** | **0.849 ± 0.055** | **0.826 ± 0.060** | 14.16 ± 0.32 | **0.587 ± 0.078** | **38.0%** |

### V1: Baseline — WaveNet Only (30/30 epochs)

V1 uses the same WaveNet denoiser and ResNet embedder, but **without Mamba** and without training augmentation.

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.739 ± 0.056 | 0.704 ± 0.056 | 12.28 ± 0.27 | 0.446 ± 0.048 | — |
| ResNet18 + MultiSim | 0.743 ± 0.061 | 0.724 ± 0.062 | 12.27 ± 0.27 | 0.505 ± 0.072 | — |
| **ResNet34 + ArcFace** | **0.785 ± 0.019** | **0.763 ± 0.009** | 12.28 ± 0.27 | 0.462 ± 0.124 | **35.2%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.806 ± 0.032 | 0.767 ± 0.030 | 37.09 ± 1.47 | 0.504 ± 0.044 | — |
| ResNet18 + MultiSim | 0.837 ± 0.044 | 0.807 ± 0.048 | 37.08 ± 1.76 | 0.475 ± 0.024 | — |
| **ResNet34 + ArcFace** | **0.874 ± 0.041** | **0.855 ± 0.043** | 37.09 ± 1.47 | 0.420 ± 0.060 | **37.7%** |

#### EMG Noise (20–80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.786 ± 0.010 | 0.746 ± 0.012 | 13.95 ± 0.35 | 0.481 ± 0.040 | — |
| ResNet18 + MultiSim | 0.752 ± 0.052 | 0.733 ± 0.060 | 13.96 ± 0.34 | 0.490 ± 0.024 | — |
| **ResNet34 + ArcFace** | **0.801 ± 0.044** | **0.786 ± 0.046** | 13.95 ± 0.35 | 0.467 ± 0.106 | **35.4%** |

### V1 vs V2 Comparison

| Feature | V1 Baseline | V2 Mamba |
|---|---|---|
| **Denoiser** | WaveNet only | WaveNet + MambaBlock |
| **Stage 1 epochs** | 30 | 30 |
| **Stage 2 epochs** | 30 | 30 |
| **Training augmentation** | ❌ None | ✅ Noise jitter + amplitude scaling |
| **Best P@1 (Gaussian)** | **78.5%** | **82.2%** |
| **Best P@1 (Powerline)** | **87.4%** | **87.3%** |
| **Best P@1 (EMG)** | **80.1%** | **84.9%** |

### Metric Definitions

| Metric | Description |
|---|---|
| **P@1** | Precision@1 — fraction of queries whose nearest neighbor shares the same identity |
| **P@5** | Precision@5 — fraction of 5 nearest neighbors that share the same identity |
| **SI-SNR** | Scale-Invariant Signal-to-Noise Ratio — denoising quality (higher = cleaner signal) |
| **AUROC** | Area Under ROC — binary verification performance (same vs different identity) |
| **EER** | Equal Error Rate — threshold where FAR = FRR (lower = better, shown in detailed results) |

### Key Findings

- **ResNet34 + ArcFace** remains strongest for V2 on P@1 and AUROC across all noise types
- **Mamba improves identification** on Gaussian/EMG (+3.7pp / +4.8pp P@1), while Powerline is roughly on par (-0.1pp)
- **SI-SNR nearly identical** between V1 and V2 (~12.3 / 37.1 / 14.0 dB) — Mamba doesn't improve raw denoising, but helps downstream embeddings
- **AUROC still moderate** (best ~0.58) — verification can improve with harder negatives and score calibration
- Latency: ResNet34 ~0.21–0.23 ms, ResNet18 ~0.19–0.21 ms per inference

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
