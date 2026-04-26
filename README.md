# Neuro-Biometrics 🧠⚡️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-blue)](https://github.com/zhugez/Neuro-Biometrics)

**Robust EEG Denoising and Biometric Verification using State Space Models (Mamba) and Metric Learning.**

> 🚀 **Latest:**
> - [2026-04-25] Updated V4 multimodal RTX 5090 3-seed results and cross-version comparison
> - [2026-02-20] Added V3 tuned quick-run results (`v3_mamba_tuned`, 1 seed, H100-optimized)
> - [2026-02-19] Refactored: extracted `experiments/shared/` module, V1/V2 are now thin wrappers (-2082 lines)
> - [2026-02-19] Fixed: deprecated AMP API, P@5 metric (CMC@5 → true Precision@5), dead code cleanup
> - [2026-02-19] Google Drive backup via [gogcli](https://github.com/steipete/gogcli)
> - [2026-02-20] Updated V1/V2 result tables in README and aligned notes with shared Stage-2 augmentation in code
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
python experiments/v2_mamba/main.py --epochs 30 --seeds 3

# V4 Multimodal: WaveNet + Mamba + EEG/spectrogram fusion
python experiments/v4_multimodal/main.py --epochs 30 --seeds 3 --batch-size 256 --num-workers 8 --spectrogram-source denoised
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
│   ├── v2_mamba/                 # V2: WaveNet + Mamba (thin wrapper)
│   │   ├── main.py               # run_cli(use_mamba=True)
│   │   └── README.md             # V2 detailed results
│   ├── v3_mamba_tuned/           # V3: tuned Mamba preset
│   └── v4_multimodal/            # V4: EEG + spectrogram multimodal fusion
│       ├── main.py
│       └── output_v4_multimodal.json
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
> Multi-seed evaluation uses 3 seeds unless noted; V3 is a single-seed quick run. Best P@1 model is highlighted per noise type.

### V2: Mamba-Augmented Denoiser (30/30 epochs)

V2 inserts a **MambaBlock** at the midpoint of the WaveNet denoiser.

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.754 ± 0.046 | 0.732 ± 0.047 | 10.64 ± 0.15 | 0.476 ± 0.031 | — |
| ResNet18 + MultiSim | 0.742 ± 0.049 | 0.722 ± 0.058 | 10.67 ± 0.15 | 0.533 ± 0.097 | — |
| **ResNet34 + ArcFace** | **0.798 ± 0.045** | **0.779 ± 0.044** | 10.65 ± 0.13 | **0.535 ± 0.032** | **36.0%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.805 ± 0.025 | 0.773 ± 0.033 | 19.56 ± 0.70 | 0.545 ± 0.065 | — |
| ResNet18 + MultiSim | 0.812 ± 0.067 | 0.790 ± 0.074 | 19.71 ± 0.38 | 0.587 ± 0.088 | — |
| **ResNet34 + ArcFace** | **0.858 ± 0.025** | **0.836 ± 0.029** | 19.61 ± 0.63 | 0.532 ± 0.062 | **37.9%** |

#### EMG Noise (20–80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.761 ± 0.110 | 0.739 ± 0.109 | 11.68 ± 0.19 | 0.547 ± 0.062 | — |
| ResNet18 + MultiSim | 0.753 ± 0.047 | 0.737 ± 0.052 | 11.66 ± 0.24 | 0.515 ± 0.067 | — |
| **ResNet34 + ArcFace** | **0.811 ± 0.050** | **0.796 ± 0.054** | 11.67 ± 0.18 | **0.553 ± 0.098** | **35.7%** |

### V1: Baseline — WaveNet Only (30/30 epochs)

V1 uses the same WaveNet denoiser and ResNet embedder, but **without Mamba**.

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.747 ± 0.035 | 0.729 ± 0.031 | 10.66 ± 0.21 | 0.451 ± 0.060 | — |
| ResNet18 + MultiSim | 0.737 ± 0.073 | 0.720 ± 0.073 | 10.70 ± 0.19 | 0.465 ± 0.073 | — |
| **ResNet34 + ArcFace** | **0.822 ± 0.048** | **0.802 ± 0.059** | 10.64 ± 0.24 | 0.570 ± 0.139 | **35.9%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.804 ± 0.051 | 0.776 ± 0.047 | 19.98 ± 0.44 | 0.569 ± 0.042 | — |
| ResNet18 + MultiSim | 0.805 ± 0.052 | 0.781 ± 0.054 | 19.85 ± 0.51 | 0.525 ± 0.096 | — |
| **ResNet34 + ArcFace** | **0.860 ± 0.042** | **0.836 ± 0.052** | 19.98 ± 0.44 | 0.578 ± 0.054 | **36.9%** |

#### EMG Noise (20–80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.772 ± 0.037 | 0.746 ± 0.033 | 11.65 ± 0.24 | 0.467 ± 0.056 | — |
| ResNet18 + MultiSim | 0.756 ± 0.068 | 0.735 ± 0.071 | 11.65 ± 0.31 | 0.507 ± 0.133 | — |
| **ResNet34 + ArcFace** | **0.824 ± 0.052** | **0.810 ± 0.058** | 11.65 ± 0.24 | 0.606 ± 0.108 | **35.3%** |

### V1 vs V2 Comparison

| Feature | V1 Baseline | V2 Mamba |
|---|---|---|
| **Denoiser** | WaveNet only | WaveNet + MambaBlock |
| **Stage 1 epochs** | 30 | 30 |
| **Stage 2 epochs** | 30 | 30 |
| **Training augmentation (Stage 2)** | ✅ Noise jitter + amplitude scaling | ✅ Noise jitter + amplitude scaling |
| **Best P@1 (Gaussian)** | **82.2%** | **79.8%** |
| **Best P@1 (Powerline)** | **86.0%** | **85.8%** |
| **Best P@1 (EMG)** | **82.4%** | **81.1%** |

### V3: Mamba Tuned (Quick Run, 1 seed)

> ⚠️ This is a **single-seed quick run** with `--optimize-h100`, so treat as directional only (not directly comparable to multi-seed V1/V2 stats).

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.717 | 0.684 | 11.19 | 0.397 | — |
| ResNet18 + MultiSim | 0.731 | 0.704 | 11.13 | 0.439 | — |
| **ResNet34 + ArcFace** | **0.749** | **0.734** | 11.18 | 0.421 | **39.3%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.818 | 0.792 | 23.80 | 0.488 | — |
| ResNet18 + MultiSim | **0.869** | **0.849** | **23.93** | 0.483 | — |
| ResNet34 + ArcFace | 0.865 | 0.849 | 23.80 | **0.562** | **30.9%** |

#### EMG Noise (20-80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.718 | 0.680 | 12.45 | **0.465** | — |
| ResNet18 + MultiSim | 0.702 | 0.692 | 12.45 | 0.418 | — |
| **ResNet34 + ArcFace** | **0.758** | **0.738** | **12.48** | 0.418 | **40.0%** |

### V4: Multimodal EEG + Spectrogram Fusion (30/30 epochs)

V4 keeps the WaveNet+Mamba denoiser, adds a spectrogram Mamba branch, and fuses EEG/spectrogram embeddings with cross-attention. This RTX 5090 run used `--batch-size 256 --num-workers 8 --spectrogram-source denoised` with `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.

#### Gaussian Noise

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.764 ± 0.050 | 0.743 ± 0.053 | **12.15 ± 0.26** | 0.443 ± 0.021 | 37.6% ± 7.2% |
| ResNet18 + MultiSim | 0.782 ± 0.028 | 0.755 ± 0.035 | 12.15 ± 0.26 | 0.441 ± 0.036 | 36.9% ± 4.6% |
| **ResNet34 + ArcFace** | **0.824 ± 0.036** | **0.803 ± 0.040** | 12.15 ± 0.26 | **0.479 ± 0.095** | **35.6% ± 5.9%** |

#### Powerline Noise (50 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.819 ± 0.040 | 0.794 ± 0.049 | **32.38 ± 1.06** | **0.533 ± 0.051** | 40.8% ± 2.4% |
| ResNet18 + MultiSim | 0.831 ± 0.047 | 0.806 ± 0.052 | 32.38 ± 1.06 | 0.450 ± 0.015 | 37.1% ± 2.1% |
| **ResNet34 + ArcFace** | **0.877 ± 0.006** | **0.855 ± 0.005** | 32.38 ± 1.06 | 0.501 ± 0.044 | **36.0% ± 3.9%** |

#### EMG Noise (20-80 Hz)

| Model | P@1 ↑ | P@5 ↑ | SI-SNR (dB) ↑ | AUROC ↑ | EER ↓ |
|---|---|---|---|---|---|
| ResNet34 + MultiSim | 0.764 ± 0.080 | 0.747 ± 0.075 | **13.83 ± 0.35** | 0.439 ± 0.065 | 38.9% ± 3.9% |
| ResNet18 + MultiSim | 0.793 ± 0.067 | 0.775 ± 0.069 | 13.83 ± 0.35 | 0.422 ± 0.038 | 35.6% ± 6.6% |
| **ResNet34 + ArcFace** | **0.838 ± 0.057** | **0.827 ± 0.065** | 13.83 ± 0.35 | **0.514 ± 0.082** | **35.0% ± 5.8%** |

### Cross-Version Best P@1 Comparison

| Noise | V1 Baseline | V2 Mamba | V3 Tuned (1 seed) | V4 Multimodal |
|---|---:|---:|---:|---:|
| Gaussian | 0.822 | 0.798 | 0.749 | **0.824** |
| Powerline | 0.860 | 0.858 | 0.869 | **0.877** |
| EMG | 0.824 | 0.811 | 0.758 | **0.838** |

| Version | Main change | Seeds | Best P@1 profile |
|---|---|---:|---|
| V1 | WaveNet denoiser only | 3 | Strong V1/V2 baseline before multimodal fusion |
| V2 | WaveNet + midpoint MambaBlock | 3 | Similar to V1, slightly lower P@1 in this run |
| V3 | Tuned Mamba preset, H100 optimized | 1 | Strong single-seed Powerline quick run, but not directly comparable to 3-seed results |
| V4 | Mamba denoiser + spectrogram Mamba + cross-attention fusion | 3 | Best observed multi-seed P@1 on Gaussian, Powerline, and EMG |

### Metric Definitions

| Metric | Description |
|---|---|
| **P@1** | Precision@1 — fraction of queries whose nearest neighbor shares the same identity |
| **P@5** | Precision@5 — fraction of 5 nearest neighbors that share the same identity |
| **SI-SNR** | Scale-Invariant Signal-to-Noise Ratio — denoising quality (higher = cleaner signal) |
| **AUROC** | Area Under ROC — binary verification performance (same vs different identity) |
| **EER** | Equal Error Rate — threshold where FAR = FRR (lower = better, shown in detailed results) |

### Key Findings

- **ResNet34 + ArcFace** remains strongest for V1/V2 by P@1, while V4 switches to **MultiSimilarity** heads for best P@1 across all noise types.
- In this run, **V1 is slightly higher than V2** on best P@1 for all three noise types (82.2 vs 79.8, 86.0 vs 85.8, 82.4 vs 81.1).
- **SI-SNR is nearly identical** between V1 and V2 (~10.6 / 19.8 / 11.7 dB across Gaussian/Powerline/EMG), indicating similar denoising quality.
- V4 reports higher SI-SNR than V1/V2 (12.3 / 37.3 / 14.0 dB), but lower biometric P@1, so the multimodal fusion path needs more tuning before it is competitive.
- **AUROC remains moderate** (~0.53 to ~0.61 depending on noise/model), so verification can still improve with calibration and harder negatives.
- V3 is single-seed only and should be treated as directional, not directly comparable to the 3-seed V1/V2/V4 summaries.
- Observed gaps are small in several settings and should be treated as **seed-sensitive** unless confirmed with larger repeated runs.

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
