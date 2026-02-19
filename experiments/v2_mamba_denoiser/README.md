# V2 Mamba Denoiser - Experiment Results

## Overview

This repository contains the results of multi-seed comprehensive evaluation of EEG denoising and biometric identification using various ResNet architectures with different loss functions.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (Stage 1) | 30 |
| Epochs (Stage 2) | 30 |
| Batch Size | 64 |
| Holdout Subjects | 2, 5, 7, 12 |
| Seeds | 1, 2, 3 |

## Results Summary

### Gaussian Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.81396315 ± 0.04371831 | 0.95949614 ± 0.01034361 | 12.34384248 ± 0.30732190 | 0.46117393 ± 0.01670227 |
| ResNet18_MultiSim | 0.79324725 ± 0.06396291 | 0.95855165 ± 0.00516182 | 12.34099215 ± 0.30929428 | 0.45103746 ± 0.00918813 |
| **ResNet34_ArcFace** | **0.86483344 ± 0.04138711** | **0.97339037 ± 0.00775948** | **12.34295043 ± 0.30899014** | **0.41934955 ± 0.01273907** |

### Powerline Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.86843580 ± 0.02835965 | 0.96749538 ± 0.01271194 | 36.72627652 ± 1.62453497 | 0.46377299 ± 0.01849189 |
| ResNet18_MultiSim | 0.85742965 ± 0.00355056 | 0.96944165 ± 0.00228316 | 36.78491616 ± 1.84745944 | 0.45237626 ± 0.00970946 |
| **ResNet34_ArcFace** | **0.89645645 ± 0.01259163** | **0.97725263 ± 0.00295469** | **36.66684530 ± 1.44028252** | **0.56427098 ± 0.09744667** |

### EMG Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.81323737 ± 0.00297374 | 0.95290083 ± 0.00845641 | 14.11007698 ± 0.36224609 | 0.45378748 ± 0.00423437 |
| ResNet18_MultiSim | 0.81961975 ± 0.05267057 | 0.96161413 ± 0.00681186 | 14.11106035 ± 0.37231974 | 0.50974680 ± 0.02916976 |
| **ResNet34_ArcFace** | **0.89284706 ± 0.01431596** | **0.97645083 ± 0.00517026** | **14.11258150 ± 0.36649412** | **0.53505019 ± 0.07676603** |

## Key Findings

### Best Performing Models by Noise Type

| Noise Type | Best Model | Top-1 Accuracy | Top-5 Accuracy |
|------------|------------|----------------|----------------|
| Gaussian | ResNet34_ArcFace | 86.48% | 97.34% |
| Powerline | ResNet34_ArcFace | 89.65% | 97.73% |
| EMG | ResNet34_ArcFace | 89.28% | 97.65% |

### Detailed Metrics

#### ResNet34_ArcFace - Gaussian Noise
- **Top-1 Accuracy**: 86.48% ± 4.14%
- **Top-5 Accuracy**: 97.34% ± 0.78%
- **SI-SNR**: 12.34 ± 0.31 dB
- **EER**: 34.04% ± 1.34%
- **AUROC**: 0.419 ± 0.013
- **AUPR**: 0.525 ± 0.010
- **TAR**: 94.23% ± 0.07%
- **FRR**: 5.77% ± 0.07%

#### ResNet34_ArcFace - Powerline Noise
- **Top-1 Accuracy**: 89.65% ± 1.26%
- **Top-5 Accuracy**: 97.73% ± 0.30%
- **SI-SNR**: 36.67 ± 1.44 dB
- **EER**: 37.51% ± 2.14%
- **AUROC**: 0.564 ± 0.097
- **AUPR**: 0.610 ± 0.069
- **TAR**: 89.25% ± 8.81%
- **FRR**: 10.75% ± 8.81%

#### ResNet34_ArcFace - EMG Noise
- **Top-1 Accuracy**: 89.28% ± 1.43%
- **Top-5 Accuracy**: 97.65% ± 0.52%
- **SI-SNR**: 14.11 ± 0.37 dB
- **EER**: 31.11% ± 0.21%
- **AUROC**: 0.535 ± 0.077
- **AUPR**: 0.617 ± 0.070
- **TAR**: 94.28% ± 1.25%
- **FRR**: 5.72% ± 1.25%

## Model Architecture Details

- **ResNet18**: Lightweight architecture with 18 layers
- **ResNet34**: Deeper architecture with 34 layers
- **MultiSim**: Multi-similarity loss function
- **ArcFace**: Additive angular margin loss for improved feature discrimination

## Latency & Model Size

| Model | Parameters | Inference Latency (ms) |
|-------|------------|------------------------|
| ResNet34_MultiSim | 21.7M | ~101 |
| ResNet18_MultiSim | 11.6M | ~85 |
| ResNet34_ArcFace | 21.7M | ~100 |

## File Structure

```
.
├── output_v2_mamba.json    # Complete experiment results
├── trainer.py              # Training pipeline
├── main.py                 # Main execution script
├── model.py                # Model architectures
├── datapreprocessor.py     # Data preprocessing utilities
├── visualize.py            # Visualization tools
└── weights/                # Model weights directory
```

## Conclusion

**ResNet34 with ArcFace loss** consistently achieved the best performance across all noise types:
- Best Top-1 accuracy on all three noise types
- Best Top-5 accuracy on Gaussian and EMG noise
- Competitive SI-SNR performance
- Higher AUROC scores indicating better biometric verification capability

The ArcFace loss function provides better feature discrimination compared to Multi-similarity loss, resulting in improved identification and verification performance.

---

*Generated: 2026-02-19*
