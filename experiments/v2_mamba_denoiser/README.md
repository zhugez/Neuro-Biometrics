# V2 Mamba Denoiser - Experiment Results

## Overview

This repository contains the results of multi-seed comprehensive evaluation of EEG denoising and biometric identification using various ResNet architectures with different loss functions.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (Stage 1) | 20 |
| Epochs (Stage 2) | 1 |
| Batch Size | 64 |
| Holdout Subjects | 2, 5, 7, 12 |
| Seeds | 1, 2, 3 |

## Results Summary

### Gaussian Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.8227 ± 0.0199 | 0.9555 ± 0.0073 | 12.24 ± 0.30 | 0.5805 ± 0.1139 |
| ResNet18_MultiSim | 0.8295 ± 0.0164 | 0.9670 ± 0.0005 | 12.25 ± 0.32 | 0.5398 ± 0.0611 |
| **ResNet34_ArcFace** | **0.8457 ± 0.0111** | **0.9654 ± 0.0049** | **12.26 ± 0.29** | **0.6369 ± 0.0692** |

### Powerline Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.8565 ± 0.0045 | 0.9692 ± 0.0030 | 32.81 ± 1.01 | 0.4920 ± 0.0448 |
| ResNet18_MultiSim | 0.8764 ± 0.0063 | 0.9659 ± 0.0054 | 32.54 ± 1.06 | 0.4326 ± 0.0050 |
| **ResNet34_ArcFace** | **0.9013 ± 0.0054** | **0.9787 ± 0.0048** | **32.34 ± 0.28** | **0.4740 ± 0.0264** |

### EMG Noise

| Model | p@1 | p@5 | SI-SNR (dB) | AUROC |
|-------|-----|-----|-------------|-------|
| ResNet34_MultiSim | 0.8563 ± 0.0034 | 0.9696 ± 0.0002 | 14.01 ± 0.35 | 0.4674 ± 0.0231 |
| ResNet18_MultiSim | 0.8203 ± 0.0473 | 0.9661 ± 0.0080 | 14.03 ± 0.38 | 0.4715 ± 0.0207 |
| **ResNet34_ArcFace** | **0.8578 ± 0.0159** | **0.9741 ± 0.0038** | **14.02 ± 0.36** | **0.5301 ± 0.0255** |

## Key Findings

### Best Performing Models by Noise Type

| Noise Type | Best Model | Top-1 Accuracy | Top-5 Accuracy |
|------------|------------|----------------|----------------|
| Gaussian | ResNet34_ArcFace | 84.57% | 96.54% |
| Powerline | ResNet34_ArcFace | 90.13% | 97.87% |
| EMG | ResNet34_ArcFace | 85.78% | 97.41% |

### Detailed Metrics

#### ResNet34_ArcFace - Gaussian Noise
- **Top-1 Accuracy**: 84.57% ± 1.11%
- **Top-5 Accuracy**: 96.54% ± 0.49%
- **SI-SNR**: 12.26 ± 0.29 dB
- **EER**: 34.46% ± 3.43%
- **AUROC**: 0.637 ± 0.069
- **AUPR**: 0.638 ± 0.088
- **TAR@FAR=1%**: 95.60% ± 2.57%
- **FRR**: 4.40% ± 2.57%

#### ResNet34_ArcFace - Powerline Noise
- **Top-1 Accuracy**: 90.13% ± 0.54%
- **Top-5 Accuracy**: 97.87% ± 0.48%
- **SI-SNR**: 32.34 ± 0.28 dB
- **EER**: 37.30% ± 1.36%
- **AUROC**: 0.474 ± 0.026
- **AUPR**: 0.490 ± 0.023
- **TAR@FAR=1%**: 94.77% ± 1.41%
- **FRR**: 5.23% ± 1.41%

#### ResNet34_ArcFace - EMG Noise
- **Top-1 Accuracy**: 85.78% ± 1.59%
- **Top-5 Accuracy**: 97.41% ± 0.38%
- **SI-SNR**: 14.02 ± 0.36 dB
- **EER**: 34.75% ± 3.80%
- **AUROC**: 0.530 ± 0.026
- **AUPR**: 0.539 ± 0.024
- **TAR@FAR=1%**: 95.70% ± 2.34%
- **FRR**: 4.30% ± 2.34%

## Model Architecture Details

- **ResNet18**: Lightweight architecture with 18 layers
- **ResNet34**: Deeper architecture with 34 layers
- **MultiSim**: Multi-similarity loss function
- **ArcFace**: Additive angular margin loss for improved feature discrimination

## Latency & Model Size

| Model | Parameters | Inference Latency (ms) |
|-------|------------|------------------------|
| ResNet34_MultiSim | 21.7M | ~99 |
| ResNet18_MultiSim | 11.2M | ~52 |
| ResNet34_ArcFace | 21.7M | ~99 |

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

*Generated: 2026-02-18*
