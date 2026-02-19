# Neuro-Biometrics — Improvement Roadmap

> Tổng hợp từ phân tích kết quả V1 vs V2 (commit `34c94bd`)
> Baseline: ResNet34+ArcFace — V1: 78.5-87.4% P@1, V2: 80.4-89.4% P@1, AUROC 0.42-0.56

---

## Phase 1: Statistical Validity & Quick Wins (1-2 ngày)

### 1.1 Tăng seeds ≥ 5 🔴 CRITICAL
Nhiều improvement hiện tại **nằm trong khoảng std** → chưa đủ statistical significance.
```bash
python experiments/v2_mamba/main.py --epochs 30 --seeds 5
python experiments/v1_baseline/main.py --epochs 30 --seeds 5
```
- Thêm báo cáo **CI 95%** (confidence interval) bên cạnh mean ± std
- Kỳ vọng: Không tăng accuracy nhưng **tăng độ tin cậy** cho mọi kết luận

### 1.2 ArcFace Hyperparameter Grid
Margin và scale mặc định (m=0.3, s=30) chưa optimal. Grid search nhỏ:

| Param | Values | Notes |
|---|---|---|
| margin `m` | 0.2, **0.3**, 0.4, 0.5 | Cao hơn → harder, discriminative hơn |
| scale `s` | 20, **30**, 40, 64 | Cao hơn → sharper decision boundary |

- Chạy riêng cho mỗi noise type (Gaussian/Powerline/EMG)
- Kỳ vọng: **P@1 +2-3%**, nhanh nhất trong tất cả improvements

### 1.3 Unit Test cho P@K Metrics
Fix P@5 (CMC@5 → true Precision@5) đã làm nhưng **chưa có test**.
- Viết unit test với synthetic data: 10 embeddings, known labels, verify P@1/P@5/CMC
- Đảm bảo reproducibility giữa các lần chạy

---

## Phase 2: Loss & Training Strategy (3-5 ngày)

### 2.1 Hybrid Loss: ArcFace + SupContrastive
ArcFace mạnh cho identification (P@1) nhưng yếu cho verification (AUROC).
```python
loss = λ * arcface_loss(emb, y) + (1 - λ) * supcon_loss(emb, y)
# Try λ = {0.6, 0.7, 0.8}
```
- SupContrastive (pair-based) bổ trợ tốt hơn MultiSim (cũng proxy-based như ArcFace)
- Kỳ vọng: **AUROC +10-15%**, P@1 giữ nguyên hoặc tăng nhẹ

### 2.2 Joint Fine-tune End-to-End
Hiện tại 2-stage: denoiser denoise cho reconstruction, không cho identification.
```python
# After 2-stage training, unfreeze denoiser:
loss = α * si_snr_loss + β * arcface_loss
# LR 1e-5 to 3e-5, 5-10 epochs
```
- Denoiser sẽ học denoise **theo hướng tối ưu cho identification**
- Kỳ vọng: **P@1 +3-5%**, **AUROC +5-10%**

### 2.3 Hard-Negative Mining
Thay random batch → **subject-balanced sampler** + memory bank:
- Mỗi batch có K subjects × M samples/subject
- Memory bank lưu embeddings gần nhất → mine hardest negatives
- Kỳ vọng: **AUROC +15-20%**, EER giảm đáng kể

---

## Phase 3: Data & Augmentation (3-5 ngày)

### 3.1 Enhanced Augmentation
Hiện tại chỉ có noise jitter + amplitude scaling. Thêm:

| Augmentation | Mô tả | Impact |
|---|---|---|
| **Time shift** | Dịch ±50ms | Temporal invariance |
| **Channel dropout** | Zero random 1/4 channels | Robustness |
| **SpecAugment** | Mask frequency bands | Spectral robustness |
| **Mixup** | Blend 2 subjects (signal + label) | Regularization |
| **SNR curriculum** | Epoch 1-10: easy SNR → Epoch 20-30: hard SNR | Progressive difficulty |

### 3.2 Cross-dataset Pre-training
Dataset hiện tại nhỏ (~14 subjects). Pre-train denoiser trên:
- **PhysioNet EEG Motor Movement** (109 subjects)
- **BCI Competition IV** datasets
- Sau đó fine-tune embedder trên target dataset
- Kỳ vọng: **P@1 +5-10%** (more data = more generalizable features)

---

## Phase 4: Architecture (5-7 ngày)

### 4.1 1D Embedder (thay 2D reshape)
Reshape (B,4,800) → (B,4,25,32) là hacky. Dùng **1D ResNet** trực tiếp:
```python
# Current: EEG 1D → reshape 2D → ResNet2D
# Proposed: EEG 1D → ResNet1D (or ConvNeXt1D)
```
- Tự nhiên hơn cho time-series, không phụ thuộc factorization T=H×W

### 4.2 Multi-scale Mamba
Hiện tại: 1 MambaBlock ở midpoint. Thử:
- 2-3 MambaBlock phân bố đều (early/mid/late)
- Multi-resolution: Mamba trên raw + strided signal
- ⚠️ Cần benchmark: mỗi MambaBlock thêm ~0.04ms latency

### 4.3 Frequency-domain Branch
Thêm parallel branch: **STFT/Wavelet → CNN** → concat với time-domain embedding.
EEG biometrics literature cho thấy alpha/beta band power rất discriminative.

### 4.4 Modern Backbone
Thay ResNet18/34 (2015) → **ConvNeXt-Tiny** hoặc **EfficientNet-B0**:
- Ít params hơn, accuracy tương đương hoặc cao hơn
- Có pretrained weights tốt hơn

---

## Priority Matrix

| Phase | Idea | Effort | Impact | ROI |
|---|---|---|---|---|
| 1 | ≥5 seeds | ⬜ Thấp | 🔴 Critical | ⭐⭐⭐⭐⭐ |
| 1 | ArcFace grid (m, s) | ⬜ Thấp | P@1 +2-3% | ⭐⭐⭐⭐⭐ |
| 1 | Unit test P@K | ⬜ Thấp | Trust | ⭐⭐⭐⭐ |
| 2 | SupCon + ArcFace | 🟡 Trung bình | AUROC +10-15% | ⭐⭐⭐⭐ |
| 2 | Joint fine-tune | 🟡 Trung bình | P@1 +3-5% | ⭐⭐⭐⭐ |
| 2 | Hard-negative mining | 🟡 Trung bình | AUROC +15-20% | ⭐⭐⭐⭐ |
| 3 | Enhanced augmentation | 🟡 Trung bình | P@1 +2-3% | ⭐⭐⭐ |
| 3 | Cross-dataset pretrain | 🔴 Cao | P@1 +5-10% | ⭐⭐⭐ |
| 4 | 1D Embedder | 🔴 Cao | TBD | ⭐⭐ |
| 4 | Multi-scale Mamba | 🔴 Cao | P@1 +2-5% | ⭐⭐ |
| 4 | Frequency branch | 🔴 Cao | P@1 +3-5% | ⭐⭐ |

---

## Target Metrics (sau tất cả improvements)

| Metric | Hiện tại (V2) | Target | Notes |
|---|---|---|---|
| **P@1** | 80-89% | **92-95%** | Phase 1+2 improvements |
| **P@5** | 78-87% | **95-98%** | Follows P@1 |
| **AUROC** | 0.42-0.56 | **0.85-0.92** | Phase 2 (SupCon + hard mining) |
| **EER** | 34-38% | **8-12%** | Phase 2 |
| **SI-SNR** | 12-37 dB | **14-40 dB** | Phase 2.2 (joint fine-tune) |
