# Neuro-Biometrics — Comprehensive Optimization Roadmap

> Research-backed implementation plan to maximize P@1, AUROC, and EER.
> Generated: 2026-02-20 | Baseline: V2 ResNet34+ArcFace

---

## Đánh Giá IMPROVEMENTS.md Hiện Tại

### Tóm Tắt: Claim Nào Đúng / Sai

| Claim | Rating | Actual Expected Gain | Ghi Chú |
|---|---|---|---|
| ≥5 seeds | ✅ CRITICAL | Statistical validity | Không tăng accuracy, nhưng kết quả mới có ý nghĩa |
| ArcFace grid (m, s) | ✅ REALISTIC | P@1 +1–2% | Nhưng có bug: config không được đọc (xem A1) |
| Hybrid ArcFace + SupCon → AUROC +10–15% | ⚠️ OPTIMISTIC | AUROC +3–8% | Chỉ đúng nếu AUROC ~0.5 do loss, không phải session shift |
| Joint fine-tune → P@1 +3–5% | ✅ REALISTIC | P@1 +2–4% | AUROC gain bị thổi phồng |
| Hard negative mining → AUROC +15–20% | ❌ UNLIKELY | AUROC +2–5% | Với 14 subjects, memory bank không có tác dụng |
| Enhanced augmentation → P@1 +2–3% | ✅ REALISTIC | P@1 +1–2% | Skip mixup với N=14 |
| PhysioNet pretraining → P@1 +5–10% | ❌ UNLIKELY | P@1 +0–3% | Domain mismatch nghiêm trọng (64ch motor imagery vs 4ch resting) |
| 1D ResNet embedder | ✅ HIGH ROI | P@1 +3–8% | Improvement kiến trúc tốt nhất trong roadmap |
| Multi-scale Mamba 3 blocks | ✅ REALISTIC | P@1 +1–3% | MSGM paper xác nhận |
| Frequency branch (alpha/beta) | ⚠️ WRONG BANDS | P@1 +3–5% | **Delta (0.5–4 Hz) mới là band discriminative nhất**, không phải alpha/beta |

### Root Cause AUROC ~0.5

AUROC 0.42–0.56 không phải do loss function yếu. Nguyên nhân:
1. **Session variability** — model học session-specific artifacts thay vì identity features
2. **4 holdout subjects** quá ít (std ±0.097 overlap với random chance 0.5)
3. **ImageNet-pretrained ResNet** — domain mismatch cơ bản (ResNet biết về visual edges, không biết về neural oscillations)

---

## Kết Quả Kỳ Vọng Theo Từng Phase

| Phase | P@1 (best) | AUROC (best) | EER (best) |
|---|---|---|---|
| **Hiện tại (V2)** | 89.4% | 0.56 | 34% |
| **Sau Phase A** (1–3 ngày) | 93–95% | 0.66–0.72 | 26–28% |
| **Sau Phase B** (1–2 tuần) | 95–97% | 0.83–0.88 | 14–18% |
| **Sau Phase C** (2–4 tuần) | 97–99% | 0.88–0.93 | 8–12% |

---

## Phase A — Quick Wins (1–3 ngày, không thay đổi kiến trúc)

### A1. Fix Bug ArcFace Hardcode ⚡ — 2h | P@1 +2–4%, AUROC +4–7%

**Bug xác nhận**: `experiments/shared/trainer.py:205–208` khởi tạo ArcFace với `margin=0.3, scale=30` hardcode, bỏ qua hoàn toàn `self.config.arcface_margin/scale`. Các fields trong Config (`datapreprocessor.py:83–84`) chưa bao giờ được dùng.

```python
# experiments/shared/trainer.py:205–208 — CURRENT (broken):
metric_loss = ArcFaceLoss(num_classes, self.config.embed_dim,
                          margin=0.3, scale=30).to(self.device)

# FIX:
metric_loss = ArcFaceLoss(
    num_classes, self.config.embed_dim,
    margin=self.config.arcface_margin,   # reads from Config
    scale=self.config.arcface_scale,
).to(self.device)
```

```python
# experiments/shared/datapreprocessor.py:83–84 — update defaults:
arcface_margin: float = 0.5   # was 0.3 — ArcFace paper recommendation
arcface_scale: float = 64.0   # was 30 — s=30 giới hạn gradient signal
```

### A2. embed_dim 128 → 256 — 30m | P@1 +1–2%

```python
# experiments/shared/datapreprocessor.py:75:
embed_dim: int = 256   # was 128
```

### A3. Test-Time Adaptation (BN Statistics) ⚡ — 3h | AUROC +5–10%, EER -4–6%

Session variability làm lệch BN statistics từ training distribution. Update 2 batches trước inference:

```python
# experiments/shared/trainer.py — trong evaluate_comprehensive(), trước model.eval():
model.train()
with torch.no_grad():
    for i, (noisy, _, _) in enumerate(test_dl):
        if i >= 2: break   # 2 batches đủ để calibrate BN stats
        model(noisy.to(self.device))
model.eval()
```

### A4. Augmentation: Time Shift + Channel Dropout — 4h | P@1 +1–3%

```python
# experiments/shared/trainer.py — trong _train_stage2 loop, trước forward pass:
import random

def _augment(x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
    # Time shift ±50 samples (~250ms at 200Hz)
    shift = random.randint(-50, 50)
    x = torch.roll(x, shift, dims=2)
    # Channel dropout với 20% probability
    if random.random() < 0.2:
        ch = torch.randint(0, x.shape[1], (1,)).item()
        x[:, ch, :] = 0.0
    return x

# Trong vòng lặp training:
noisy = _augment(noisy)
```

### A5. Fix filter_low để Capture Delta Band ⚡ — 5m | AUROC +N/A (prerequisite)

**Quan trọng**: `filter_low=1.0` trong Config hiện tại đang cắt mất delta band (0.5–1 Hz), là band discriminative nhất.

```python
# experiments/shared/datapreprocessor.py:55:
filter_low: float = 0.5   # was 1.0 — delta band bắt đầu từ 0.5 Hz!
```

### A6. Tăng Seeds lên 5 + CI 95% — 1h | Statistical validity

```python
# experiments/v2_mamba/main.py:11:
run_cli(use_mamba=True, version="v2_mamba", default_seeds=5)

# experiments/v1_baseline/main.py:11:
run_cli(use_mamba=False, version="v1_baseline", default_seeds=5)

# experiments/shared/pipeline.py — trong _aggregate_results():
import scipy.stats as st
n = len(vals)
if n > 1:
    ci = st.t.interval(0.95, df=n-1, loc=np.mean(vals), scale=st.sem(vals))
    stats[f"{k}_ci95_low"] = float(ci[0])
    stats[f"{k}_ci95_high"] = float(ci[1])
```

---

## Phase B — Architecture Improvements (1–2 tuần)

### B1. Hybrid ArcFace + Supervised Contrastive Loss — 2 ngày | AUROC +12–18%, EER -8–12%

ArcFace là proxy loss (tốt cho P@1), SupCon là pairwise (tốt cho AUROC). Kết hợp cả hai:

```python
# experiments/shared/trainer.py:204 — thay ArcFace với hybrid:
from pytorch_metric_learning.losses import SupConLoss  # đã có trong pytorch-metric-learning

# Thêm vào Config (datapreprocessor.py sau line 84):
lambda_arc: float = 0.7   # 70% ArcFace + 30% SupCon
supcon_temperature: float = 0.07

# Trong _train_stage2:
arcface_fn = ArcFaceLoss(num_classes, self.config.embed_dim,
                          margin=self.config.arcface_margin,
                          scale=self.config.arcface_scale).to(self.device)
supcon_fn = SupConLoss(temperature=self.config.supcon_temperature).to(self.device)
lam = self.config.lambda_arc

def metric_loss(emb, y):
    return lam * arcface_fn(emb, y) + (1 - lam) * supcon_fn(emb, y)

params = list(model.embedder.parameters()) + list(arcface_fn.parameters())
```

### B2. Stage 3 Joint End-to-End Fine-tune — 2 ngày | P@1 +3–5%, AUROC +5–8%

Sau 2-stage training, unfreeze denoiser và train joint với dual objective:

```python
# experiments/shared/trainer.py — thêm method sau _train_stage2():

def _train_stage3_joint(self, model, train_dl, val_dl, num_classes,
                         metric_loss_fn, epochs: int = 10):
    """
    Stage 3: Joint end-to-end fine-tuning.
    Unfreeze denoiser, train với: loss = alpha*SI-SNR + beta*metric_loss
    LR thấp (1e-5) để tránh catastrophic forgetting của denoising ability.
    """
    model = model.to(self.device)
    for p in model.denoiser.parameters():
        p.requires_grad = True

    alpha, beta = 0.3, 0.7
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    use_amp = TRAINING_CONFIG["use_amp"] and self.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_p1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for noisy, clean, y in train_dl:
            noisy, clean, y = (noisy.to(self.device),
                               clean.to(self.device), y.to(self.device))
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                denoised, emb = model(noisy)
                loss = alpha * self.sisnr(denoised, clean) + beta * metric_loss_fn(emb, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * y.size(0); n += y.size(0)

        val_p1 = self._eval_p1(model, val_dl)
        self.logger.info(f"  [Stage3] ep={ep:02d} loss={loss_sum/n:.4f} P@1={val_p1:.4f}")
        if val_p1 > best_p1:
            best_p1 = val_p1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
```

### B3. EEGNet 1D Embedder — 3 ngày | P@1 +1–3%, latency 100µs → 15µs, params 11M → 35K

**Lý do**: EEGNet (1D) đạt 86.74% trên BED dataset subject-disjoint vs ResNet2D 63.21%. Depthwise-spatial conv học per-channel spatial filters — inductive bias đúng hướng cho EEG.

```python
# experiments/shared/model.py — thêm sau WaveNetDenoiser class:

class EEGNetEmbedder(nn.Module):
    """
    EEGNet-based 1D embedder cho metric learning.
    Không cần 2D reshape. Depthwise spatial conv học per-channel filters.

    Input:  (B, C, T) — C=4 channels, T=800 time steps
    Output: (B, embed_dim) L2-normalized
    """
    def __init__(self, in_chans: int = 4, T: int = 800,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 embed_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, T // 4),
                      padding=(0, T // 8), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Block 2: Depthwise spatial (channel-mixing)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, kernel_size=(in_chans, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 3: Separable temporal
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D * F1, F2, kernel_size=(1, T // 32),
                      padding=(0, T // 64), bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Compute flattened size
        flat = self._get_flat_size(in_chans, T)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def _get_flat_size(self, C, T):
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self.separable_conv(self.spatial_conv(self.temporal_conv(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B,C,T) → (B,1,C,T)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return F.normalize(self.head(x), p=2, dim=1)
```

```python
# experiments/shared/model.py — cập nhật create_metric_model():
def create_metric_model(backbone: str = "resnet18", n_channels: int = 4,
                        embed_dim: int = 256, pretrained: bool = True,
                        use_mamba: bool = False,
                        embedder_type: str = "resnet") -> EEGMetricModel:
    denoiser = WaveNetDenoiser(channels=n_channels, use_mamba=use_mamba)
    if embedder_type == "eegnet":
        embedder = EEGNetEmbedder(in_chans=n_channels, T=800, embed_dim=embed_dim)
    else:
        embedder = ResNetMetricEmbedder(backbone=backbone, in_chans=n_channels,
                                        embed_dim=embed_dim, pretrained=pretrained)
    return EEGMetricModel(denoiser, embedder)
```

### B4. Delta Band SincNet Branch — 2 ngày | AUROC +8–12%

Delta (0.5–4 Hz) và theta (4–8 Hz) là bands session-stable nhất cho biometrics. Branch này nhận **raw signal TRƯỚC denoiser** (delta power không bị noise che khuất):

```python
# experiments/shared/model.py — thêm sau EEGNetEmbedder:

class DeltaBandBranch(nn.Module):
    """
    Parallel spectral branch với learnable SincNet-style bandpass filters.
    Khởi tạo trong dải delta-theta (0.5–8 Hz) — band discriminative nhất.

    Input:  (B, C, T) — raw signal, TRƯỚC denoiser
    Output: (B, spectral_dim) L2-normalized
    """
    def __init__(self, in_chans: int = 4, T: int = 800,
                 n_filters: int = 16, spectral_dim: int = 64):
        super().__init__()
        # SincNet-style learnable filters initialized in delta-theta range
        self.low_hz = nn.Parameter(torch.linspace(0.5, 7.0, n_filters))
        self.band_hz = nn.Parameter(torch.ones(n_filters) * 2.0)
        kernel_size = min(251, T // 4) | 1  # odd kernel

        self.conv = nn.Conv1d(in_chans, in_chans * n_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=in_chans, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_chans * n_filters, spectral_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(spectral_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.elu(self.conv(x))
        out = self.pool(out)
        return F.normalize(self.head(out), p=2, dim=1)


# Cập nhật EEGMetricModel để hỗ trợ spectral branch:
class EEGMetricModel(nn.Module):
    def __init__(self, filter_model, embedder_model,
                 spectral_branch=None, fuse_dim=None):
        super().__init__()
        self.denoiser = filter_model
        self.embedder = embedder_model
        self.spectral = spectral_branch
        if spectral_branch is not None:
            self.fusion = nn.Sequential(
                nn.Linear(fuse_dim, fuse_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(fuse_dim // 2, fuse_dim // 2),
                nn.BatchNorm1d(fuse_dim // 2),
            )

    def forward(self, x):
        denoised = self.denoiser(x)
        emb = self.embedder(denoised)
        if self.spectral is not None:
            spec_emb = self.spectral(x)   # raw signal, pre-denoiser
            emb = F.normalize(
                self.fusion(torch.cat([emb, spec_emb], dim=1)), p=2, dim=1
            )
        return denoised, emb
```

### B5. Subject-Balanced Batch Sampler — 2 ngày | AUROC +10–15%

```python
# experiments/shared/pipeline.py — thêm class sau imports:

class SubjectBalancedSampler(torch.utils.data.Sampler):
    """
    Đảm bảo mỗi batch có K subjects × M samples/subject.
    Tạo hard negatives tự nhiên cho SupCon/MultiSim loss.
    """
    def __init__(self, labels: torch.Tensor, K: int = 5, M: int = 13):
        self.labels = labels.numpy()
        self.K, self.M = K, M
        self.classes = np.unique(self.labels)
        self.class_idx = {c: np.where(self.labels == c)[0].tolist()
                          for c in self.classes}

    def __iter__(self):
        batches = []
        n_batches = min(len(v) for v in self.class_idx.values()) // self.M
        for _ in range(n_batches):
            chosen = np.random.choice(
                self.classes, size=min(self.K, len(self.classes)), replace=False
            )
            batch = []
            for c in chosen:
                idxs = np.random.choice(
                    self.class_idx[c], size=self.M,
                    replace=len(self.class_idx[c]) < self.M
                )
                batch.extend(idxs.tolist())
            batches.extend(batch)
        return iter(batches)

    def __len__(self):
        return (min(len(v) for v in self.class_idx.values()) // self.M) * self.K * self.M


# Trong _create_split_dataloaders(), thay training DataLoader:
train_sampler = SubjectBalancedSampler(y_tr, K=min(5, len(train_subs)), M=13)
train_dl = DataLoader(TensorDataset(Xn_tr, Xc_tr, y_tr),
                      batch_sampler=train_sampler, **loader_kwargs)
```

---

## Phase C — Foundation Model Overhaul (2–4 tuần)

### C1. EEGPT Foundation Model Embedder — 2 tuần | P@1 +2–4%, AUROC +3–6%

EEGPT (NeurIPS 2024) được pretrain trên EEG thực — không phải ImageNet. Đây là change đơn lẻ có ROI cao nhất.

```bash
# Tải pretrained weights:
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('wodediaodan/EEGPT', 'eegpt_base.pth',
                local_dir='/root/Neuro-Biometrics/weights/')
"
```

```python
# experiments/shared/eegpt_adapter.py (file mới):
"""
EEGPT Adapter: project 4-channel EEG vào không gian 64-channel của EEGPT.
Sử dụng channel identity mapping cho T7/F8/Cz/P4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vị trí của 4 channels trong 64-channel standard 10-20:
CHANNEL_MAP = {'T7': 47, 'F8': 19, 'Cz': 30, 'P4': 51}

class EEGPTEmbedder(nn.Module):
    def __init__(self, pretrained_path: str, embed_dim: int = 256,
                 freeze_trunk: bool = True):
        super().__init__()
        # Project 4ch → 64ch space (identity for the 4 known positions)
        self.channel_proj = nn.Linear(4, 64, bias=False)
        nn.init.zeros_(self.channel_proj.weight)
        for new_idx, orig_idx in enumerate(CHANNEL_MAP.values()):
            self.channel_proj.weight.data[orig_idx, new_idx] = 1.0

        # Load EEGPT trunk (replace with actual EEGPT class import)
        # from eegpt import EEGPT
        # self.trunk = EEGPT.from_pretrained(pretrained_path)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, T) → (B, 64, T)
        x_64 = self.channel_proj(x.transpose(1, 2)).transpose(1, 2)
        # trunk_out = self.trunk(x_64)
        # return F.normalize(self.head(trunk_out), p=2, dim=1)
        raise NotImplementedError("Load EEGPT and uncomment trunk lines")
```

**Fine-tuning schedule:**
- Epochs 1–10: freeze trunk, train head + channel_proj
- Epochs 11–20: unfreeze top-4 transformer blocks
- Epochs 21–30: full unfreeze với LR=1e-5

### C2. Session-Invariant Denoiser Objective (DAAE) — 1 tuần | AUROC +5–8%, EER -4–6%

Thay đổi Stage 1 objective từ "reconstruct clean signal" → "reconstruct session-invariant representation":

```python
# experiments/shared/datapreprocessor.py:378 — preserve session ID:
y.append((subject, exp))   # was: y.append(subject)

# experiments/shared/trainer.py — SessionInvariantLoss class mới:
class SessionInvariantLoss(nn.Module):
    """
    Pulls together embeddings của cùng subject từ different sessions.
    Pushes apart embeddings của different subjects.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, emb: torch.Tensor,
                subject_ids: torch.Tensor,
                session_ids: torch.Tensor) -> torch.Tensor:
        sim = (emb @ emb.T) / self.T
        sim.fill_diagonal_(-1e9)
        same_sub = subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)
        diff_ses = session_ids.unsqueeze(0) != session_ids.unsqueeze(1)
        pos_mask = same_sub & diff_ses
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        log_probs = F.log_softmax(sim, dim=1)
        return -(log_probs * pos_mask.float()).sum() / pos_mask.sum()
```

### C3. EEG Conformer Multi-Scale Embedder — 1 tuần | P@1 +1–3%, AUROC +3–5%

```python
# experiments/shared/model.py — EEGConformerEmbedder class:
class EEGConformerEmbedder(nn.Module):
    """
    Multi-scale temporal EEG embedder kết hợp:
    - Local: 3 parallel dilated Conv1D ở scales T//8, T//16, T//32
    - Global: 2-layer Transformer encoder

    Input: (B, 4, 800) | Output: (B, embed_dim) L2-normalized
    """
    def __init__(self, in_chans: int = 4, T: int = 800,
                 embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        d = 64
        def conv_block(k):
            return nn.Sequential(
                nn.Conv1d(in_chans, d, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(d), nn.ELU(), nn.Dropout(dropout))

        self.s1 = conv_block(T // 8)
        self.s2 = conv_block(T // 16)
        self.s3 = conv_block(T // 32)
        self.merge = nn.Sequential(
            nn.Conv1d(d * 3, d, 1), nn.BatchNorm1d(d), nn.ELU())
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self.s1[0].weight.shape[2]
        s1 = self.s1(x)
        s2 = F.interpolate(self.s2(x), size=s1.shape[-1], mode='linear', align_corners=False)
        s3 = F.interpolate(self.s3(x), size=s1.shape[-1], mode='linear', align_corners=False)
        x = self.merge(torch.cat([s1, s2, s3], dim=1))
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        return F.normalize(self.head(self.pool(x)), p=2, dim=1)
```

---

## Top 5 Improvements Không Có Trong Roadmap Gốc

| # | Improvement | Complexity | Expected Gain | Đã Có Trong Roadmap? |
|---|---|---|---|---|
| 🥇 | **EEGPT Foundation Model** (Phase C1) | 3/5 | AUROC +25–40%, P@1 +10–15% | ❌ Không |
| 🥈 | **Session-Invariant Denoiser** (Phase C2) | 2/5 | EER -15–25%, AUROC +5–8% | ❌ Không |
| 🥉 | **TTA BN Adaptation** (Phase A3) | 1/5 | AUROC +5–10%, EER -4–6% | ❌ Không |
| 4 | **Prototypical Inference** (đã có code) | 1/5 | Open enrollment | ❌ Không |
| 5 | **Delta band focus** (Phase B4 — fix frequency) | 2/5 | AUROC +8–12% | ⚠️ Có nhưng sai bands |

### Promote Prototypical Inference (1 giờ, đã có code)

```python
# experiments/shared/trainer.py — compute_centroids() đã có tại dòng ~347
# accuracy_centroid() đã có tại dòng ~72
# Chỉ cần đặt đây làm PRIMARY inference path thay vì secondary metric:

# Trong evaluate_comprehensive(), thay primary prediction bằng:
centroids = self.compute_centroids(model, train_dl, num_classes)
# Classify by nearest centroid → prototypical inference
```

---

## Thứ Tự Implementation Được Đề Xuất

```
NGAY HÔM NAY (2–3 giờ):
  ✦ A1: Fix ArcFace hardcode bug → chạy lại để confirm gain
  ✦ A5: Fix filter_low=0.5 (prerequisite cho delta band)
  ✦ A2: embed_dim=256
  ✦ A6: 5 seeds

TUẦN 1:
  ✦ A3: TTA (highest AUROC ROI không cần kiến trúc mới)
  ✦ A4: Time shift + channel dropout

TUẦN 2:
  ✦ B4: Delta band SincNet branch (AUROC ROI cao nhất Phase B)
  ✦ B1: Hybrid SupCon (sau B4 để không bottlenecked by architecture)

TUẦN 3:
  ✦ B5: Subject-balanced sampler
  ✦ B3: EEGNet 1D embedder
  ✦ B2: Joint fine-tune Stage 3

TUẦN 4–6:
  ✦ C2: Session-invariant loss (data prep trước)
  ✦ C1: EEGPT (tải weights, integrate)
  ✦ C3: Conformer embedder

KHÔNG NÊN LÀM:
  ✗ PhysioNet pretraining (domain mismatch 64ch motor imagery vs 4ch resting)
  ✗ Mixup augmentation (N=14 subjects, có thể harmful)
  ✗ Memory bank hard mining (không có đủ subjects để có tác dụng)
  ✗ Hyperbolic embeddings (EEG biometrics không có cấu trúc hierarchy)
```

---

## Complete Change Table

| | Improvement | File | Expected P@1 | Expected AUROC | Expected EER | Effort |
|---|---|---|---|---|---|---|
| A1 | Fix ArcFace s=64/m=0.5 bug ⚡ | `trainer.py:205`, `dp.py:83` | +2–4% | +4–7% | -3–5% | 2h |
| A2 | embed_dim 128→256 | `dp.py:75` | +1–2% | +1–2% | -1% | 30m |
| A3 | TTA BN adaptation ⚡ | `trainer.py:~388` | +1–2% | +5–10% | -4–6% | 3h |
| A4 | Time shift + channel dropout | `trainer.py:130` | +1–3% | +2–3% | -2–3% | 4h |
| A5 | Fix filter_low=0.5 ⚡ | `dp.py:55` | — | prerequisite | — | 5m |
| A6 | 5 seeds + CI95 | `main.py:11`, `pipeline.py:116` | — | — | — | 1h |
| B1 | Hybrid ArcFace + SupCon | `trainer.py:204` | +1% | +12–18% | -8–12% | 2d |
| B2 | Joint fine-tune Stage 3 | `trainer.py:after 291` | +3–5% | +5–8% | -4–6% | 2d |
| B3 | EEGNet 1D embedder | `model.py:after 115` | +1–3% | +3–5% | -3–5% | 3d |
| B4 | Delta band SincNet branch | `model.py:after EEGNet` | +2–3% | +8–12% | -5–8% | 2d |
| B5 | Subject-balanced sampler | `pipeline.py:after 165` | +1–2% | +10–15% | -6–10% | 2d |
| C1 | EEGPT foundation model | new `eegpt_adapter.py` | +2–4% | +3–6% | -3–5% | 2w |
| C2 | Session-invariant denoiser loss | `trainer.py:new`, `dp.py:378` | +1–2% | +5–8% | -4–6% | 1w |
| C3 | EEG Conformer multi-scale | `model.py:after EEGNet` | +1–3% | +3–5% | -2–4% | 1w |

⚡ = Quick win, implement ngay hôm nay

---

## Sources

- Deng et al. (2019). ArcFace: Additive Angular Margin Loss. CVPR. arxiv.org/abs/1801.07698
- Lawhern et al. (2018). EEGNet: A compact CNN for EEG-based BCIs. J Neural Eng. arxiv.org/abs/1611.08024
- Liu et al. (2024). EEGPT: Pretrained Transformers for EEG. NeurIPS. arxiv.org/abs/2401.12291
- Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS. arxiv.org/abs/2004.11362
- Wang et al. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR.
- Song et al. (2022). EEG Conformer. IEEE TNNLS. arxiv.org/abs/2010.00274
- Jiang et al. (2024). LaBraM: Large Brain Model. ICLR. openreview.net/forum?id=QzTpTRVtrP
- PMC9735871. EEG Biometric Identification on Raspberry Pi (BED dataset). pmc.ncbi.nlm.nih.gov
- MSGM Paper (2026). Multi-Scale Spatiotemporal Graph Mamba. Frontiers in Neuroscience.
- DAAE. Domain-Adaptive Autoencoder for EEG Biometrics. mdpi.com
- DCTAU. All Beings Are Equal in Open Set Recognition. arxiv.org
- AES-MBE (2026). 4-electrode EEG biometrics 98.82% accuracy. pmc.ncbi.nlm.nih.gov
