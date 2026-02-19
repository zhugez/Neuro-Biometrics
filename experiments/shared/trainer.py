"""
Two-stage trainer for EEG denoising + metric learning.

Stage 1: Train denoiser with SI-SNR loss.
Stage 2: Freeze denoiser, train embedder with ArcFace/MultiSimilarity loss.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, det_curve, roc_auc_score, average_precision_score

from .datapreprocessor import Config, get_logger

try:
    from pytorch_metric_learning.losses import ArcFaceLoss, MultiSimilarityLoss
    HAS_METRIC = True
except ImportError:
    HAS_METRIC = False

# ---------------------------------------------------------------------------
# Training hyperparameters (module-level for easy override in tests)
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "stage1_epochs": 30,
    "grad_clip_norm": 1.0,
    "use_amp": True,
    "scheduler_T0": 5,
    "scheduler_Tmult": 2,
    "early_stop_delta": 0.001,
}


# ---------------------------------------------------------------------------
# Loss & metric helpers
# ---------------------------------------------------------------------------
class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        B, C, T = est.shape
        est, ref = est.view(B * C, T), ref.view(B * C, T)
        ref0 = ref - ref.mean(1, keepdim=True)
        est0 = est - est.mean(1, keepdim=True)
        dot = (est0 * ref0).sum(1, keepdim=True)
        s = dot * ref0 / ((ref0 ** 2).sum(1, keepdim=True) + self.eps)
        e = est0 - s
        return -10 * torch.log10(
            (s ** 2).sum(1) / ((e ** 2).sum(1) + self.eps) + self.eps
        ).mean()


def p_at_1(emb: torch.Tensor, lbl: torch.Tensor) -> float:
    """Precision@1 via nearest-neighbor."""
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    return (lbl == lbl[sim.argmax(1)]).float().mean().item()


def accuracy_centroid(emb: torch.Tensor, lbl: torch.Tensor,
                      centroids: torch.Tensor) -> float:
    """Centroid-based classification accuracy."""
    dist = torch.cdist(emb, centroids, p=2)
    pred = dist.argmin(dim=1)
    return (pred == lbl).float().mean().item()


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------
@dataclass
class CaseMetrics:
    noise_type: str = ""
    model_name: str = ""
    stage1_epochs: List[Dict] = field(default_factory=list)
    stage2_epochs: List[Dict] = field(default_factory=list)
    test_p1: float = 0.0
    test_p5: float = 0.0
    test_sisnr: float = 0.0
    test_accuracy: float = 0.0
    cmc_curve: List[float] = field(default_factory=list)
    roc_curve: Dict = field(default_factory=dict)
    det_curve: Dict = field(default_factory=dict)
    eer: float = 0.0
    model_params: int = 0
    inference_latency: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "noise_type": self.noise_type,
            "model_name": self.model_name,
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "test_results": {
                "p@1": self.test_p1, "p@5": self.test_p5,
                "si_snr": self.test_sisnr, "accuracy": self.test_accuracy,
                "eer": self.eer, "model_params": self.model_params,
                "inference_latency": self.inference_latency,
            },
            "curves": {
                "cmc": self.cmc_curve,
                "roc": self.roc_curve,
                "det": self.det_curve,
            },
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TwoStageTrainer:
    def __init__(self, config: Config, logger=None):
        self.config = config
        self.device = torch.device(config.device)
        self.sisnr = SISNRLoss()
        self.logger = logger or get_logger("eeg.trainer")

    def _augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        """EEG augmentation: gaussian jitter, amplitude scaling, time shift, channel dropout."""
        B, C, T = x.shape
        # Gaussian jitter
        noise = 0.01 * x.std() * torch.randn_like(x)
        # Amplitude scaling
        scale = 0.9 + 0.2 * torch.rand(B, C, 1, device=x.device)
        x = (x + noise) * scale
        # Time shift ±50 samples (±0.25 sec at 200 Hz) — temporal invariance
        shifts = torch.randint(-50, 51, (B,)).tolist()
        x = torch.stack([torch.roll(x[i], shifts[i], dims=-1) for i in range(B)])
        # Channel dropout: zero 1 of C channels with p=0.2 — electrode robustness
        if torch.rand(1).item() < 0.2:
            ch = torch.randint(0, C, (1,)).item()
            x[:, ch, :] = 0.0
        return x

    def train(self, model, train_dl, val_dl, num_classes,
              loss_type="arcface", noise_type="", model_name="") -> CaseMetrics:
        metrics = CaseMetrics(noise_type=noise_type, model_name=model_name)
        print(f"  [Stage 1] Training Denoiser (SI-SNR)...")
        self._train_stage1(model.denoiser, train_dl, val_dl, metrics)
        print(f"  [Stage 2] Training Embedder (Denoiser Frozen)...")
        self._train_stage2(model, train_dl, val_dl, num_classes, loss_type, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Stage 1: Denoiser
    # ------------------------------------------------------------------
    def _train_stage1(self, denoiser, train_dl, val_dl, metrics, epochs=None):
        epochs = epochs or TRAINING_CONFIG["stage1_epochs"]
        denoiser = denoiser.to(self.device)
        opt = optim.Adam(denoiser.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=TRAINING_CONFIG["scheduler_T0"],
            T_mult=TRAINING_CONFIG["scheduler_Tmult"],
        )

        use_amp = TRAINING_CONFIG["use_amp"] and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for ep in range(1, epochs + 1):
            start = time.time()
            denoiser.train()
            loss_sum, n = 0.0, 0
            for noisy, clean, _ in train_dl:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = self.sisnr(denoiser(noisy), clean)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    denoiser.parameters(), max_norm=TRAINING_CONFIG["grad_clip_norm"]
                )
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item() * noisy.size(0)
                n += noisy.size(0)

            scheduler.step()
            val_sisnr = self._eval_sisnr(denoiser, val_dl)
            elapsed = time.time() - start

            metrics.stage1_epochs.append({
                "epoch": ep, "loss": round(loss_sum / n, 4),
                "val_sisnr": round(val_sisnr, 2), "time": round(elapsed, 1),
                "lr": round(scheduler.get_last_lr()[0], 6),
            })
            print(f"    Ep {ep:02d} | Loss: {loss_sum/n:.4f} | SI-SNR: {val_sisnr:.2f} dB "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Stage 2: Embedder (denoiser frozen)
    # ------------------------------------------------------------------
    def _train_stage2(self, model, train_dl, val_dl, num_classes,
                      loss_type, metrics):
        if not HAS_METRIC:
            raise ImportError("pip install pytorch-metric-learning")

        model = model.to(self.device)
        for p in model.denoiser.parameters():
            p.requires_grad = False
        model.denoiser.eval()

        if loss_type == "arcface":
            metric_loss = ArcFaceLoss(
                num_classes, self.config.embed_dim,
                margin=self.config.arcface_margin,
                scale=self.config.arcface_scale,
            ).to(self.device)
            params = list(model.embedder.parameters()) + list(metric_loss.parameters())
        else:
            metric_loss = MultiSimilarityLoss(alpha=2, beta=50, base=0.5).to(self.device)
            params = list(model.embedder.parameters())

        opt = optim.Adam(params, lr=self.config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=TRAINING_CONFIG["scheduler_T0"],
            T_mult=TRAINING_CONFIG["scheduler_Tmult"],
        )

        use_amp = TRAINING_CONFIG["use_amp"] and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_p1, best_state, best_accuracy, patience_cnt = 0.0, None, 0.0, 0
        early_stop_delta = TRAINING_CONFIG["early_stop_delta"]

        for ep in range(1, self.config.epochs + 1):
            start = time.time()
            model.embedder.train()
            loss_sum, n = 0.0, 0

            for noisy, clean, y in train_dl:
                noisy, y = noisy.to(self.device), y.to(self.device)
                opt.zero_grad()
                with torch.no_grad():
                    denoised = model.denoiser(noisy)
                denoised_aug = self._augment_batch(denoised)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    emb = model.embedder(denoised_aug)
                    loss = metric_loss(emb, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    params, max_norm=TRAINING_CONFIG["grad_clip_norm"]
                )
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item() * y.size(0)
                n += y.size(0)

            scheduler.step()
            val_p1 = self._eval_p1(model, val_dl)
            elapsed = time.time() - start

            metrics.stage2_epochs.append({
                "epoch": ep, "loss": round(loss_sum / n, 4),
                "val_p1": round(val_p1, 4),
                "best_p1": round(max(best_p1, val_p1), 4),
                "time": round(elapsed, 1),
                "lr": round(scheduler.get_last_lr()[0], 6),
            })

            val_accuracy = 0.0
            if ep % 5 == 0 or ep == self.config.epochs:
                centroids = self.compute_centroids(model, train_dl, num_classes)
                val_accuracy = self._eval_accuracy(model, val_dl, centroids)

            print(f"    Ep {ep:02d} | Loss: {loss_sum/n:.4f} | P@1: {val_p1:.4f} "
                  f"| Acc: {val_accuracy:.4f} | Best: {max(best_p1, val_p1):.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

            if val_p1 > best_p1 + early_stop_delta:
                best_p1, best_accuracy, patience_cnt = val_p1, val_accuracy, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.config.patience:
                    print(f"    Early stopping at epoch {ep}")
                    break

        if best_state:
            model.load_state_dict(best_state)
            os.makedirs("weights", exist_ok=True)
            weight_path = f"weights/best_{metrics.noise_type}_{metrics.model_name}.pth"
            checkpoint = {
                "model_state_dict": best_state,
                "config": {k: v for k, v in self.config.__dict__.items()
                           if not k.startswith("_")},
                "metrics": {"best_p1": best_p1, "best_accuracy": best_accuracy},
            }
            torch.save(checkpoint, weight_path)
            print(f"    Saved best model checkpoint to: {weight_path}")

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_sisnr(self, denoiser, dl: DataLoader) -> float:
        denoiser.eval()
        sisnr_sum, n = 0.0, 0
        for noisy, clean, _ in dl:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            sisnr_sum += (-self.sisnr(denoiser(noisy), clean).item()) * noisy.size(0)
            n += noisy.size(0)
        return sisnr_sum / n

    @torch.no_grad()
    def _eval_p1(self, model, dl: DataLoader) -> float:
        model.eval()
        embs, lbls = [], []
        for noisy, _, y in dl:
            _, emb = model(noisy.to(self.device))
            embs.append(emb.cpu())
            lbls.append(y)
        return p_at_1(torch.cat(embs), torch.cat(lbls))

    @torch.no_grad()
    def _eval_accuracy(self, model, dl: DataLoader,
                       centroids: torch.Tensor) -> float:
        model.eval()
        embs, lbls = [], []
        for noisy, _, y in dl:
            _, emb = model(noisy.to(self.device))
            embs.append(emb.cpu())
            lbls.append(y)
        return accuracy_centroid(torch.cat(embs), torch.cat(lbls), centroids.cpu())

    def _tta_adapt(self, model) -> None:
        """Enable BN layers to update running stats from test data (TTA warm-up).

        Call model.embedder.train() before forward to let BN absorb test
        distribution, then restore eval() for the actual embedding pass.
        Only BN momentum is used — no gradient update, no label needed.
        """
        for m in model.embedder.modules():
            if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()

    def _eval_p1_with_tta(self, model, dl: DataLoader, tta_steps: int = 1) -> float:
        """Precision@1 with Test-Time BN Adaptation.

        For each batch: one warm-up forward in train() mode updates BN stats,
        then a second forward in eval() uses those adapted stats for embedding.
        This corrects the train/test distribution shift for holdout subjects.
        """
        model.to(self.device)
        embs, lbls = [], []
        for noisy, _, y in dl:
            noisy = noisy.to(self.device)
            # Warm-up: update BN running stats with this test batch (no grad)
            model.embedder.train()
            with torch.no_grad():
                _ = model.embedder(model.denoiser(noisy))
            # Inference: use the just-adapted BN stats
            model.embedder.eval()
            with torch.no_grad():
                _, emb = model(noisy)
            embs.append(emb.cpu())
            lbls.append(y)
        return p_at_1(torch.cat(embs), torch.cat(lbls))

    @torch.no_grad()
    def evaluate(self, model, dl, train_dl=None, num_classes=None) -> Dict:
        model.to(self.device).eval()
        embs, lbls, sisnr_sum, n = [], [], 0.0, 0
        for noisy, clean, y in dl:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            den, emb = model(noisy)
            sisnr_sum += (-self.sisnr(den, clean).item()) * y.size(0)
            n += y.size(0)
            embs.append(emb.cpu())
            lbls.append(y)
        result = {"p@1": p_at_1(torch.cat(embs), torch.cat(lbls)),
                  "si_snr": sisnr_sum / n}
        if train_dl is not None and num_classes is not None:
            centroids = self.compute_centroids(model, train_dl, num_classes)
            result["accuracy"] = self._eval_accuracy(model, dl, centroids)
        return result

    @torch.no_grad()
    def compute_centroids(self, model, train_dl, num_classes) -> torch.Tensor:
        model.to(self.device).eval()
        emb_by_class = {c: [] for c in range(num_classes)}
        for noisy, _, y in train_dl:
            _, emb = model(noisy.to(self.device))
            for e, c in zip(emb.cpu(), y):
                emb_by_class[c.item()].append(e)
        centroids = []
        for c in range(num_classes):
            if emb_by_class[c]:
                centroids.append(torch.stack(emb_by_class[c]).mean(0))
            else:
                centroids.append(torch.zeros(self.config.embed_dim))
        return torch.stack(centroids)

    @torch.no_grad()
    def compute_threshold(self, model, val_dl, centroids,
                          percentile: int = 95) -> float:
        """Compute distance threshold for novelty detection."""
        model.to(self.device).eval()
        centroids = centroids.to(self.device)
        all_dists = []
        for noisy, _, _ in val_dl:
            _, emb = model(noisy.to(self.device))
            dist = torch.cdist(emb, centroids, p=2)
            all_dists.extend(dist.min(dim=1).values.cpu().tolist())
        return float(np.percentile(all_dists, percentile))

    # ------------------------------------------------------------------
    # Comprehensive evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_comprehensive(self, model, dl, train_dl=None,
                               num_classes=None) -> Dict:
        """Full evaluation with CMC, ROC, DET, EER, robustness breakdown."""
        model.to(self.device).eval()

        embs, lbls = [], []
        sisnr_sum, n = 0.0, 0
        input_sisnrs, output_sisnrs = [], []

        start_time = time.time()
        for noisy, clean, y in dl:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            den, emb = model(noisy)

            B, C, T = noisy.shape

            def _batch_sisnr(est, ref):
                est, ref = est.view(B, -1), ref.view(B, -1)
                ref0 = ref - ref.mean(dim=1, keepdim=True)
                est0 = est - est.mean(dim=1, keepdim=True)
                dot = (est0 * ref0).sum(dim=1, keepdim=True)
                s = dot * ref0 / ((ref0 ** 2).sum(dim=1, keepdim=True) + 1e-8)
                e = est0 - s
                return 10 * torch.log10(
                    (s ** 2).sum(dim=1) / ((e ** 2).sum(dim=1) + 1e-8) + 1e-8
                )

            input_sisnrs.append(_batch_sisnr(noisy, clean).cpu())
            output_sisnrs.append(_batch_sisnr(den, clean).cpu())
            sisnr_sum += (-self.sisnr(den, clean).item()) * y.size(0)
            n += y.size(0)
            embs.append(emb.cpu())
            lbls.append(y)

        end_time = time.time()

        embs = torch.cat(embs)
        lbls = torch.cat(lbls)
        input_sisnrs = torch.cat(input_sisnrs)
        output_sisnrs = torch.cat(output_sisnrs)
        delta_sisnrs = output_sisnrs - input_sisnrs

        latency_ms = (end_time - start_time) * 1000 / n
        params = sum(p.numel() for p in model.parameters())

        # Retrieval metrics
        cmc, p1, p5 = self._calculate_retrieval_metrics(embs, lbls)

        # Verification metrics (pairwise cosine similarity)
        sim_matrix = embs @ embs.T
        label_matrix = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).int()
        mask = ~torch.eye(n, dtype=torch.bool)
        scores = sim_matrix[mask].numpy()
        labels = label_matrix[mask].numpy()

        if len(scores) > 1_000_000:
            idx = np.random.choice(len(scores), 1_000_000, replace=False)
            scores, labels = scores[idx], labels[idx]

        if len(labels) == 0 or len(np.unique(labels)) < 2:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            fpr_det = np.array([0.0, 1.0])
            fnr_det = np.array([1.0, 0.0])
            eer = 0.0
        else:
            fpr, tpr, _ = roc_curve(labels, scores)
            fpr_det, fnr_det, _ = det_curve(labels, scores)
            eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        # Robustness breakdown by input SNR
        robustness = {}
        for t_snr in [0, 5, 10, 20]:
            mask_snr = (input_sisnrs >= t_snr - 2.5) & (input_sisnrs < t_snr + 2.5)
            if mask_snr.sum() > 0:
                sub_embs = embs[mask_snr]
                sub_lbls = lbls[mask_snr]
                _, sub_p1, _ = self._calculate_retrieval_metrics(sub_embs, sub_lbls)
                robustness[t_snr] = {
                    "p@1": sub_p1,
                    "delta_sisnr": delta_sisnrs[mask_snr].mean().item(),
                }

        acc = 0.0
        if train_dl is not None and num_classes is not None:
            centroids = self.compute_centroids(model, train_dl, num_classes)
            acc = self._eval_accuracy(model, dl, centroids)

        return {
            "p@1": p1, "p@5": p5,
            "si_snr": sisnr_sum / n, "accuracy": acc,
            "cmc": cmc.tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "det": {"fpr": fpr_det.tolist(), "fnr": fnr_det.tolist()},
            "eer": eer, "latency": latency_ms, "params": params,
            "robustness": robustness,
        }

    def _calculate_retrieval_metrics(self, embs: torch.Tensor,
                                     lbls: torch.Tensor,
                                     k_max: int = 20):
        """CMC curve, Precision@1, Precision@5."""
        n = len(lbls)
        sim = embs @ embs.T
        sim.fill_diagonal_(-1e9)
        _, indices = sim.sort(dim=1, descending=True)
        match_matrix = (lbls[indices] == lbls.unsqueeze(1))

        # CMC: probability of correct match in top-k
        cmc = np.array([
            match_matrix[:, :k].any(dim=1).float().mean().item()
            for k in range(1, k_max + 1)
        ])

        # Precision@K: fraction of top-K that share same identity
        p1 = cmc[0]  # For P@1, CMC@1 == P@1
        p5 = match_matrix[:, :5].float().mean().item() if n >= 5 else cmc[-1]

        return cmc, p1, p5

    @torch.no_grad()
    def evaluate_novelty_comprehensive(self, model, known_dl, unknown_noisy,
                                       centroids, threshold) -> Dict:
        """Evaluate novelty detection with AUROC/AUPR."""
        model.to(self.device).eval()
        centroids = centroids.to(self.device)

        known_dists = []
        for noisy, _, _ in known_dl:
            _, emb = model(noisy.to(self.device))
            dist = torch.cdist(emb, centroids, p=2)
            known_dists.extend(dist.min(dim=1).values.cpu().tolist())

        unknown_dists = []
        batch_size = 64
        for i in range(0, len(unknown_noisy), batch_size):
            batch = unknown_noisy[i:i + batch_size].to(self.device)
            _, emb = model(batch)
            dist = torch.cdist(emb, centroids, p=2)
            unknown_dists.extend(dist.min(dim=1).values.cpu().tolist())

        y_true = [0] * len(known_dists) + [1] * len(unknown_dists)
        y_scores = known_dists + unknown_dists

        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        known_total = len(known_dists)
        unknown_total = len(unknown_dists)
        tar = sum(d < threshold for d in known_dists) / known_total if known_total else 0
        trr = sum(d >= threshold for d in unknown_dists) / unknown_total if unknown_total else 0

        return {
            "tar": tar, "trr": trr, "far": 1 - trr, "frr": 1 - tar,
            "auroc": auroc, "aupr": aupr, "threshold": threshold,
            "known_samples": known_total, "unknown_samples": unknown_total,
        }
