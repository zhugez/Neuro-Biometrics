"""
BimodalTrainer: Two-stage trainer extended for bimodal (EEG + spectrogram) inputs.

Stage 1: WaveNet+Mamba denoiser with SI-SNR loss (reused from TwoStageTrainer).
Stage 2: Bimodal embedding + cross-attention fusion with ArcFace loss.
"""
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from .datapreprocessor import Config, get_logger
from .dataset_spectrogram import compute_spectrogram, normalize_spectrograms
from .trainer import TwoStageTrainer, TRAINING_CONFIG
try:
    from pytorch_metric_learning.losses import ArcFaceLoss, MultiSimilarityLoss
    HAS_METRIC = True
except ImportError:
    HAS_METRIC = False


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BIMODAL_TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    "stage2_epochs": 30,
    "differential_lr": {
        "embedder": 3e-4,
        "fusion": 2e-4,
        "spec_embedder": 1e-4,
    },
    "early_stop_metric": "p1",
    "early_stop_patience": 7,
    "early_stop_delta": 0.001,
}


class BimodalTrainer(TwoStageTrainer):
    """
    Two-stage trainer extended for bimodal (EEG + spectrogram) inputs.

    Stage 1: Train denoiser with SI-SNR loss (inherited from TwoStageTrainer).
    Stage 2: Train EEG embedder + SpectrogramMambaBranch + CrossAttentionFusion
              with ArcFace loss. Frozen denoiser. Differential LR per branch.
              Early stopping on validation AUROC.
    """

    def __init__(self, config: Config, logger=None):
        super().__init__(config, logger)
        # spectrogram_source: "noisy" (default) or "denoised"
        self.spec_source = getattr(config, "spectrogram_source", "noisy")

    def train(self, model, train_dl, val_dl, num_classes,
              loss_type="arcface", noise_type="", model_name="", seed=0,
              train_stage1=True):
        """
        Overridden to use _train_stage2_bimodal instead of the parent's _train_stage2.
        """
        from .trainer import CaseMetrics
        metrics = CaseMetrics(noise_type=noise_type, model_name=model_name)
        if train_stage1:
            print(f"  [Stage 1] Training Denoiser (SI-SNR)...")
            self._train_stage1(
                model.denoiser, train_dl, val_dl, metrics,
                epochs=getattr(self.config, "stage1_epochs", None),
            )
        else:
            print("  [Stage 1] Using cached denoiser state.")
        print(f"  [Stage 2] Training Bimodal Embedder + Fusion (Denoiser Frozen)...")
        self._train_stage2_bimodal(
            model, train_dl, val_dl, num_classes, loss_type, metrics, seed=seed
        )
        return metrics

    # ------------------------------------------------------------------
    # Stage 2: Bimodal embedding + fusion
    # ------------------------------------------------------------------
    def _train_stage2_bimodal(self, model, train_dl, val_dl, num_classes,
                              loss_type, metrics, seed=0):
        """
        Stage 2 training with differential learning rates and AUROC early stopping.

        Dataloader returns: (noisy_EEG, clean_EEG, labels, spectrograms)
        - noisy_EEG: (B, 4, 800)
        - clean_EEG: (B, 4, 800)
        - labels: (B,)
        - spectrograms: (B, 4, 65, 13)
        """
        if not HAS_METRIC:
            raise ImportError("pip install pytorch-metric-learning")

        model = model.to(self.device)
        device = self.device

        # Freeze denoiser (eval mode, requires_grad=False)
        for p in model.denoiser.parameters():
            p.requires_grad = False
        model.denoiser.eval()

        # Separate parameter groups for differential learning rates
        embedder_params = list(model.embedder.parameters())
        fusion_params = list(model.fusion.parameters())
        spec_embedder_params = list(model.spec_embedder.parameters())

        default_diff_lr = BIMODAL_TRAINING_CONFIG["differential_lr"]
        diff_lr = {
            "embedder": getattr(self.config, "learning_rate", default_diff_lr["embedder"]),
            "fusion": getattr(self.config, "fusion_lr", default_diff_lr["fusion"]),
            "spec_embedder": getattr(self.config, "spec_embedder_lr", default_diff_lr["spec_embedder"]),
        }
        param_groups = [
            {"params": embedder_params, "lr": diff_lr["embedder"]},
            {"params": fusion_params, "lr": diff_lr["fusion"]},
            {"params": spec_embedder_params, "lr": diff_lr["spec_embedder"]},
        ]

        # Metric loss
        if loss_type == "arcface":
            metric_loss = ArcFaceLoss(
                num_classes, self.config.embed_dim,
                margin=getattr(self.config, "arcface_margin", 0.3),
                scale=getattr(self.config, "arcface_scale", 30.0),
            ).to(device)
            param_groups.append({"params": list(metric_loss.parameters()),
                                  "lr": diff_lr["embedder"]})
        else:
            metric_loss = MultiSimilarityLoss(alpha=2, beta=50, base=0.5).to(device)

        optimizer = optim.Adam(
            param_groups,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=BIMODAL_TRAINING_CONFIG["scheduler_T0"],
            T_mult=BIMODAL_TRAINING_CONFIG["scheduler_Tmult"],
        )

        # AMP setup
        use_amp = BIMODAL_TRAINING_CONFIG["use_amp"] and device.type == "cuda"
        amp_dtype = (torch.bfloat16 if getattr(self.config, "optimize_h100", False)
                     else torch.float16)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(use_amp and amp_dtype == torch.float16)
        )

        # Early stopping can track retrieval P@1, AUROC, or a weighted blend.
        early_stop_metric = getattr(
            self.config,
            "early_stop_metric",
            BIMODAL_TRAINING_CONFIG["early_stop_metric"],
        )
        best_score = 0.0
        best_p1 = 0.0
        best_auroc = 0.0
        best_state = None
        patience_cnt = 0
        early_stop_delta = BIMODAL_TRAINING_CONFIG["early_stop_delta"]
        early_stop_patience = BIMODAL_TRAINING_CONFIG["early_stop_patience"]
        aux_eeg_weight = float(getattr(self.config, "aux_eeg_loss_weight", 0.3))
        aux_spec_weight = float(getattr(self.config, "aux_spec_loss_weight", 0.2))

        for ep in range(1, self.config.epochs + 1):
            start = time.time()
            model.embedder.train()
            model.spec_embedder.train()
            model.fusion.train()

            loss_sum, n = 0.0, 0
            aug_warmup = int(getattr(self.config, "aug_warmup_epochs", 0))
            aug_strength = 0.0 if ep <= aug_warmup else 1.0

            for batch in train_dl:
                # Bimodal batch: (noisy_EEG, clean_EEG, labels, spectrograms)
                noisy_eeg, clean_eeg, y, spectrograms = batch
                noisy_eeg = noisy_eeg.to(device)
                clean_eeg = clean_eeg.to(device)
                y = y.to(device)
                spectrograms = spectrograms.to(device=device, dtype=noisy_eeg.dtype)

                optimizer.zero_grad()

                # Denoise with frozen denoiser (no_grad)
                with torch.no_grad():
                    denoised = model.denoiser(noisy_eeg)  # (B, 4, 800)

                    spectrograms = self._spectrograms_for_source(denoised, spectrograms)

                # Augment denoised EEG
                denoised_aug = self._augment_batch(denoised, strength=aug_strength)

                # Bimodal forward pass
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    fused_emb, eeg_emb, spec_emb = model.encode(
                        denoised_aug, spectrograms, return_branches=True
                    )
                    loss = metric_loss(fused_emb, y)
                    if aux_eeg_weight:
                        loss = loss + aux_eeg_weight * metric_loss(eeg_emb, y)
                    if aux_spec_weight:
                        loss = loss + aux_spec_weight * metric_loss(spec_emb, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Collect all trainable params for gradient clipping
                trainable_params = (
                    embedder_params + fusion_params + spec_embedder_params +
                    (list(metric_loss.parameters()) if loss_type == "arcface" else [])
                )
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=BIMODAL_TRAINING_CONFIG["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()

                loss_sum += loss.item() * y.size(0)
                n += y.size(0)

            scheduler.step()

            # Compute AUROC for early stopping
            val_auroc = self._eval_auroc(model, val_dl)
            val_p1 = self._eval_p1_bimodal(model, val_dl)
            val_score = self._select_early_stop_score(
                early_stop_metric, val_p1=val_p1, val_auroc=val_auroc
            )
            elapsed = time.time() - start

            metrics.stage2_epochs.append({
                "epoch": ep,
                "loss": round(loss_sum / n, 4),
                "val_p1": round(val_p1, 4),
                "val_auroc": round(val_auroc, 4),
                "early_stop_metric": early_stop_metric,
                "val_score": round(val_score, 4),
                "best_score": round(max(best_score, val_score), 4),
                "time": round(elapsed, 1),
                "lr_embedder": diff_lr["embedder"],
            })

            print(f"    Ep {ep:02d} | Loss: {loss_sum/n:.4f} | P@1: {val_p1:.4f} "
                  f"| AUROC: {val_auroc:.4f} | Score({early_stop_metric}): {val_score:.4f} "
                  f"| Best: {max(best_score, val_score):.4f} | {elapsed:.1f}s")

            if val_score > best_score + early_stop_delta:
                best_score, best_p1, best_auroc, patience_cnt = val_score, val_p1, val_auroc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= early_stop_patience:
                    print(f"    Early stopping at epoch {ep}")
                    break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)
            os.makedirs("weights", exist_ok=True)
            weight_path = (
                f"weights/best_v4_bimodal_{metrics.noise_type}_{metrics.model_name}"
                f"_seed{seed}.pth"
            )
            checkpoint = {
                "model_state_dict": best_state,
                "config": {k: v for k, v in self.config.__dict__.items()
                           if not k.startswith("_")},
                "metrics": {
                    "early_stop_metric": early_stop_metric,
                    "best_score": best_score,
                    "best_p1": best_p1,
                    "best_auroc": best_auroc,
                },
            }
            torch.save(checkpoint, weight_path)
            print(f"    Saved best model checkpoint to: {weight_path}")

    # ------------------------------------------------------------------
    # Spectrogram computation utility
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_spectrogram(eeg: torch.Tensor, n_fft: int = 128,
                              hop_length: int = 64) -> torch.Tensor:
        return compute_spectrogram(eeg, n_fft=n_fft, hop_length=hop_length)

    def _spectrograms_for_source(
        self,
        denoised: torch.Tensor,
        provided_spectrograms: torch.Tensor,
    ) -> torch.Tensor:
        if self.spec_source == "denoised":
            specs = self._compute_spectrogram(
                denoised,
                n_fft=getattr(self.config, "spectrogram_n_fft", 128),
                hop_length=getattr(self.config, "spectrogram_hop_length", 64),
            )
            return normalize_spectrograms(specs)
        return provided_spectrograms

    @staticmethod
    def _select_early_stop_score(metric: str, val_p1: float, val_auroc: float) -> float:
        if metric == "auroc":
            return val_auroc
        if metric == "combined":
            return 0.7 * val_p1 + 0.3 * val_auroc
        return val_p1

    @torch.no_grad()
    def forward_bimodal(self, model, noisy: torch.Tensor,
                        spectrograms: torch.Tensor):
        noisy = noisy.to(self.device)
        spectrograms = spectrograms.to(device=self.device, dtype=noisy.dtype)
        denoised = model.denoiser(noisy)
        spectrograms = self._spectrograms_for_source(denoised, spectrograms)
        emb = model.encode(denoised, spectrograms)
        return denoised, emb

    # ------------------------------------------------------------------
    # Evaluation helpers (bimodal-aware)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_p1_bimodal(self, model, dl: DataLoader) -> float:
        """Precision@1 for bimodal model."""
        model.eval()
        embs, lbls = [], []
        for noisy, _, y, spectrograms in dl:
            _, emb = self.forward_bimodal(model, noisy, spectrograms)
            embs.append(emb.cpu())
            lbls.append(y)
        from .trainer import p_at_1
        return p_at_1(torch.cat(embs), torch.cat(lbls))

    @torch.no_grad()
    def _eval_auroc(self, model, dl: DataLoader) -> float:
        """
        Compute AUROC from pairwise cosine distances on fused embeddings.

        Positive pairs: same subject (label match)
        Negative pairs: different subjects (label mismatch)
        Score: cosine similarity between embeddings
        """
        model.eval()
        embs, lbls = [], []
        for noisy, _, y, spectrograms in dl:
            _, emb = self.forward_bimodal(model, noisy, spectrograms)
            embs.append(emb.cpu())
            lbls.append(y)

        embs = torch.cat(embs)
        lbls = torch.cat(lbls)
        n = len(lbls)

        # Pairwise cosine similarity
        sim_matrix = embs @ embs.T  # (n, n)
        label_matrix = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).int()
        mask = ~torch.eye(n, dtype=torch.bool)
        scores = sim_matrix[mask].numpy()
        pair_labels = label_matrix[mask].numpy()

        # Subsample if too large
        if len(scores) > 1_000_000:
            idx = np.random.choice(len(scores), 1_000_000, replace=False)
            scores = scores[idx]
            pair_labels = pair_labels[idx]

        if len(pair_labels) == 0 or len(np.unique(pair_labels)) < 2:
            return 0.5

        return roc_auc_score(pair_labels, scores)

    @torch.no_grad()
    def evaluate_bimodal(self, model, dl, train_dl=None, num_classes=None) -> Dict:
        """
        Comprehensive evaluation for bimodal model.
        Uses fused embeddings for all biometric metrics.
        """
        model.to(self.device).eval()
        embs, lbls = [], []
        sisnr_sum, n = 0.0, 0

        start_time = time.time()
        for noisy, clean, y, spectrograms in dl:
            clean = clean.to(self.device)
            denoised, emb = self.forward_bimodal(model, noisy, spectrograms)

            sisnr_sum += (-self.sisnr(denoised, clean).item()) * y.size(0)
            n += y.size(0)
            embs.append(emb.cpu())
            lbls.append(y)
        end_time = time.time()

        embs = torch.cat(embs)
        lbls = torch.cat(lbls)

        # P@1 and P@5
        cmc, p1, p5 = self._calculate_retrieval_metrics(embs, lbls)

        sim_matrix = embs @ embs.T
        label_matrix = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).int()
        mask = ~torch.eye(len(lbls), dtype=torch.bool)
        scores = sim_matrix[mask].numpy()
        pair_labels = label_matrix[mask].numpy()

        if len(scores) > 1_000_000:
            idx = np.random.choice(len(scores), 1_000_000, replace=False)
            scores = scores[idx]
            pair_labels = pair_labels[idx]

        if len(pair_labels) == 0 or len(np.unique(pair_labels)) < 2:
            val_auroc = 0.5
            eer = 0.0
        else:
            val_auroc = roc_auc_score(pair_labels, scores)
            fpr, tpr, _ = roc_curve(pair_labels, scores)
            eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])

        result = {
            "p@1": p1, "p@5": p5,
            "si_snr": sisnr_sum / n,
            "auroc": val_auroc,
            "eer": eer,
            "latency": (end_time - start_time) * 1000 / n,
            "params": sum(p.numel() for p in model.parameters()),
            "cmc": cmc.tolist(),
        }

        if train_dl is not None and num_classes is not None:
            centroids = self.compute_centroids_bimodal(model, train_dl, num_classes)
            from .trainer import accuracy_centroid
            result["accuracy"] = accuracy_centroid(embs, lbls, centroids.cpu())

        return result

    @torch.no_grad()
    def compute_centroids_bimodal(self, model, train_dl, num_classes) -> torch.Tensor:
        """Compute class centroids for bimodal model."""
        model.to(self.device).eval()
        emb_by_class = {c: [] for c in range(num_classes)}
        for noisy, _, y, spectrograms in train_dl:
            _, emb = self.forward_bimodal(model, noisy, spectrograms)
            for e, c in zip(emb.cpu(), y):
                emb_by_class[c.item()].append(e)
        centroids = []
        for c in range(num_classes):
            if emb_by_class[c]:
                centroids.append(torch.stack(emb_by_class[c]).mean(0))
            else:
                centroids.append(torch.zeros(self.config.embed_dim))
        return torch.stack(centroids)
