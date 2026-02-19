import json, logging, time
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from datapreprocessor import Config, get_logger
from sklearn.metrics import roc_curve, det_curve, roc_auc_score, average_precision_score
import time

# Training configuration
TRAINING_CONFIG = {
    "stage1_epochs": 30,
    "grad_clip_norm": 1.0,
    "use_amp": True,
    "scheduler_T0": 5,
    "scheduler_Tmult": 2,
    "early_stop_delta": 0.001,
}

try:
    from pytorch_metric_learning.losses import ArcFaceLoss, MultiSimilarityLoss
    HAS_METRIC = True
except ImportError:
    HAS_METRIC = False

class SISNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, est, ref):
        B,C,T = est.shape
        est, ref = est.view(B*C,T), ref.view(B*C,T)
        ref0, est0 = ref - ref.mean(1,keepdim=True), est - est.mean(1,keepdim=True)
        dot = (est0*ref0).sum(1,keepdim=True)
        s = dot*ref0 / ((ref0**2).sum(1,keepdim=True)+self.eps)
        e = est0 - s
        return -10*torch.log10((s**2).sum(1)/((e**2).sum(1)+self.eps)+self.eps).mean()

def p_at_1(emb, lbl):
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    return (lbl == lbl[sim.argmax(1)]).float().mean().item()

def accuracy_centroid(emb, lbl, centroids):
    dist_matrix = torch.cdist(emb, centroids, p=2)
    pred = dist_matrix.argmin(dim=1)
    return (pred == lbl).float().mean().item()

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
    
    # New Metrics
    cmc_curve: List[float] = field(default_factory=list)
    roc_curve: Dict = field(default_factory=dict)  # fpr, tpr, thresholds
    det_curve: Dict = field(default_factory=dict)  # fpr, fnr, thresholds
    eer: float = 0.0
    model_params: int = 0
    inference_latency: float = 0.0
    
    def to_dict(self):
        return {
            "noise_type": self.noise_type,
            "model_name": self.model_name,
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "test_results": {
                "p@1": self.test_p1, 
                "p@5": self.test_p5,
                "si_snr": self.test_sisnr, 
                "accuracy": self.test_accuracy,
                "eer": self.eer,
                "model_params": self.model_params,
                "inference_latency": self.inference_latency
            },
            "curves": {
                "cmc": self.cmc_curve,
                "roc": self.roc_curve,
                "det": self.det_curve
            }
        }

class TwoStageTrainer:
    def __init__(self, config: Config, logger=None):
        self.config = config
        self.device = torch.device(config.device)
        self.sisnr = SISNRLoss()
        self.logger = logger or get_logger("eeg.trainer")
        if not HAS_METRIC: raise ImportError("pip install pytorch-metric-learning")

    def _augment_batch(self, x):
        """Simple EEG augmentation: noise jitter + random amplitude scaling."""
        # Small gaussian noise (1% of signal std)
        noise = 0.01 * x.std() * torch.randn_like(x)
        # Random per-channel amplitude scaling (0.9 ~ 1.1)
        scale = 0.9 + 0.2 * torch.rand(x.size(0), x.size(1), 1, device=x.device)
        return (x + noise) * scale
    
    def train(self, model, train_dl, val_dl, num_classes, loss_type="arcface", 
              noise_type="", model_name="") -> CaseMetrics:
        metrics = CaseMetrics(noise_type=noise_type, model_name=model_name)
        print(f"  [Stage 1] Training Denoiser (SI-SNR)...")
        self._train_stage1(model.denoiser, train_dl, val_dl, metrics)
        print(f"  [Stage 2] Training Embedder (Denoiser Frozen)...")
        self._train_stage2(model, train_dl, val_dl, num_classes, loss_type, metrics)
        return metrics
    
    def _train_stage1(self, denoiser, train_dl, val_dl, metrics, epochs=None):
        epochs = epochs or TRAINING_CONFIG["stage1_epochs"]
        denoiser = denoiser.to(self.device)
        opt = optim.Adam(denoiser.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=TRAINING_CONFIG["scheduler_T0"], T_mult=TRAINING_CONFIG["scheduler_Tmult"])
        
        # Mixed precision scaler
        use_amp = TRAINING_CONFIG["use_amp"] and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        for ep in range(1, epochs+1):
            start = time.time()
            denoiser.train()
            loss_sum, n = 0.0, 0
            for noisy, clean, _ in train_dl:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                opt.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.sisnr(denoiser(noisy), clean)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=TRAINING_CONFIG["grad_clip_norm"])
                scaler.step(opt)
                scaler.update()
                
                loss_sum += loss.item() * noisy.size(0)
                n += noisy.size(0)
            
            scheduler.step()
            val_sisnr = self._eval_sisnr(denoiser, val_dl)
            elapsed = time.time() - start
            
            metrics.stage1_epochs.append({
                "epoch": ep, "loss": round(loss_sum/n, 4), 
                "val_sisnr": round(val_sisnr, 2), "time": round(elapsed, 1),
                "lr": round(scheduler.get_last_lr()[0], 6)
            })
            print(f"    Ep {ep:02d} | Loss: {loss_sum/n:.4f} | SI-SNR: {val_sisnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
    
    def _train_stage2(self, model, train_dl, val_dl, num_classes, loss_type, metrics):
        model = model.to(self.device)
        
        for p in model.denoiser.parameters():
            p.requires_grad = False
        model.denoiser.eval()
        
        if loss_type == "arcface":
            metric_loss = ArcFaceLoss(num_classes, self.config.embed_dim, margin=0.3, scale=30).to(self.device)
            params = list(model.embedder.parameters()) + list(metric_loss.parameters())
        else:
            metric_loss = MultiSimilarityLoss(alpha=2, beta=50, base=0.5).to(self.device)
            params = list(model.embedder.parameters())
        
        opt = optim.Adam(params, lr=self.config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=TRAINING_CONFIG["scheduler_T0"], T_mult=TRAINING_CONFIG["scheduler_Tmult"])
        
        # Mixed precision scaler
        use_amp = TRAINING_CONFIG["use_amp"] and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        best_p1, best_state, best_accuracy, patience = 0.0, None, 0.0, 0
        early_stop_delta = TRAINING_CONFIG["early_stop_delta"]
        
        for ep in range(1, self.config.epochs+1):
            start = time.time()
            model.embedder.train()
            loss_sum, n = 0.0, 0
            
            for noisy, clean, y in train_dl:
                noisy, y = noisy.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                with torch.no_grad():
                    denoised = model.denoiser(noisy)
                
                # Augment denoised signal for better embedding generalization
                denoised_aug = self._augment_batch(denoised)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    emb = model.embedder(denoised_aug)
                    loss = metric_loss(emb, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=TRAINING_CONFIG["grad_clip_norm"])
                scaler.step(opt)
                scaler.update()
                
                loss_sum += loss.item() * y.size(0)
                n += y.size(0)
            
            scheduler.step()
            val_p1 = self._eval_p1(model, val_dl)
            elapsed = time.time() - start
            
            metrics.stage2_epochs.append({
                "epoch": ep, "loss": round(loss_sum/n, 4), 
                "val_p1": round(val_p1, 4), "best_p1": round(max(best_p1, val_p1), 4),
                "time": round(elapsed, 1), "lr": round(scheduler.get_last_lr()[0], 6)
            })
            
            # OPTIMIZATION: Only compute centroids (expensive) at the end or every 10 epochs
            val_accuracy = 0.0
            if ep % 5 == 0 or ep == self.config.epochs:
                centroids = self.compute_centroids(model, train_dl, num_classes)
                val_accuracy = self._eval_accuracy(model, val_dl, centroids)
            
            print(f"    Ep {ep:02d} | Loss: {loss_sum/n:.4f} | P@1: {val_p1:.4f} | Acc: {val_accuracy:.4f} | Best: {max(best_p1, val_p1):.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
            
            # Improved early stopping with delta threshold
            if val_p1 > best_p1 + early_stop_delta:
                best_p1, best_accuracy, patience = val_p1, val_accuracy, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.config.patience:
                    print(f"    Early stopping at epoch {ep}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
            import os
            os.makedirs("weights", exist_ok=True)
            weight_path = f"weights/best_{metrics.noise_type}_{metrics.model_name}_v1.pth"
            # Save full checkpoint for resume capability
            checkpoint = {
                "model_state_dict": best_state,
                "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
                "metrics": {"best_p1": best_p1, "best_accuracy": best_accuracy}
            }
            torch.save(checkpoint, weight_path)
            print(f"    Saved best model checkpoint to: {weight_path}")
    
    @torch.no_grad()
    def _eval_sisnr(self, denoiser, dl):
        denoiser.eval()
        sisnr_sum, n = 0.0, 0
        for noisy, clean, _ in dl:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            sisnr_sum += (-self.sisnr(denoiser(noisy), clean).item()) * noisy.size(0)
            n += noisy.size(0)
        return sisnr_sum / n
    
    @torch.no_grad()
    def _eval_p1(self, model, dl):
        model.eval()
        embs, lbls = [], []
        for noisy, _, y in dl:
            _, emb = model(noisy.to(self.device))
            embs.append(emb.cpu())
            lbls.append(y)
        return p_at_1(torch.cat(embs), torch.cat(lbls))
    
    @torch.no_grad()
    def _eval_accuracy(self, model, dl, centroids):
        model.eval()
        centroids = centroids.to(self.device)
        embs, lbls = [], []
        for noisy, _, y in dl:
            _, emb = model(noisy.to(self.device))
            embs.append(emb.cpu())
            lbls.append(y)
        return accuracy_centroid(torch.cat(embs), torch.cat(lbls), centroids.cpu())
    
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
        result = {"p@1": p_at_1(torch.cat(embs), torch.cat(lbls)), "si_snr": sisnr_sum/n}
        if train_dl is not None and num_classes is not None:
            centroids = self.compute_centroids(model, train_dl, num_classes)
            result["accuracy"] = self._eval_accuracy(model, dl, centroids)
        return result

    @torch.no_grad()
    def compute_centroids(self, model, train_dl, num_classes):
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
    def compute_threshold(self, model, val_dl, centroids, percentile=95):
        """
        Compute distance threshold for novelty detection based on validation set.
        
        Uses the distances from validation embeddings to their nearest centroids,
        returning the given percentile as the threshold. Samples with distances
        above this threshold will be classified as unknown/novel.
        
        Args:
            model: The trained model
            val_dl: Validation DataLoader
            centroids: Class centroids tensor
            percentile: Percentile of distances to use as threshold (default: 95)
            
        Returns:
            float: Distance threshold value
        """
        model.to(self.device).eval()
        centroids = centroids.to(self.device)
        
        all_dists = []
        for noisy, _, _ in val_dl:
            _, emb = model(noisy.to(self.device))
            dist_matrix = torch.cdist(emb, centroids, p=2)
            min_dist = dist_matrix.min(dim=1).values
            all_dists.extend(min_dist.cpu().tolist())
        
        threshold = np.percentile(all_dists, percentile)
        return threshold
    
    @torch.no_grad()
    def evaluate_comprehensive(self, model, dl, train_dl=None, num_classes=None) -> Dict:
        """Evaluate with comprehensive set of metrics and robustness analysis"""
        model.to(self.device).eval()
        
        # 1. Collect Embeddings and Labels
        embs, lbls = [], []
        sisnr_sum, n = 0.0, 0
        input_sisnrs, output_sisnrs, valid_indices = [], [], []
        
        # Define SNR buckets (approximate centers)
        snr_buckets = {0: [], 5: [], 10: [], 20: []}
        
        # Time the inference for latency calculation
        start_time = time.time()
        batch_start_idx = 0
        
        for noisy, clean, y in dl:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            den, emb = model(noisy)
            
            # Compute Input and Output SI-SNR per sample
            # Reshape for broadcasting
            B, C, T = noisy.shape
            
            # Helper to compute SI-SNR per sample
            def compute_batch_sisnr(est, ref):
                # Flatten C, T
                est = est.view(B, -1)
                ref = ref.view(B, -1)
                ref0 = ref - ref.mean(dim=1, keepdim=True)
                est0 = est - est.mean(dim=1, keepdim=True)
                dot = (est0 * ref0).sum(dim=1, keepdim=True)
                ref_norm = (ref0 ** 2).sum(dim=1, keepdim=True) + 1e-8
                s = dot * ref0 / ref_norm
                e = est0 - s
                # SI-SNR
                return 10 * torch.log10((s ** 2).sum(dim=1) / ((e ** 2).sum(dim=1) + 1e-8) + 1e-8)

            in_snr = compute_batch_sisnr(noisy, clean).cpu()
            out_snr = compute_batch_sisnr(den, clean).cpu()

            # Accumulate Total Loss (using the mean formulation from SISNRLoss class)
            # Re-using the class loss for consistency in 'si_snr' aggregate metric
            batch_loss = self.sisnr(den, clean).item()
            sisnr_sum += (-batch_loss) * y.size(0)
            n += y.size(0)
            
            embs.append(emb.cpu())
            lbls.append(y)
            
            # Store for robustness
            input_sisnrs.append(in_snr)
            output_sisnrs.append(out_snr)
        
        end_time = time.time()
        
        embs = torch.cat(embs)
        lbls = torch.cat(lbls)
        input_sisnrs = torch.cat(input_sisnrs)
        output_sisnrs = torch.cat(output_sisnrs)
        delta_sisnrs = output_sisnrs - input_sisnrs
        
        # 2. Calculate Efficiency Metrics
        latency_ms = (end_time - start_time) * 1000 / n
        params = sum(p.numel() for p in model.parameters())
        
        # 3. Calculate Retrieval Metrics (CMC, P@1, P@5)
        cmc, p1, p5 = self._calculate_rank_k(embs, lbls)
        
        # 4. Calculate Verification Metrics (ROC, DET, EER)
        # Using all-to-all cosine similarity
        sim_matrix = embs @ embs.T
        # Create label mask: 1 if same class, 0 if diff
        label_matrix = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).int()
        
        # Exclude self-comparisons (diagonal)
        mask = ~torch.eye(n, dtype=torch.bool)
        
        scores = sim_matrix[mask].numpy()
        labels = label_matrix[mask].numpy()
        
        # Downsample if too large (e.g. > 1M pairs) to speed up sklearn metrics
        if len(scores) > 1_000_000:
            idx = np.random.choice(len(scores), 1_000_000, replace=False)
            scores = scores[idx]
            labels = labels[idx]
            
        fpr, tpr, roc_thresh = roc_curve(labels, scores)
        fpr_det, fnr_det, det_thresh = det_curve(labels, scores)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        
        # 5. Robustness Breakdown
        robustness = {}
        # Bucket by closest SNR level (0, 5, 10, 20)
        target_snrs = torch.tensor([0, 5, 10, 20], dtype=torch.float32)
        
        for t_snr in target_snrs:
            # Find samples close to this SNR (+/- 2.5 dB)
            mask_snr = (input_sisnrs >= t_snr - 2.5) & (input_sisnrs < t_snr + 2.5)
            if mask_snr.sum() > 0:
                snr_key = int(t_snr.item())
                # Subset metrics
                sub_embs = embs[mask_snr]
                sub_lbls = lbls[mask_snr]
                _, sub_p1, _ = self._calculate_rank_k(sub_embs, sub_lbls)
                
                avg_delta = delta_sisnrs[mask_snr].mean().item()
                robustness[snr_key] = {"p@1": sub_p1, "delta_sisnr": avg_delta}
        
        # 6. Accuracy (if train_dl provided)
        acc = 0.0
        if train_dl is not None and num_classes is not None:
            centroids = self.compute_centroids(model, train_dl, num_classes)
            acc = self._eval_accuracy(model, dl, centroids)

        return {
            "p@1": p1,
            "p@5": p5,
            "si_snr": sisnr_sum/n,
            "accuracy": acc,
            "cmc": cmc.tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}, 
            "det": {"fpr": fpr_det.tolist(), "fnr": fnr_det.tolist()},
            "eer": eer,
            "latency": latency_ms,
            "params": params,
            "robustness": robustness
        }

    def _calculate_rank_k(self, embs, lbls, k_max=20):
        """Calculate Cumulative Match Characteristic (CMC) curve and P@K"""
        n = len(lbls)
        sim_matrix = embs @ embs.T
        sim_matrix.fill_diagonal_(-1e9)
        
        # Sort by similarity (descending)
        _, indices = sim_matrix.sort(dim=1, descending=True)
        
        # Check matches at each rank
        match_matrix = (lbls[indices] == lbls.unsqueeze(1))
        
        # CMC: probability that correct match is in top k
        cmc = []
        for k in range(1, k_max + 1):
            hits = match_matrix[:, :k].any(dim=1).float().mean().item()
            cmc.append(hits)
            
        p1 = cmc[0]
        p5 = cmc[4] if k_max >= 5 else cmc[-1]
        
        return np.array(cmc), p1, p5

    @torch.no_grad()
    def evaluate_novelty_comprehensive(self, model, known_dl, unknown_noisy, centroids, threshold):
        """Evaluate novelty detection with AUROC/AUPR"""
        model.to(self.device).eval()
        centroids = centroids.to(self.device)
        
        # 1. Get Known Scores (Distance to nearest centroid)
        known_dists = []
        for noisy, _, _ in known_dl:
            _, emb = model(noisy.to(self.device))
            dist_matrix = torch.cdist(emb, centroids, p=2)
            min_dist = dist_matrix.min(dim=1).values
            known_dists.extend(min_dist.cpu().tolist())
            
        # 2. Get Unknown Scores
        unknown_dists = []
        batch_size = 64
        for i in range(0, len(unknown_noisy), batch_size):
            batch = unknown_noisy[i:i+batch_size].to(self.device)
            _, emb = model(batch)
            dist_matrix = torch.cdist(emb, centroids, p=2)
            min_dist = dist_matrix.min(dim=1).values
            unknown_dists.extend(min_dist.cpu().tolist())
            
        y_true = [0] * len(known_dists) + [1] * len(unknown_dists) # 0=Known, 1=Unknown
        y_scores = known_dists + unknown_dists # Higher distance = more likely unknown
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Existing hard threshold metrics
        known_accepted = sum(d < threshold for d in known_dists)
        unknown_rejected = sum(d >= threshold for d in unknown_dists)
        
        known_total = len(known_dists)
        unknown_total = len(unknown_dists)
        
        tar = known_accepted / known_total if known_total > 0 else 0
        trr = unknown_rejected / unknown_total if unknown_total > 0 else 0
        
        return {
            "tar": tar,
            "trr": trr,
            "far": 1 - trr,
            "frr": 1 - tar,
            "auroc": auroc,
            "aupr": aupr,
            "threshold": threshold,
            "known_samples": known_total,
            "unknown_samples": unknown_total
        }
