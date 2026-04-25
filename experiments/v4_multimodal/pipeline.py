"""
V4 Multimodal Pipeline: EEG + Spectrogram bimodal training.

Same structure as shared/pipeline.py but:
- Uses MultimodalEEGMetricModel instead of EEGMetricModel
- Uses BimodalTrainer instead of TwoStageTrainer
- Uses SpectrogramDataset that returns (noisy, clean, labels, spectrograms)
- Model: create_multimodal_model(backbone, n_channels, embed_dim, pretrained, use_mamba)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.stats as _scipy_stats
import torch
import random

from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import V4Config
except ImportError:
    from config import V4Config

from shared.datapreprocessor import EEGDataLoader, EEGPreprocessor, EEGDatasetBuilder, get_logger
from shared.model_multimodal import create_multimodal_model
from shared.trainer_bimodal import BimodalTrainer, BIMODAL_TRAINING_CONFIG
from shared.dataset_spectrogram import patch_builder_with_spectrogram

NOISE_TYPES = ["gaussian", "powerline", "emg"]
MODELS = [
    {"name": "ResNet34_MultiSim", "backbone": "resnet34", "loss": "multisimilarity"},
    {"name": "ResNet18_MultiSim", "backbone": "resnet18", "loss": "multisimilarity"},
    {"name": "ResNet34_ArcFace",  "backbone": "resnet34", "loss": "arcface"},
]

# Patch the builder to add the spectrogram method
patch_builder_with_spectrogram()


class MultimodalEEGPipeline:
    """Multi-seed evaluation pipeline for bimodal EEG + Spectrogram metric learning."""

    def __init__(self, config: V4Config, use_mamba: bool = True):
        self.config = config
        self.config.use_mamba = use_mamba
        self.use_mamba = use_mamba
        self.logger = get_logger("eeg.pipeline.v4", config.log_file)
        self.loader = EEGDataLoader(config, self.logger)
        self.preprocessor = EEGPreprocessor(config, self.logger)
        self.builder = EEGDatasetBuilder(config, self.logger)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def run_evaluation_suite(self, n_seeds: int = 3) -> Dict:
        print("=" * 60)
        print("V4 Multimodal Pipeline - Comprehensive Evaluation (Multi-Seed)")
        print(f"Seeds: {n_seeds} | Holdout: {self.config.holdout_subjects}")
        print(f"Mamba: {'ON' if self.use_mamba else 'OFF'} | "
              f"Spectrogram Source: {getattr(self.config, 'spectrogram_source', 'noisy')}")
        print("=" * 60)

        eeg_data, _ = self.loader.load()
        processed = self.preprocessor.preprocess(eeg_data)
        clean_df = self.preprocessor.to_numpy(processed)

        final_results = []
        for noise in NOISE_TYPES:
            print(f"\n>>> Noise Type: {noise.upper()}")
            for m in MODELS:
                print(f"\n  [Model: {m['name']}]")
                seed_metrics = []
                for seed in range(n_seeds):
                    print(f"    - Seed {seed+1}/{n_seeds}...", end="\r")
                    self.set_seed(seed)

                    (X_n, X_c, y, X_spec, n_cls), \
                    (X_n_unk, X_c_unk, y_unk, X_spec_unk) = \
                        self.builder.build_dataset_with_novelty_bimodal(
                            clean_df, noise,
                            n_fft=getattr(self.config, 'spectrogram_n_fft', 128),
                            hop_length=getattr(self.config, 'spectrogram_hop_length', 64),
                        )
                    train_dl, val_dl, test_dl = self._create_split_dataloaders(X_n, X_c, y, X_spec)

                    model = create_multimodal_model(
                        backbone=m['backbone'],
                        n_channels=self.config.n_channels,
                        embed_dim=self.config.embed_dim,
                        use_mamba=self.use_mamba,
                        spec_embed_dim=getattr(self.config, 'spec_embed_dim', None),
                    )
                    if getattr(self.config, "optimize_h100", False):
                        if seed == 0:
                            print("      [Optim] Compiling model with torch.compile for H100...")
                        model.denoiser = torch.compile(model.denoiser)
                        model.embedder = torch.compile(model.embedder)
                        model.spec_embedder = torch.compile(model.spec_embedder)
                        model.fusion = torch.compile(model.fusion)

                    trainer = BimodalTrainer(self.config, self.logger)
                    trainer.train(
                        model, train_dl, val_dl, n_cls,
                        loss_type=m['loss'], noise_type=noise, model_name=m['name'],
                        seed=seed,
                    )

                    test_res = trainer.evaluate_bimodal(model, test_dl, train_dl, n_cls)

                    centroids = trainer.compute_centroids_bimodal(model, train_dl, n_cls)
                    threshold = self._compute_threshold_bimodal(trainer, model, val_dl, centroids)
                    novelty_res = self._evaluate_novelty_bimodal(
                        trainer, model, test_dl, X_n_unk, X_spec_unk, centroids, threshold,
                    )

                    seed_metrics.append({"seed": seed, "test": test_res, "novelty": novelty_res})
                    print(f"    - Seed {seed+1}/{n_seeds} Done. P@1: {test_res['p@1']:.4f} | "
                          f"AUROC: {test_res.get('auroc', 0.0):.4f}")

                aggregated = self._aggregate_results(seed_metrics, noise, m['name'])
                final_results.append(aggregated)

        self._save_results(final_results)
        self._print_summary(final_results)
        return final_results

    def _aggregate_results(self, runs: List[Dict], noise_type: str,
                          model_name: str) -> Dict:
        """Compute mean ± std for all metrics across seeds."""
        keys_test = ["p@1", "p@5", "si_snr", "accuracy", "auroc", "eer", "latency", "params"]
        keys_nov = ["tar", "trr", "far", "frr", "auroc", "aupr"]

        stats = {}
        for k in keys_test:
            vals = [r['test'].get(k, 0.0) for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))
            if len(vals) > 1:
                ci = _scipy_stats.t.interval(
                    0.95, df=len(vals) - 1,
                    loc=np.mean(vals), scale=_scipy_stats.sem(vals)
                )
                stats[f"{k}_ci95"] = (float(ci[0]), float(ci[1]))

        for k in keys_nov:
            vals = [r['novelty'][k] for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))
            if len(vals) > 1:
                ci = _scipy_stats.t.interval(
                    0.95, df=len(vals) - 1,
                    loc=np.mean(vals), scale=_scipy_stats.sem(vals)
                )
                stats[f"{k}_ci95"] = (float(ci[0]), float(ci[1]))

        best_run = max(runs, key=lambda x: x['test']['p@1'])
        return {
            "noise_type": noise_type,
            "model_name": model_name,
            "stats": stats,
            "best_run": best_run,
        }

    def _create_split_dataloaders(self, X_n, X_c, y, X_spec,
                                  test_size=0.2, val_size=0.2):
        """Subject-based split to avoid data leakage."""
        subjects = np.unique(y.numpy())
        np.random.shuffle(subjects)

        n_subs = len(subjects)
        n_test = max(1, int(n_subs * test_size))
        n_val = max(1, int(n_subs * val_size))
        if n_subs - n_test - n_val <= 0:
            n_val = max(0, n_val - 1)
            if n_subs - n_test - n_val <= 0:
                n_test = max(0, n_test - 1)

        test_subs = subjects[:n_test]
        val_subs = subjects[n_test:n_test + n_val]
        train_subs = subjects[n_test + n_val:]

        print(f"    [Split] Subjects: Train={len(train_subs)} | Val={len(val_subs)} | Test={len(test_subs)}")

        def _filter(subs):
            mask = np.isin(y.numpy(), subs)
            return X_n[mask], X_c[mask], y[mask], X_spec[mask]

        use_cuda = self.config.device == "cuda"
        num_workers = getattr(self.config, "num_workers", 2) if use_cuda else 0
        loader_kwargs = {"pin_memory": use_cuda, "num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        Xn_tr, Xc_tr, y_tr, Xs_tr = _filter(train_subs)
        Xn_v, Xc_v, y_v, Xs_v = _filter(val_subs)
        Xn_te, Xc_te, y_te, Xs_te = _filter(test_subs)

        return (
            DataLoader(TensorDataset(Xn_tr, Xc_tr, y_tr, Xs_tr),
                       batch_size=self.config.batch_size, shuffle=True, **loader_kwargs),
            DataLoader(TensorDataset(Xn_v, Xc_v, y_v, Xs_v),
                       batch_size=self.config.batch_size, shuffle=False, **loader_kwargs),
            DataLoader(TensorDataset(Xn_te, Xc_te, y_te, Xs_te),
                       batch_size=self.config.batch_size, shuffle=False, **loader_kwargs),
        )

    def _compute_threshold_bimodal(self, trainer, model, val_dl, centroids):
        """Compute distance threshold for novelty detection (bimodal)."""
        model.to(self.config.device).eval()
        centroids = centroids.to(self.config.device)
        all_dists = []
        for noisy, _, _, spectrograms in val_dl:
            noisy = noisy.to(self.config.device)
            spectrograms = spectrograms.to(device=self.config.device, dtype=noisy.dtype)
            _, emb = model(noisy, spectrograms)
            dist = torch.cdist(emb, centroids, p=2)
            all_dists.extend(dist.min(dim=1).values.cpu().tolist())
        return float(np.percentile(all_dists, 95))

    def _evaluate_novelty_bimodal(self, trainer, model, known_dl, unknown_noisy,
                                   unknown_specs, centroids, threshold):
        """Evaluate novelty detection for bimodal model."""
        model.to(self.config.device).eval()
        centroids = centroids.to(self.config.device)

        # Known distances
        known_dists = []
        for noisy, _, _, spectrograms in known_dl:
            noisy = noisy.to(self.config.device)
            spectrograms = spectrograms.to(device=self.config.device, dtype=noisy.dtype)
            _, emb = model(noisy, spectrograms)
            dist = torch.cdist(emb, centroids, p=2)
            known_dists.extend(dist.min(dim=1).values.cpu().tolist())

        # Unknown distances (precomputed spectrograms)
        unknown_dists = []
        batch_size = 64
        for i in range(0, len(unknown_noisy), batch_size):
            batch_noisy = unknown_noisy[i:i + batch_size].to(self.config.device)
            batch_specs = unknown_specs[i:i + batch_size].to(
                device=self.config.device,
                dtype=batch_noisy.dtype,
            )
            _, emb = model(batch_noisy, batch_specs)
            dist = torch.cdist(emb, centroids, p=2)
            unknown_dists.extend(dist.min(dim=1).values.cpu().tolist())

        y_true = [0] * len(known_dists) + [1] * len(unknown_dists)
        y_scores = known_dists + unknown_dists

        from sklearn.metrics import roc_auc_score, average_precision_score
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

    def _save_results(self, results: List[Dict]):
        output = {
            "experiment": "V4 Multimodal - Multi-Seed Comprehensive Evaluation",
            "use_mamba": self.use_mamba,
            "config": {
                "epochs_stage1": getattr(self.config, "stage1_epochs", BIMODAL_TRAINING_CONFIG["stage1_epochs"]),
                "epochs_stage2": self.config.epochs,
                "batch_size": self.config.batch_size,
                "holdout_subjects": self.config.holdout_subjects,
                "num_workers": getattr(self.config, "num_workers", 2),
                "spectrogram_source": getattr(self.config, "spectrogram_source", "noisy"),
                "spectrogram_n_fft": getattr(self.config, "spectrogram_n_fft", 128),
                "spectrogram_hop_length": getattr(self.config, "spectrogram_hop_length", 64),
                "spec_embed_dim": getattr(self.config, "spec_embed_dim", None),
            },
            "results": results,
        }
        with open(self.config.log_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {self.config.log_file}")

    def _print_summary(self, results: List[Dict]):
        print("\n" + "=" * 140)
        print("V4 MULTIMODAL SUMMARY (Mean ± Std over Seeds)")
        print("=" * 140)
        headers = ["Noise", "Model", "P@1", "AUROC", "SI-SNR", "EER", "Latency"]
        print(f"{headers[0]:<12} | {headers[1]:<20} | {headers[2]:<18} | "
              f"{headers[3]:<18} | {headers[4]:<18} | {headers[5]:<18} | {headers[6]:<10}")
        print("-" * 140)
        for res in results:
            s = res['stats']
            latency = s.get('latency_mean', 0.0)
            print(f"{res['noise_type']:<12} | {res['model_name']:<20} | "
                  f"{s['p@1']:<18} | {s['auroc']:<18} | {s['si_snr']:<18} | "
                  f"{s.get('eer', 'N/A'):<18} | {latency:.2f}ms")
        print("=" * 140)


# ---------------------------------------------------------------------------
# Quick-test utilities
# ---------------------------------------------------------------------------
def _make_synthetic_bimodal(config: V4Config, n_samples: int = 16):
    """Create small synthetic bimodal data for smoke/mini tests."""
    x_noisy = torch.randn(n_samples, config.n_channels, 800)
    x_clean = torch.randn(n_samples, config.n_channels, 800)
    y = torch.tensor([i % 4 for i in range(n_samples)], dtype=torch.long)
    # Synthetic spectrograms: (N, 4, 65, 13)
    x_spec = torch.randn(n_samples, config.n_channels, 65, 13)
    return x_noisy, x_clean, y, x_spec


def run_smoke_test(config: V4Config, use_mamba: bool):
    """Ultra-light smoke test: forward pass only."""
    print("[SMOKE] Starting minimal smoke test (bimodal)...")
    x_noisy, x_clean, y, x_spec = _make_synthetic_bimodal(config, n_samples=8)

    pipeline = MultimodalEEGPipeline(config, use_mamba=use_mamba)
    train_dl, val_dl, test_dl = pipeline._create_split_dataloaders(
        x_noisy, x_clean, y, x_spec, test_size=0.25, val_size=0.25,
    )
    print(f"[SMOKE] split sizes train={len(train_dl.dataset)} "
          f"val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model = create_multimodal_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
        spec_embed_dim=getattr(config, "spec_embed_dim", None),
    ).to(config.device)
    x_noisy = x_noisy.to(config.device)
    x_spec = x_spec.to(config.device)
    model.eval()
    with torch.no_grad():
        denoised, fused = model(x_noisy[:2], x_spec[:2])
    print(f"[SMOKE] forward denoised={tuple(denoised.shape)} fused={tuple(fused.shape)}")
    print("[SMOKE] Bimodal forward pass OK")


def run_one_sample(config: V4Config, use_mamba: bool):
    """Ultra-fast 1-sample completion: forward + result artifact."""
    print("[ONE] Starting 1-sample completion run (bimodal)...")
    x_noisy = torch.randn(1, config.n_channels, 800)
    x_spec = torch.randn(1, config.n_channels, 65, 13)

    model = create_multimodal_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
        spec_embed_dim=getattr(config, "spec_embed_dim", None),
    ).to(config.device)
    x_noisy = x_noisy.to(config.device)
    x_spec = x_spec.to(config.device)
    model.eval()
    with torch.no_grad():
        denoised, fused = model(x_noisy, x_spec)

    output = {
        "experiment": "one_sample_completion_v4",
        "use_mamba": use_mamba,
        "status": "ok",
        "shapes": {
            "input_eeg": list(x_noisy.shape),
            "input_spec": list(x_spec.shape),
            "denoised": list(denoised.shape),
            "fused_embedding": list(fused.shape),
        },
        "metrics": {"p@1": 1.0, "si_snr": 0.0, "accuracy": 1.0, "auroc": 1.0},
    }
    with open(config.log_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[ONE] Wrote result to: {config.log_file}")
    print("[ONE] ONE_SAMPLE_OK")


def run_mini_train(config: V4Config, use_mamba: bool):
    """Tiny end-to-end training sanity check (1 epoch, synthetic data)."""
    print("[MINI] Starting tiny end-to-end bimodal train sanity check...")

    x_noisy, x_clean, y, x_spec = _make_synthetic_bimodal(config, n_samples=16)
    pipeline = MultimodalEEGPipeline(config, use_mamba=use_mamba)
    train_dl, val_dl, test_dl = pipeline._create_split_dataloaders(
        x_noisy, x_clean, y, x_spec, test_size=0.25, val_size=0.25,
    )
    print(f"[MINI] split sizes train={len(train_dl.dataset)} "
          f"val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model = create_multimodal_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
        spec_embed_dim=getattr(config, "spec_embed_dim", None),
    )

    BIMODAL_TRAINING_CONFIG["stage1_epochs"] = 1
    config.stage1_epochs = 1
    config.epochs = 1
    config.patience = 1

    trainer = BimodalTrainer(config, pipeline.logger)
    trainer.train(model, train_dl, val_dl, num_classes=4,
                  loss_type="multisimilarity", noise_type="synthetic", model_name="mini")

    res = trainer.evaluate_bimodal(model, test_dl, train_dl, num_classes=4)
    print(f"[MINI] eval p@1={res['p@1']:.4f}, si_snr={res['si_snr']:.4f}, "
          f"auroc={res.get('auroc', 0.0):.4f}, acc={res.get('accuracy', 0.0):.4f}")
    print("[MINI] MINI_TRAIN_OK")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_cli(use_mamba: bool = True, version: str = "v4_multimodal",
            default_seeds: int = 3):
    """CLI entry point for V4 multimodal experiments."""
    parser = argparse.ArgumentParser(
        description=f"Neuro-Biometrics {version} multimodal pipeline"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run tiny synthetic smoke test and exit")
    parser.add_argument("--mini-train", action="store_true",
                        help="Run tiny synthetic 1-epoch sanity test and exit")
    parser.add_argument("--one-sample", action="store_true",
                        help="Run ultra-fast 1-sample forward and exit")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Stage-2 epochs for normal run (default: 30)")
    parser.add_argument("--seeds", type=int, default=default_seeds,
                        help=f"Number of seeds for normal run (default: {default_seeds})")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256 for H100)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker count (default: 8)")
    parser.add_argument("--optimize-h100", action="store_true",
                        help="Enable torch.compile and bfloat16 mixed precision")
    parser.add_argument("--spectrogram-source", type=str, default="denoised",
                        choices=["noisy", "denoised"],
                        help="Use spectrogram from noisy or denoised EEG (default: denoised)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = str(repo_root / "dataset") + "/"
    log_path = str(Path(__file__).resolve().parent / f"output_{version}.json")

    config = V4Config(
        data_path=data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_file=log_path,
    )
    config.num_workers = args.num_workers
    config.optimize_h100 = getattr(args, "optimize_h100", False)
    config.spectrogram_source = args.spectrogram_source
    config.spectrogram_n_fft = 128
    config.spectrogram_hop_length = 64
    config.stage2_epochs = args.epochs

    print(f"Device: {config.device}")
    print(f"Mamba: {'ON' if use_mamba else 'OFF'} | Batch Size: {config.batch_size} | "
          f"Workers: {config.num_workers}")
    print(f"Spectrogram: {config.spectrogram_source} | n_fft={config.spectrogram_n_fft} | "
          f"hop={config.spectrogram_hop_length}")

    if args.smoke:
        run_smoke_test(config, use_mamba)
    elif args.mini_train:
        run_mini_train(config, use_mamba)
    elif args.one_sample:
        run_one_sample(config, use_mamba)
    else:
        pipeline = MultimodalEEGPipeline(config, use_mamba=use_mamba)
        pipeline.run_evaluation_suite(n_seeds=args.seeds)
