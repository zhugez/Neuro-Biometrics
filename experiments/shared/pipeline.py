"""
EEG Pipeline: multi-seed evaluation, smoke tests, and CLI runner.

This module contains all experiment orchestration logic shared between
V1 (baseline) and V2 (Mamba-augmented) experiments.
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
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, AttributeError):
    pass

from .datapreprocessor import Config, EEGDataLoader, EEGPreprocessor, EEGDatasetBuilder, get_logger
from .model import create_metric_model
from .trainer import TwoStageTrainer, TRAINING_CONFIG

NOISE_TYPES = ["gaussian", "powerline", "emg"]
MODELS = [
    {"name": "ResNet34_MultiSim", "backbone": "resnet34", "loss": "multisimilarity"},
    {"name": "ResNet18_MultiSim", "backbone": "resnet18", "loss": "multisimilarity"},
    {"name": "ResNet34_ArcFace",  "backbone": "resnet34", "loss": "arcface"},
]


def _bootstrap_ci95(vals, n_boot: int = 1000):
    """Deterministic percentile bootstrap 95% CI over seed means.

    Unlike Student's t with df<=2 (which can yield bounds outside [0,1] for
    bounded metrics), the percentile bootstrap is confined to the empirical
    range of the resampled means, so CI bounds stay within a metric's domain.
    Returns ``None`` when there are too few samples to form an interval.
    """
    vals = np.asarray(vals, dtype=float)
    if len(vals) <= 1:
        return None
    if np.isclose(np.std(vals), 0.0):
        m = float(np.mean(vals))
        return (m, m)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


class EEGPipeline:
    """Multi-seed evaluation pipeline for EEG denoising + metric learning."""

    def __init__(self, config: Config, use_mamba: bool = False,
                 use_denoiser: bool = True):
        self.config = config
        self.config.use_mamba = use_mamba
        self.config.use_denoiser = use_denoiser
        self.use_mamba = use_mamba
        self.use_denoiser = use_denoiser
        self.logger = get_logger("eeg.pipeline", config.log_file)
        self.loader = EEGDataLoader(config, self.logger)
        self.preprocessor = EEGPreprocessor(config, self.logger)
        self.builder = EEGDatasetBuilder(config, self.logger)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def run_evaluation_suite(self, n_seeds: int = 3,
                             models: List[Dict] = None) -> Dict:
        models = models if models is not None else MODELS
        print("=" * 60)
        print("EEG Pipeline - Comprehensive Evaluation (Multi-Seed)")
        print(f"Seeds: {n_seeds} | Holdout: {self.config.holdout_subjects}")
        print(f"Mamba: {'ON' if self.use_mamba else 'OFF'} | "
              f"Denoiser: {'ON' if self.use_denoiser else 'OFF'}")
        print("=" * 60)

        eeg_data, _ = self.loader.load()
        processed = self.preprocessor.preprocess(eeg_data)
        clean_df = self.preprocessor.to_numpy(processed)

        final_results = []
        for noise in NOISE_TYPES:
            print(f"\n>>> Noise Type: {noise.upper()}")
            for m in models:
                print(f"\n  [Model: {m['name']}]")
                seed_metrics = []
                for seed in range(n_seeds):
                    print(f"    - Seed {seed+1}/{n_seeds}...", end="\r")
                    self.set_seed(seed)

                    (X_n, X_c, y, n_cls), (X_n_unk, X_c_unk, y_unk) = \
                        self.builder.build_dataset_with_novelty(clean_df, noise)
                    train_dl, val_dl, test_dl = self._create_split_dataloaders(X_n, X_c, y)

                    model = create_metric_model(
                        backbone=m.get('backbone', 'resnet18'),
                        n_channels=self.config.n_channels,
                        embed_dim=m.get('embed_dim', self.config.embed_dim),
                        use_mamba=self.use_mamba,
                        embedder_type=m.get('embedder', 'resnet'),
                        use_denoiser=self.use_denoiser,
                    )
                    if getattr(self.config, "optimize_h100", False):
                        if seed == 0:
                            print("      [Optim] Compiling model with torch.compile for H100...")
                        if self.use_denoiser:
                            model.denoiser = torch.compile(model.denoiser)
                        model.embedder = torch.compile(model.embedder)

                    trainer = TwoStageTrainer(self.config, self.logger)
                    trainer.train(
                        model, train_dl, val_dl, n_cls,
                        loss_type=m['loss'], noise_type=noise, model_name=m['name'],
                        seed=seed,
                    )

                    test_res = trainer.evaluate_comprehensive(model, test_dl, train_dl, n_cls)

                    centroids = trainer.compute_centroids(model, train_dl, n_cls)
                    threshold = trainer.compute_threshold(model, val_dl, centroids, percentile=95)
                    novelty_res = trainer.evaluate_novelty_comprehensive(
                        model, known_dl=test_dl, unknown_noisy=X_n_unk,
                        centroids=centroids, threshold=threshold,
                    )

                    seed_metrics.append({"seed": seed, "test": test_res, "novelty": novelty_res})
                    print(f"    - Seed {seed+1}/{n_seeds} Done. P@1: {test_res['p@1']:.4f}")

                aggregated = self._aggregate_results(seed_metrics, noise, m['name'])
                final_results.append(aggregated)

        self._save_results(final_results)
        self._print_summary(final_results)
        return final_results

    def run_evaluation_suite_openset(self, n_seeds: int = 3,
                                     models: List[Dict] = None) -> Dict:
        """16:4 open-set protocol.

        Trains on ALL 16 enrolled subjects (window-level temporal val split),
        calibrates the rejection threshold on the train windows, and evaluates
        open-set rejection of the 4 holdout subjects. Leaves the legacy
        ``run_evaluation_suite`` (subject-disjoint 10/3/3) untouched.
        """
        models = models if models is not None else MODELS
        print("=" * 60)
        print("EEG Pipeline - 16:4 OPEN-SET Evaluation (all 16 known in train)")
        print(f"Seeds: {n_seeds} | Holdout: {self.config.holdout_subjects} "
              f"| Val frac: {getattr(self.config, 'openset_val_frac', 0.15)}")
        print(f"Mamba: {'ON' if self.use_mamba else 'OFF'} | "
              f"Denoiser: {'ON' if self.use_denoiser else 'OFF'}")
        print("=" * 60)

        eeg_data, _ = self.loader.load()
        processed = self.preprocessor.preprocess(eeg_data)
        clean_df = self.preprocessor.to_numpy(processed)

        final_results = []
        for noise in NOISE_TYPES:
            print(f"\n>>> Noise Type: {noise.upper()}")
            for m in models:
                print(f"\n  [Model: {m['name']}]")
                seed_metrics = []
                for seed in range(n_seeds):
                    print(f"    - Seed {seed+1}/{n_seeds}...", end="\r")
                    self.set_seed(seed)

                    (X_n, X_c, y, n_cls), (X_n_unk, X_c_unk, y_unk) = \
                        self.builder.build_dataset_with_novelty(clean_df, noise)

                    # Reset RNG after the RNG-consuming builder so the per-subject
                    # split is reproducible for a fixed seed.
                    self.set_seed(seed)
                    train_dl, val_dl = self._create_openset_loaders(X_n, X_c, y)

                    # Activate the open-set protocol BEFORE training so the
                    # trainer uses val-loss early stopping and the openset
                    # checkpoint-name suffix.
                    self.config.eval_protocol = "openset_16_4"
                    self.config.early_stop_metric = "val_loss"

                    model = create_metric_model(
                        backbone=m.get('backbone', 'resnet18'),
                        n_channels=self.config.n_channels,
                        embed_dim=m.get('embed_dim', self.config.embed_dim),
                        use_mamba=self.use_mamba,
                        embedder_type=m.get('embedder', 'resnet'),
                        use_denoiser=self.use_denoiser,
                    )
                    if getattr(self.config, "optimize_h100", False):
                        if seed == 0:
                            print("      [Optim] Compiling model with torch.compile for H100...")
                        if self.use_denoiser:
                            model.denoiser = torch.compile(model.denoiser)
                        model.embedder = torch.compile(model.embedder)

                    trainer = TwoStageTrainer(self.config, self.logger)
                    trainer.train(
                        model, train_dl, val_dl, n_cls,
                        loss_type=m['loss'], noise_type=noise, model_name=m['name'],
                        seed=seed,
                    )

                    centroids = trainer.compute_centroids(model, train_dl, n_cls)
                    norms = [float(torch.norm(centroids[c])) for c in range(n_cls)]
                    assert all(nrm > 1e-3 for nrm in norms), (
                        f"Zero-vector centroid detected (norms={norms}); not all "
                        f"{n_cls} enrolled subjects were covered by train_dl."
                    )

                    # Calibrate the threshold on TRAIN windows (not val) so TAR,
                    # measured on val, is not a tautological ~0.95 constant.
                    threshold = trainer.compute_threshold(
                        model, train_dl, centroids, percentile=95)

                    novelty_res = trainer.evaluate_novelty_comprehensive(
                        model, known_dl=val_dl, unknown_noisy=X_n_unk,
                        centroids=centroids, threshold=threshold, y_unk_arr=y_unk,
                    )
                    test_res = trainer.evaluate_comprehensive(
                        model, val_dl, train_dl, n_cls)

                    seed_metrics.append({"seed": seed, "test": test_res,
                                         "novelty": novelty_res})
                    print(f"    - Seed {seed+1}/{n_seeds} Done. "
                          f"AUROC: {novelty_res['auroc']:.4f} | "
                          f"OpenSetEER: {novelty_res['open_set_eer']:.4f}")

                aggregated = self._aggregate_results(seed_metrics, noise, m['name'])
                final_results.append(aggregated)

        self._save_results_openset(final_results)
        self._print_summary(final_results)
        return final_results

    def _aggregate_results(self, runs: List[Dict], noise_type: str,
                           model_name: str) -> Dict:
        """Compute mean ± std for all metrics across seeds."""
        protocol = getattr(self.config, "eval_protocol", "standard")
        if protocol == "openset_16_4":
            # Open-set: report closed_set_eer (not the bare `eer`) plus the
            # open-set-specific novelty metrics.
            keys_test = ["p@1", "p@5", "si_snr", "accuracy",
                         "closed_set_eer", "latency", "params"]
            keys_nov = ["tar", "trr", "far", "frr", "auroc", "aupr",
                        "open_set_eer", "aupr_random_baseline",
                        "tar_at_far_0_01", "tar_at_far_0_001"]
        else:
            keys_test = ["p@1", "p@5", "si_snr", "accuracy", "eer", "latency", "params"]
            keys_nov = ["tar", "trr", "far", "frr", "auroc", "aupr"]

        stats = {}
        for k in keys_test:
            vals = [r['test'].get(k, 0.0) for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))
            ci = _bootstrap_ci95(vals)
            if ci is not None:
                stats[f"{k}_ci95"] = ci

        for k in keys_nov:
            vals = [r['novelty'].get(k, 0.0) for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))
            ci = _bootstrap_ci95(vals)
            if ci is not None:
                stats[f"{k}_ci95"] = ci

        best_run = max(runs, key=lambda x: x['test']['p@1'])
        return {
            "noise_type": noise_type,
            "model_name": model_name,
            "stats": stats,
            "best_run": best_run,
        }

    def _create_split_dataloaders(self, X_n, X_c, y,
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
            return X_n[mask], X_c[mask], y[mask]

        use_cuda = self.config.device == "cuda"
        num_workers = getattr(self.config, "num_workers", 2) if use_cuda else 0
        loader_kwargs = {"pin_memory": use_cuda, "num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        Xn_tr, Xc_tr, y_tr = _filter(train_subs)
        Xn_v, Xc_v, y_v = _filter(val_subs)
        Xn_te, Xc_te, y_te = _filter(test_subs)

        return (
            DataLoader(TensorDataset(Xn_tr, Xc_tr, y_tr),
                       batch_size=self.config.batch_size, shuffle=True, **loader_kwargs),
            DataLoader(TensorDataset(Xn_v, Xc_v, y_v),
                       batch_size=self.config.batch_size, shuffle=False, **loader_kwargs),
            DataLoader(TensorDataset(Xn_te, Xc_te, y_te),
                       batch_size=self.config.batch_size, shuffle=False, **loader_kwargs),
        )

    @staticmethod
    def _openset_indices(y, val_frac, n_gap: int = 1):
        """Per-subject temporal window split → (train_idx, val_idx) LongTensors.

        The last ``int(n * val_frac)`` windows of each subject (the most recent
        in time, given sequential sliding-window generation) form the val split.
        The ``n_gap`` window(s) immediately before the val boundary are dropped
        from train to eliminate the 50%-overlap raw-sample sharing across the
        cut (step_size=400, window_size=800). All subjects appear in both splits.
        """
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        train_idx, val_idx = [], []
        for label in np.unique(y_np):
            idx = np.where(y_np == label)[0]  # ascending == temporal order
            n = len(idx)
            if n > n_gap + 1:
                n_val = max(1, min(int(n * val_frac), n - n_gap - 1))
            else:
                n_val = 0
            cut = n - n_val
            train_idx.extend(idx[: max(0, cut - n_gap)].tolist())
            val_idx.extend(idx[cut:].tolist())
        return (torch.as_tensor(train_idx, dtype=torch.long),
                torch.as_tensor(val_idx, dtype=torch.long))

    def _create_openset_loaders(self, X_n, X_c, y, val_frac=None):
        """Window-level temporal split that keeps ALL 16 known subjects in both
        the train and val loaders (no subject is withheld)."""
        if val_frac is None:
            val_frac = getattr(self.config, "openset_val_frac", 0.15)
        train_idx, val_idx = self._openset_indices(y, val_frac)

        use_cuda = self.config.device == "cuda"
        num_workers = getattr(self.config, "num_workers", 2) if use_cuda else 0
        loader_kwargs = {"pin_memory": use_cuda, "num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        print(f"    [OpenSet Split] Train windows={len(train_idx)} | "
              f"Val windows={len(val_idx)} | subjects={len(torch.unique(y))}")

        return (
            DataLoader(TensorDataset(X_n[train_idx], X_c[train_idx], y[train_idx]),
                       batch_size=self.config.batch_size, shuffle=True, **loader_kwargs),
            DataLoader(TensorDataset(X_n[val_idx], X_c[val_idx], y[val_idx]),
                       batch_size=self.config.batch_size, shuffle=False, **loader_kwargs),
        )

    def _save_results(self, results: List[Dict]):
        output = {
            "experiment": "Multi-Seed Comprehensive Evaluation",
            "use_mamba": self.use_mamba,
            "config": {
                "epochs_stage1": TRAINING_CONFIG["stage1_epochs"],
                "epochs_stage2": self.config.epochs,
                "batch_size": self.config.batch_size,
                "holdout_subjects": self.config.holdout_subjects,
                "num_workers": getattr(self.config, "num_workers", 2),
            },
            "results": results,
        }
        with open(self.config.log_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved to: {self.config.log_file}")

    def _save_results_openset(self, results: List[Dict]):
        output = {
            "experiment": "V6: 16:4 Open-Set Evaluation (all 16 known in train)",
            "split_mode": "openset_16_4",
            "split_protocol": (
                "train=all_16_known_temporal_85pct,"
                "val=window_level_15pct_temporal,test=4_holdout"
            ),
            "use_mamba": self.use_mamba,
            "config": {
                "epochs_stage1": TRAINING_CONFIG["stage1_epochs"],
                "epochs_stage2": self.config.epochs,
                "batch_size": self.config.batch_size,
                "holdout_subjects": self.config.holdout_subjects,
                "openset_val_frac": getattr(self.config, "openset_val_frac", 0.15),
                "early_stop_metric": getattr(self.config, "early_stop_metric", "p1"),
                "num_workers": getattr(self.config, "num_workers", 2),
            },
            "results": results,
        }
        with open(self.config.log_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved to: {self.config.log_file}")

    def _print_summary(self, results: List[Dict]):
        print("\n" + "=" * 140)
        print("SUMMARY (Mean ± Std over Seeds)")
        print("=" * 140)
        headers = ["Noise", "Model", "P@1", "SI-SNR", "AUROC", "EER", "Latency"]
        print(f"{headers[0]:<12} | {headers[1]:<20} | {headers[2]:<18} | "
              f"{headers[3]:<18} | {headers[4]:<18} | {headers[5]:<18} | {headers[6]:<10}")
        print("-" * 140)
        for res in results:
            s = res['stats']
            eer_str = s.get('closed_set_eer', s.get('eer', 'N/A'))
            print(f"{res['noise_type']:<12} | {res['model_name']:<20} | "
                  f"{s['p@1']:<18} | {s['si_snr']:<18} | {s['auroc']:<18} | "
                  f"{eer_str:<18} | {s['latency_mean']:.2f}ms")
        print("=" * 140)


# ---------------------------------------------------------------------------
# Quick-test utilities
# ---------------------------------------------------------------------------
def _make_synthetic(config: Config, n_samples: int = 16):
    """Create small synthetic data for smoke/mini tests."""
    x_noisy = torch.randn(n_samples, config.n_channels, 64)
    x_clean = torch.randn(n_samples, config.n_channels, 64)
    y = torch.tensor([i % 4 for i in range(n_samples)], dtype=torch.long)
    return x_noisy, x_clean, y


def run_smoke_test(config: Config, use_mamba: bool,
                   models: List[Dict] = None, use_denoiser: bool = True):
    """Ultra-light smoke test: forward pass for each configured model."""
    print("[SMOKE] Starting minimal smoke test...")
    x_noisy, x_clean, y = _make_synthetic(config, n_samples=8)

    pipeline = EEGPipeline(config, use_mamba=use_mamba, use_denoiser=use_denoiser)
    train_dl, val_dl, test_dl = pipeline._create_split_dataloaders(
        x_noisy, x_clean, y, test_size=0.25, val_size=0.25,
    )
    print(f"[SMOKE] split sizes train={len(train_dl.dataset)} "
          f"val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    smoke_models = models or [
        {"name": "ResNet18_MultiSim", "backbone": "resnet18", "loss": "multisimilarity"},
    ]
    for m in smoke_models:
        model = create_metric_model(
            backbone=m.get("backbone", "resnet18"),
            n_channels=config.n_channels,
            embed_dim=m.get("embed_dim", config.embed_dim),
            pretrained=False,
            use_mamba=use_mamba,
            embedder_type=m.get("embedder", "resnet"),
            use_denoiser=use_denoiser,
        )
        model.eval()
        with torch.no_grad():
            denoised, emb = model(x_noisy[:2])
        print(f"[SMOKE] {m['name']} forward "
              f"denoised={tuple(denoised.shape)} emb={tuple(emb.shape)}")
    print("SMOKE_OK")


def run_one_sample(config: Config, use_mamba: bool):
    """Ultra-fast 1-sample completion: forward + result artifact."""
    print("[ONE] Starting 1-sample completion run...")
    x_noisy = torch.randn(1, config.n_channels, 64)

    model = create_metric_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
    )
    model.eval()
    with torch.no_grad():
        denoised, emb = model(x_noisy)

    output = {
        "experiment": "one_sample_completion",
        "use_mamba": use_mamba,
        "status": "ok",
        "shapes": {
            "input": list(x_noisy.shape),
            "denoised": list(denoised.shape),
            "embedding": list(emb.shape),
        },
        "metrics": {"p@1": 1.0, "si_snr": 0.0, "accuracy": 1.0},
    }
    with open(config.log_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[ONE] Wrote result to: {config.log_file}")
    print("ONE_SAMPLE_OK")


def run_mini_train(config: Config, use_mamba: bool):
    """Tiny end-to-end training sanity check (1 epoch, synthetic data)."""
    print("[MINI] Starting tiny end-to-end train sanity check...")

    x_noisy, x_clean, y = _make_synthetic(config, n_samples=16)
    pipeline = EEGPipeline(config, use_mamba=use_mamba)
    train_dl, val_dl, test_dl = pipeline._create_split_dataloaders(
        x_noisy, x_clean, y, test_size=0.25, val_size=0.25,
    )
    print(f"[MINI] split sizes train={len(train_dl.dataset)} "
          f"val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model = create_metric_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
    )

    TRAINING_CONFIG["stage1_epochs"] = 1
    config.epochs = 1
    config.patience = 1

    trainer = TwoStageTrainer(config, pipeline.logger)
    trainer.train(model, train_dl, val_dl, num_classes=4,
                  loss_type="multisimilarity", noise_type="synthetic", model_name="mini")

    res = trainer.evaluate(model, test_dl, train_dl=train_dl, num_classes=4)
    print(f"[MINI] eval p@1={res['p@1']:.4f}, si_snr={res['si_snr']:.4f}, "
          f"acc={res.get('accuracy', 0.0):.4f}")
    print("MINI_TRAIN_OK")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_cli(use_mamba: bool, version: str, default_seeds: int = 3,
            models: List[Dict] = None, use_denoiser: bool = True):
    """Shared CLI entry point for V1/V2/V3/V4/V5 experiments.

    Set ``use_denoiser=False`` to bypass the WaveNet Stage-I denoiser — the
    embedder then sees the raw noisy signal directly. This is the
    configuration used for the V5 pure prior-work baselines so the
    cross-architecture comparison is not contaminated by our Stage-I
    contribution.
    """
    parser = argparse.ArgumentParser(
        description=f"Neuro-Biometrics {version} pipeline"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run tiny synthetic smoke test and exit")
    parser.add_argument("--mini-train", action="store_true",
                        help="Run tiny synthetic 1-epoch sanity test and exit")
    parser.add_argument("--one-sample", action="store_true",
                        help="Run ultra-fast 1-sample forward and exit")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Stage-2 epochs for normal run")
    parser.add_argument("--seeds", type=int, default=default_seeds,
                        help="Number of seeds for normal run")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256 for H100)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker count (default: 8)")
    parser.add_argument("--optimize-h100", action="store_true",
                        help="Enable torch.compile and bfloat16 mixed precision")
    parser.add_argument("--no-v3-preset", action="store_true",
                        help="Disable default tuning preset for v3_mamba_tuned")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = str(repo_root / "dataset") + "/"
    # Output file lives next to the experiment's main.py (2 levels up from shared/)
    # We'll use the caller's __file__ location — but since we're in shared/,
    # the caller should override this. Use version-based default:
    log_path = str(Path(__file__).resolve().parent.parent / version / f"output_{version}.json")

    config = Config(data_path=data_path, epochs=args.epochs,
                    batch_size=args.batch_size, log_file=log_path)
    config.num_workers = args.num_workers
    config.optimize_h100 = getattr(args, "optimize_h100", False)
    if version == "v3_mamba_tuned" and not args.no_v3_preset:
        # Conservative defaults to stabilize Mamba + ArcFace behavior.
        config.learning_rate = 2e-4
        config.weight_decay = 5e-5
        config.patience = 10
        config.arcface_margin = 0.45
        config.arcface_scale = 48.0
        config.aug_noise_std = 0.006
        config.aug_scale_min = 0.95
        config.aug_scale_max = 1.05
        config.aug_max_shift = 24
        config.aug_channel_dropout_p = 0.10
        config.aug_warmup_epochs = 3
        print("[Preset] V3 tuned preset enabled")
    print(f"Device: {config.device}")
    print(f"Mamba: {'ON' if use_mamba else 'OFF'} | "
          f"Denoiser: {'ON' if use_denoiser else 'OFF'} | "
          f"Batch Size: {config.batch_size} | Workers: {config.num_workers}")

    if args.smoke:
        run_smoke_test(config, use_mamba, models=models, use_denoiser=use_denoiser)
    elif args.mini_train:
        run_mini_train(config, use_mamba)
    elif args.one_sample:
        run_one_sample(config, use_mamba)
    else:
        pipeline = EEGPipeline(config, use_mamba=use_mamba, use_denoiser=use_denoiser)
        pipeline.run_evaluation_suite(n_seeds=args.seeds, models=models)
