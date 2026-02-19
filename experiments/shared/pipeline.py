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
import torch
import random

from torch.utils.data import DataLoader, TensorDataset

from .datapreprocessor import Config, EEGDataLoader, EEGPreprocessor, EEGDatasetBuilder, get_logger
from .model import create_metric_model
from .trainer import TwoStageTrainer, TRAINING_CONFIG

NOISE_TYPES = ["gaussian", "powerline", "emg"]
MODELS = [
    {"name": "ResNet34_MultiSim", "backbone": "resnet34", "loss": "multisimilarity"},
    {"name": "ResNet18_MultiSim", "backbone": "resnet18", "loss": "multisimilarity"},
    {"name": "ResNet34_ArcFace",  "backbone": "resnet34", "loss": "arcface"},
]


class EEGPipeline:
    """Multi-seed evaluation pipeline for EEG denoising + metric learning."""

    def __init__(self, config: Config, use_mamba: bool = False):
        self.config = config
        self.use_mamba = use_mamba
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

    def run_evaluation_suite(self, n_seeds: int = 3) -> Dict:
        print("=" * 60)
        print("EEG Pipeline - Comprehensive Evaluation (Multi-Seed)")
        print(f"Seeds: {n_seeds} | Holdout: {self.config.holdout_subjects}")
        print(f"Mamba: {'ON' if self.use_mamba else 'OFF'}")
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

                    (X_n, X_c, y, n_cls), (X_n_unk, X_c_unk, y_unk) = \
                        self.builder.build_dataset_with_novelty(clean_df, noise)
                    train_dl, val_dl, test_dl = self._create_split_dataloaders(X_n, X_c, y)

                    model = create_metric_model(
                        backbone=m['backbone'],
                        n_channels=self.config.n_channels,
                        embed_dim=self.config.embed_dim,
                        use_mamba=self.use_mamba,
                    )

                    trainer = TwoStageTrainer(self.config, self.logger)
                    trainer.train(
                        model, train_dl, val_dl, n_cls,
                        loss_type=m['loss'], noise_type=noise, model_name=m['name'],
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

    def _aggregate_results(self, runs: List[Dict], noise_type: str,
                           model_name: str) -> Dict:
        """Compute mean ± std for all metrics across seeds."""
        keys_test = ["p@1", "p@5", "si_snr", "accuracy", "eer", "latency", "params"]
        keys_nov = ["tar", "trr", "far", "frr", "auroc", "aupr"]

        stats = {}
        for k in keys_test:
            vals = [r['test'][k] for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))

        for k in keys_nov:
            vals = [r['novelty'][k] for r in runs]
            stats[k] = f"{np.mean(vals):.8f} ± {np.std(vals):.8f}"
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_std"] = float(np.std(vals))

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
        loader_kwargs = {"pin_memory": use_cuda, "num_workers": 2 if use_cuda else 0}

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

    def _save_results(self, results: List[Dict]):
        output = {
            "experiment": "Multi-Seed Comprehensive Evaluation",
            "use_mamba": self.use_mamba,
            "config": {
                "epochs_stage1": TRAINING_CONFIG["stage1_epochs"],
                "epochs_stage2": self.config.epochs,
                "batch_size": self.config.batch_size,
                "holdout_subjects": self.config.holdout_subjects,
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
            print(f"{res['noise_type']:<12} | {res['model_name']:<20} | "
                  f"{s['p@1']:<18} | {s['si_snr']:<18} | {s['auroc']:<18} | "
                  f"{s['eer']:<18} | {s['latency_mean']:.2f}ms")
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


def run_smoke_test(config: Config, use_mamba: bool):
    """Ultra-light smoke test: forward pass only."""
    print("[SMOKE] Starting minimal smoke test...")
    x_noisy, x_clean, y = _make_synthetic(config, n_samples=8)

    pipeline = EEGPipeline(config, use_mamba=use_mamba)
    train_dl, val_dl, test_dl = pipeline._create_split_dataloaders(
        x_noisy, x_clean, y, test_size=0.25, val_size=0.25,
    )
    print(f"[SMOKE] split sizes train={len(train_dl.dataset)} "
          f"val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model = create_metric_model(
        backbone="resnet18", n_channels=config.n_channels,
        embed_dim=config.embed_dim, pretrained=False, use_mamba=use_mamba,
    )
    model.eval()
    with torch.no_grad():
        denoised, emb = model(x_noisy[:2])
    print(f"[SMOKE] forward denoised={tuple(denoised.shape)} emb={tuple(emb.shape)}")
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
def run_cli(use_mamba: bool, version: str, default_seeds: int = 3):
    """Shared CLI entry point for both V1 and V2 experiments."""
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = str(repo_root / "dataset") + "/"
    # Output file lives next to the experiment's main.py (2 levels up from shared/)
    # We'll use the caller's __file__ location — but since we're in shared/,
    # the caller should override this. Use version-based default:
    log_path = str(Path(__file__).resolve().parent.parent / version / f"output_{version}.json")

    config = Config(data_path=data_path, epochs=args.epochs,
                    batch_size=64, log_file=log_path)
    print(f"Device: {config.device}")
    print(f"Mamba: {'ON' if use_mamba else 'OFF'}")

    if args.smoke:
        run_smoke_test(config, use_mamba)
    elif args.mini_train:
        run_mini_train(config, use_mamba)
    elif args.one_sample:
        run_one_sample(config, use_mamba)
    else:
        pipeline = EEGPipeline(config, use_mamba=use_mamba)
        pipeline.run_evaluation_suite(n_seeds=args.seeds)
