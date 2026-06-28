"""V6: 16:4 Open-Set Evaluation.

A dedicated experiment line (NOT part of V5) that evaluates models under the
open-set protocol: train on ALL 16 enrolled subjects, reject the 4 holdout
subjects [2, 5, 7, 12]. It reuses the open-set engine already living in
``shared/pipeline.py`` and ``v4_multimodal/pipeline.py`` — V6 is only the
entry point + output naming.

Two model families are evaluated for a "ours vs prior-work under open-set"
comparison (mirroring how V5 set up the prior-work baselines):

    * Flagship    : V4 bimodal (WaveNet+Mamba denoiser + Spectrogram + Fusion)
                    -> output_v6_openset_bimodal.json
    * Prior-work  : MindID (LSTM+Attn, ArcFace) and BrainNet (1D-CNN, Triplet),
                    raw signal, no denoiser
                    -> output_v6_openset_baselines.json

Run (from the repo root):
    python -m experiments.v6_openset.main --seeds 3 --epochs 30
    python -m experiments.v6_openset.main --only bimodal
    python -m experiments.v6_openset.main --only baselines
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO_ROOT = EXPERIMENTS.parent

# Robust regardless of `python -m ...` vs `python path/to/main.py`:
#   - experiments/        -> `shared.*` package
#   - experiments/v4_multimodal/ -> top-level `pipeline` (v4) and `config` (V4Config)
for p in (str(EXPERIMENTS), str(EXPERIMENTS / "v4_multimodal")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.datapreprocessor import Config  # noqa: E402
from shared.pipeline import EEGPipeline      # noqa: E402

# Prior-work baseline models (kept in sync with experiments/v5_baselines/main.py).
BASELINE_MODELS = [
    {"name": "MindID_ArcFace", "embedder": "mindid", "loss": "arcface"},
    {"name": "BrainNet_Triplet", "embedder": "brainnet", "loss": "triplet", "embed_dim": 32},
]


def _data_path() -> str:
    return str(REPO_ROOT / "dataset") + "/"


def run_baselines(args):
    """Prior-work encoders (raw signal, no denoiser) under the 16:4 protocol."""
    out = str(HERE / "output_v6_openset_baselines.json")
    config = Config(data_path=_data_path(), epochs=args.epochs,
                    batch_size=args.batch_size, log_file=out)
    config.num_workers = args.num_workers
    config.optimize_h100 = args.optimize_h100
    print("\n########## V6 open-set: prior-work baselines (no denoiser) ##########")
    pipeline = EEGPipeline(config, use_mamba=False, use_denoiser=False)
    pipeline.run_evaluation_suite_openset(n_seeds=args.seeds, models=BASELINE_MODELS)


def run_bimodal(args):
    """Flagship V4 bimodal model under the 16:4 protocol."""
    # Imported lazily so a `--only baselines` run does not need the bimodal stack.
    import pipeline as v4pipeline          # experiments/v4_multimodal/pipeline.py
    from config import V4Config            # experiments/v4_multimodal/config.py

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    config = V4Config(data_path=_data_path(), epochs=args.epochs,
                      batch_size=args.batch_size,
                      log_file=str(log_dir / "v6_openset.log"))
    config.output_file = str(HERE / "output_v6_openset_bimodal.json")
    config.num_workers = args.num_workers
    config.optimize_h100 = args.optimize_h100
    config.spectrogram_source = "denoised"
    config.spectrogram_n_fft = 128
    config.spectrogram_hop_length = 64
    print("\n########## V6 open-set: flagship V4 bimodal ##########")
    pipeline = v4pipeline.MultimodalEEGPipeline(config, use_mamba=True)
    pipeline.run_evaluation_suite_openset(n_seeds=args.seeds)


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Biometrics V6 — 16:4 open-set evaluation")
    parser.add_argument("--only", choices=["both", "baselines", "bimodal"],
                        default="both",
                        help="Which model family to run under 16:4 (default: both)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Stage-2 epochs (default: 30)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker count (default: 8)")
    parser.add_argument("--optimize-h100", action="store_true",
                        help="Enable torch.compile and bfloat16 mixed precision")
    args = parser.parse_args()

    if args.only in ("both", "baselines"):
        run_baselines(args)
    if args.only in ("both", "bimodal"):
        run_bimodal(args)


if __name__ == "__main__":
    main()
