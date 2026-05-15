"""V5 Prior-work baselines: MindID + BrainNet on the AEP-Hybrid protocol.

This experiment re-implements two cited prior-work architectures and
trains them under the same evaluation protocol used by the main paper
(held-out subjects, three noise families, multi-seed) — but **without**
the WaveNet Stage-I denoiser, which is part of our contribution. The
prior-work encoders therefore receive the raw noisy signal directly,
matching the inputs assumed by their original papers and ensuring the
cross-architecture comparison is not contaminated by our denoising stage.

    * MindID   — LSTM + attention encoder, ArcFace head.
    * BrainNet — 1D CNN tower trained with FaceNet-style triplet loss.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.pipeline import run_cli


BASELINE_MODELS = [
    {
        "name": "MindID_ArcFace",
        "embedder": "mindid",
        "loss": "arcface",
    },
    {
        "name": "BrainNet_Triplet",
        "embedder": "brainnet",
        "loss": "triplet",
        "embed_dim": 32,
    },
]


if __name__ == "__main__":
    run_cli(
        use_mamba=False,
        use_denoiser=False,
        version="v5_baselines",
        default_seeds=3,
        models=BASELINE_MODELS,
    )
