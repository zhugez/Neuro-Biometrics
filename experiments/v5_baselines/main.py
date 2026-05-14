"""V5 Prior-work baselines: MindID + BrainNet on the AEP-Hybrid protocol.

This experiment re-implements two cited prior-work architectures and
trains them under the exact same protocol used by the main paper
(WaveNet denoiser, held-out subjects, three noise families, multi-seed).
The point is to give the comparison table a row that is *not* taken at
face value from the source paper's own dataset.

    * MindID   — LSTM + attention encoder, ArcFace head.
    * BrainNet — 1D CNN tower trained with FaceNet-style triplet loss.

Both share the WaveNet denoiser with the main pipeline so the denoising
stage is held constant; only the encoder + classification head differs.
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
        version="v5_baselines",
        default_seeds=3,
        models=BASELINE_MODELS,
    )
