"""V1 Baseline: WaveNet denoiser + ResNet embedder (no Mamba)."""
import sys
from pathlib import Path

# Add experiments/ to path so shared package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.pipeline import run_cli

if __name__ == "__main__":
    run_cli(use_mamba=False, version="v1_baseline", default_seeds=3)
