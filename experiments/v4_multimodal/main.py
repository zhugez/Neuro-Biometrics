"""V4 Multimodal: WaveNet+Mamba + Spectrogram+Mamba + CrossAttention Fusion."""
import sys
from pathlib import Path

# Add experiments/ to path so shared package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import run_cli

if __name__ == "__main__":
    run_cli(use_mamba=True, version="v4_multimodal", default_seeds=3)
