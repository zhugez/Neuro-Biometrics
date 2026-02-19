"""V2 Mamba: WaveNet + MambaBlock denoiser + ResNet embedder."""
import sys
from pathlib import Path

# Add experiments/ to path so shared package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.pipeline import run_cli

if __name__ == "__main__":
    run_cli(use_mamba=True, version="v2_mamba", default_seeds=5)
