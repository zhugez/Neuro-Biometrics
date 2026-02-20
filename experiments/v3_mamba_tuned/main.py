"""V3 Mamba Tuned: isolated experiment track for V2-improvement runs."""
import sys
from pathlib import Path

# Add experiments/ to path so shared package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.pipeline import run_cli


if __name__ == "__main__":
    # Keep V1/V2 untouched; V3 runs in its own output path and defaults.
    run_cli(use_mamba=True, version="v3_mamba_tuned", default_seeds=10)
