## Why

The V4 multimodal experiment path is intended to compare a fused EEG-plus-spectrogram model against the existing V2 EEG-only baseline, but the current implementation cannot reliably complete a training step. The blocking issues are concentrated in the V4 code path and prevent end-to-end validation of the newest pipeline.

## What Changes

- Fix blocking spectrogram-branch forward errors and remove hard-coded shape assumptions that make the branch fragile to STFT configuration changes.
- Correct bimodal fusion so cross-attention uses separate directional weights and the fused embedding remains appropriate for metric-learning losses.
- Repair spectrogram dataset preprocessing by removing ambiguous shape handling, switching to per-channel normalization, and consolidating duplicate spectrogram generation logic.
- Expose the V4 multimodal factory option needed to control spectrogram embedding size without breaking existing callers.
- Add a synthetic one-step smoke test and align the V4 trainer and pipeline expectations around tensor order, device placement, and dtype handling.

## Capabilities

### New Capabilities
- `v4-multimodal-training`: The V4 multimodal pipeline can build the model, consume EEG and spectrogram tensors with the expected shapes, and complete a finite forward/backward training step with sensible defaults.

### Modified Capabilities
- None.

## Impact

- Affected code: `experiments/shared/model_spectrogram.py`, `experiments/shared/fusion.py`, `experiments/shared/model_multimodal.py`, `experiments/shared/dataset_spectrogram.py`, `experiments/shared/trainer_bimodal.py`, `experiments/v4_multimodal/config.py`, `experiments/v4_multimodal/main.py`, `experiments/v4_multimodal/pipeline.py`
- Affected validation: add a focused V4 smoke test for one synthetic training step
- Public API impact: preserve existing call sites, with an optional `spec_embed_dim` kwarg added where needed
- Dependencies: no new runtime dependencies; continue using the existing PyTorch and mamba-ssm stack
