## Context

The V4 path combines a WaveNet-based EEG denoiser, a ResNet metric embedder, a spectrogram Mamba branch, and a cross-attention fusion head. The implementation spans `experiments/shared/model_spectrogram.py`, `experiments/shared/fusion.py`, `experiments/shared/model_multimodal.py`, `experiments/shared/dataset_spectrogram.py`, `experiments/shared/trainer_bimodal.py`, and `experiments/v4_multimodal/{config.py,main.py,pipeline.py}`.

The current V4 implementation has multiple concentrated failures: the spectrogram branch can crash on first forward because its local shape variables shadow `torch.nn.functional`; the branch also hard-codes one STFT shape; fusion reuses one attention module for both directions and clamps the final embedding with a ReLU; the spectrogram dataset path mixes ambiguous input shapes with global normalization and duplicate STFT implementations; the multimodal factory does not expose the spectrogram branch width; and the V4 path does not have a focused regression test that exercises one full multimodal train step.

Constraints for the change are strict: only the V4 path and a dedicated smoke test may change, existing V1/V2/V3 behavior must remain untouched, existing public call sites should stay compatible unless an additive kwarg is needed, and no new dependencies may be introduced.

## Goals / Non-Goals

**Goals:**
- Make the V4 multimodal model complete a forward and backward training step with default settings.
- Remove the known V4-only correctness bugs in spectrogram modeling, fusion, preprocessing, and trainer handoff.
- Keep the default V4 interface stable while exposing a configurable spectrogram embedding width.
- Add one focused synthetic smoke test that catches shape, device, dtype, loss, and gradient regressions in the V4 path.

**Non-Goals:**
- Redesign the V4 architecture beyond the targeted bug fixes.
- Change any V1, V2, or V3 code paths, configs, or saved results.
- Introduce new training objectives, new dependencies, or broad refactors outside the V4 path.
- Optimize the H100-specific `torch.compile` path beyond keeping it compatible with the fixes.

## Decisions

### 1. Standardize the spectrogram input contract around shaped EEG tensors
The shared spectrogram helper will treat `(B, C, T)` as the only supported EEG input shape for the V4 path. The dead `(B*C, T)` branch will be removed rather than preserved with ambiguous reshape logic.

This keeps the data contract aligned with the existing V4 dataset builders and trainer, which already operate on shaped EEG tensors. The alternative was to preserve both shapes and attempt to infer batch and channel counts during reshape, but that keeps the current ambiguity alive and does not serve any verified V4 caller.

### 2. Make spectrogram shape validation config-tolerant instead of literal
The spectrogram model will continue to require a 4D tensor with the expected channel count, but it will stop enforcing a single literal `65 x 13` STFT shape inside `forward`. The V4 path already carries `spectrogram_n_fft` and `spectrogram_hop_length` configuration, so the model should tolerate alternate time-frequency shapes as long as the tensor rank and channel count are valid.

The alternative was to derive one exact expected `(F, T_spec)` pair inside every forward pass and reject any other shape. That approach tightly couples the model to one signal-length assumption and makes the branch unnecessarily brittle when the STFT configuration is changed for experimentation.

### 3. Consolidate dataset spectrogram generation and preserve channel structure
`dataset_spectrogram.py` will use a single vectorized STFT implementation for dataset building, and all cached spectrogram normalization will be per-sample and per-channel over `(F, T_spec)` only.

Per-channel normalization preserves cross-channel amplitude relationships that are meaningful for EEG biometrics. The alternative was to keep the current global per-sample normalization across all channels, but that discards channel-specific scale information and makes the V4 spectrogram branch less faithful to the input signal.

### 4. Make fusion directional and keep the fused embedding signed
The fusion module will use separate attention layers for EEG-to-spectrogram and spectrogram-to-EEG attention. Its projection head will keep the fused embedding signed by removing the terminal ReLU that currently forces the output into the non-negative half-space.

The alternative was to keep the shared attention weights or move the ReLU deeper into a larger residual projection block. Reusing one attention module hides directional differences, while a larger projection redesign is a broader architectural change than this bug-fix change requires.

### 5. Expose `spec_embed_dim` and project into the fusion width when needed
`create_multimodal_model` will accept `spec_embed_dim` as an additive kwarg. When `spec_embed_dim == embed_dim`, behavior stays unchanged. When the spectrogram branch width differs, the model will project the spectrogram embedding into `embed_dim` before fusion.

The alternative was to expose the kwarg but still require it to equal `embed_dim`. That would surface a configuration knob that cannot safely vary, which is a poor contract for later experiments.

### 6. Add one synthetic regression smoke test for the full multimodal train step
A dedicated `tests/test_v4_smoke.py` test will create synthetic `(noisy, clean, labels, spectrogram)` tensors, run one forward and backward step through the V4 multimodal model, and assert finite loss plus gradient flow through the denoiser, EEG embedder, spectrogram embedder, and fusion modules. Trainer and pipeline changes will stay minimal and only ensure that spectrogram tensors reach the multimodal forward path on the same device and with a compatible effective dtype as the EEG tensors.

The alternative was to rely on the existing pipeline smoke helpers, but those only validate a forward pass and do not protect the loss or backward path that currently matters for end-to-end training.

## Risks / Trade-offs

- [Mamba runtime availability in the test environment] → The smoke test depends on the existing `mamba-ssm` stack from `requirements.txt`; validate in an environment that has the project dependencies installed before treating the change as complete.
- [Changing the fusion output distribution] → Removing the terminal ReLU changes the fused embedding distribution; mitigate by keeping the default width unchanged and validating the output through the finite-loss smoke test.
- [Tighter spectrogram helper contract] → Any undocumented caller that still passes flattened `(B*C, T)` EEG will now fail fast; mitigate by limiting the change to the V4 path and keeping the error explicit.
- [Additional projection when branch widths differ] → Supporting mismatched `spec_embed_dim` adds a small projection layer when widths differ; mitigate by using an identity path when the widths already match.

## Migration Plan

- Apply the code changes only under the V4 multimodal path and add the new smoke test file.
- Keep default configuration values aligned with the existing V4 defaults so current callers still create a model with matching branch widths.
- Validate the change with the new synthetic smoke test before running full experiments.
- No checkpoint, dataset, or results migration is required because the change is isolated to V4 implementation behavior.

## Open Questions

- Whether automated test environments for this repository will always have `mamba-ssm` installed remains an environment question, but it does not block authoring the V4 change itself.
