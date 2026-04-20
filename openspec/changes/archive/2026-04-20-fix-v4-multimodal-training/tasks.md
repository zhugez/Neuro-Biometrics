## 1. Model and fusion fixes

- [x] 1.1 Rename the spectrogram branch shape variables and replace the literal STFT shape check with rank-and-channel validation in `experiments/shared/model_spectrogram.py`
- [x] 1.2 Split fusion into separate directional attention modules and remove the terminal output ReLU in `experiments/shared/fusion.py`
- [x] 1.3 Expose `spec_embed_dim` in `create_multimodal_model` and project spectrogram embeddings into the fusion width when needed in `experiments/shared/model_multimodal.py`

## 2. Dataset and trainer alignment

- [x] 2.1 Make `compute_spectrogram` accept only `(B, C, T)` tensors and reuse it for every dataset spectrogram build path in `experiments/shared/dataset_spectrogram.py`
- [x] 2.2 Switch spectrogram normalization to per-sample, per-channel normalization and remove the `EEGDatasetBuilder = None` placeholder in `experiments/shared/dataset_spectrogram.py`
- [x] 2.3 Keep spectrogram tensors aligned with the EEG device and effective dtype, and verify bimodal batch tuple ordering, in `experiments/shared/trainer_bimodal.py` and `experiments/v4_multimodal/pipeline.py`

## 3. V4 defaults and regression coverage

- [x] 3.1 Wire sensible V4 multimodal defaults for spectrogram settings and model construction in `experiments/v4_multimodal/config.py`, `experiments/v4_multimodal/main.py`, and `experiments/v4_multimodal/pipeline.py`
- [x] 3.2 Add `tests/test_v4_smoke.py` to run one synthetic multimodal forward and backward step and assert finite loss plus gradients across all V4 branches
- [x] 3.3 Run the focused V4 smoke test and record the passing command output for review
