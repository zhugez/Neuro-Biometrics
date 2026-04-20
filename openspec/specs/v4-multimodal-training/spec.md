## ADDED Requirements

### Requirement: V4 multimodal model completes one training step
The V4 multimodal model SHALL accept paired noisy EEG and spectrogram tensors, produce a denoised EEG output and fused embedding, and support one finite forward and backward training step with the default V4 configuration.

#### Scenario: Synthetic bimodal batch trains end to end
- **WHEN** a batch of noisy EEG, clean EEG, labels, and spectrogram tensors with shapes `(4, 4, 800)`, `(4, 4, 800)`, `(4,)`, and `(4, 4, 65, 13)` is passed through the V4 multimodal model with a metric-learning loss
- **THEN** the loss is finite and gradients are produced for the denoiser, EEG embedder, spectrogram embedder, and fusion modules

#### Scenario: Spectrogram tensors follow the multimodal device path
- **WHEN** the V4 trainer prepares a bimodal batch for CPU or CUDA execution
- **THEN** the spectrogram tensor is moved with the EEG tensors before multimodal forward so the branches do not fail from device or dtype mismatch

### Requirement: V4 spectrogram preprocessing preserves channel structure
The V4 spectrogram preprocessing SHALL accept shaped EEG tensors of `(B, C, T)`, compute spectrograms through one shared implementation, normalize each sample per channel over `(F, T_spec)`, and return dataset items in the order `(noisy, clean, labels, spectrogram)`.

#### Scenario: Cached spectrograms are normalized per channel
- **WHEN** the V4 dataset path precomputes spectrograms for a batch of EEG samples
- **THEN** each sample retains separate normalization statistics for each channel and the cached tensor shape remains `(B, C, F, T_spec)`

#### Scenario: Bimodal batches preserve tuple ordering
- **WHEN** the V4 pipeline builds dataloaders from noisy EEG, clean EEG, labels, and spectrogram tensors
- **THEN** the trainer receives each batch as `(noisy, clean, labels, spectrogram)` without field reordering

### Requirement: V4 fusion supports asymmetric attention and configurable spectrogram width
The V4 fusion path SHALL use separate attention parameters for EEG-to-spectrogram and spectrogram-to-EEG attention, SHALL return a signed fused embedding suitable for metric learning, and SHALL allow the spectrogram branch embedding width to be configured without breaking existing callers.

#### Scenario: Default multimodal construction remains compatible
- **WHEN** the multimodal model is created without a `spec_embed_dim` override
- **THEN** the spectrogram branch uses the main embedding width and the fused embedding shape remains `(B, embed_dim)`

#### Scenario: Custom spectrogram width still fuses correctly
- **WHEN** the multimodal model is created with `spec_embed_dim` different from `embed_dim`
- **THEN** the spectrogram branch output is projected into the fusion width and the model still returns a fused embedding of shape `(B, embed_dim)`

### Requirement: V4 spectrogram forward path tolerates configured STFT sizes
The V4 spectrogram branch SHALL validate tensor rank and channel count without hard-coding a single frequency-bin and time-bin pair, so configured STFT settings can change without causing an immediate forward failure.

#### Scenario: Alternate STFT dimensions are accepted
- **WHEN** the spectrogram input has the configured channel count and four dimensions but different frequency-bin or time-bin counts than `65 x 13`
- **THEN** the spectrogram branch proceeds through the forward path instead of raising a literal shape error tied to one STFT configuration
