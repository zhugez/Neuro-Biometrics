## Why

Reviewer #1 returned a hard reject (1/5, 0/5, 1/5, 1/5; "Reject and not
have opportunity to revise") on the APSIPA 2026 submission. The review
contains ~30 distinct comments grouped along four problem axes:

1. **Framing weakness** — Abstract is too long (207 words vs the
   reviewer's ≤100 ask); Introduction never states the input/output
   problem before listing contributions; V1--V4 are referenced without
   being introduced; multiple terms are undefined or under-motivated
   (`identity embedding`, `unit-norm vectors`, `cross-attention`,
   `σ`, `W_g`, the role of SI-SNR).
2. **Missing experiments** — no λ_e/λ_s ablation, no m-per-class
   ablation, no fusion-component ablation (gate vs concat,
   self-attention vs MLP, aux loss on/off); no paired statistical
   significance test backing the V4>V1 claim (mean deltas are within
   across-seed std).
3. **Missing visualisations** — Discussion makes mechanistic claims
   (50 Hz localisation, EMG high-frequency tilt, Mamba's effect on
   receptive field, denoiser quality) without any waveform,
   spectrogram, t-SNE, or attention-map figure to back them.
4. **Missing prior-work comparison** — the paper cites EEGNet,
   ChronoNet, BrainNet, MindID etc. but reports no numbers for them
   under the same noise-injection protocol; Reviewer reads this as
   "results without comparison".

Two earlier proposals (`apsipa-reviewer1-prose-fix`,
`apsipa-reviewer1-deep-fix`) addressed only axis (1) and the prose
half of axis (4). They live on a side worktree
(`.claude/worktrees/jolly-herschel-289deb/...`) and are
**superseded by this change** at the user's request, to avoid task
overlap and three-way merge work.

This change covers all four axes within a 60h RTX 5090 budget and
preserves the 5-page envelope so the manuscript is ready for a
similar-tier resubmission (workshop / second-tier conference).

## What Changes

### Prose (subsumes the two superseded proposals)

- Trim Abstract to ≤100 words while keeping the headline numbers
  (P@1 82.4 / 87.7 / 83.8 and SI-SNR 12.15 / 32.38 / 13.83).
- Restructure Introduction: opening problem statement (input/output);
  two paragraphs on practical challenges; one paragraph naming
  observations on the prior pipeline that motivate each contribution;
  one paragraph introducing V1--V4 with a one-line description each.
- Method clarifications: define σ, W_g, "unit-norm vectors", explain
  the role of SI-SNR (scale-invariant reconstruction quality for
  source separation), explain why two Mamba sweeps over (frequency,
  time) axes, expand the projection-head purpose, expand the
  fusion-gate role, retire `cross-attention` in favor of
  `gated self-attention fusion` (parenthetical synonym kept once for
  citation continuity to `vu2026aisei`).
- Move λ_e=0.3, λ_s=0.2, m=4 from Method into Experimental Setup as
  fixed hyperparameters; cross-reference the new ablation table.
- Experimental Setup: spell out AUROC and EER formulas; clarify
  scores vs metrics; flag V3 as single-seed H100 quick run, used
  only as a directional reference and not for fair comparisons;
  drop the redundant "P@5 ≤ P@1 by construction" sentence.
- Discussion: hedge mechanistic claims; replace "is consistent with"
  language for any claim that lacks a corresponding figure; pair each
  Limitation with a one-clause future-work direction; keep the existing
  Acknowledgment paragraph (no funding info to add).

### New experiments (GPU budget ~37.5h, leaves 22.5h buffer)

- **Component ablation V4** (4 configurations × 3 seeds, ~18h):
  V4 minus gate, V4 minus self-attention, V4 minus auxiliary losses,
  V4 minus Mamba spectrogram sweeps (CNN-only spec branch).
- **λ-grid coarse** (3 × 3 single-seed sensitivity, ~13.5h):
  λ_e ∈ {0.1, 0.3, 0.5} × λ_s ∈ {0.0, 0.2, 0.5}, reported as a
  small heatmap of P@1 over the (λ_e, λ_s) grid for the
  ResNet-34+ArcFace head on the strongest noise family (power-line).
- **m-per-class sweep** (4 single-seed configurations, ~6h):
  m ∈ {2, 4, 8, 16}, reported as a small line plot to defend m=4 as
  not arbitrary and to address Reviewer's "SSL needs large m" concern.

### New analyses (no GPU)

- **Paired bootstrap significance** test of V4 vs V1 P@1 per noise
  family on the existing per-seed embeddings, reported as a 95% CI on
  the V4-V1 delta in the ablation table caption.
- **Prior-work comparison table** combining (a) re-implemented EEGNet
  numbers under the same noise protocol (already available per user)
  and (b) numbers cited from prior papers under their own protocols
  with explicit footnotes flagging protocol differences.

### New figures (target 2-3 within page envelope)

- **Fig. 3 — Denoising evidence**: stacked panels showing waveform
  before/after denoiser and spectrogram before/after for each noise
  family (Gaussian, 50 Hz line, EMG). Backs the "50 Hz line is
  removed" and "EMG high-frequency tilt" mechanistic claims.
- **Fig. 4 — Embedding geometry and gate behaviour**: t-SNE of
  V1 vs V4 fused embeddings on the test pool (left), and mean gate
  vector heatmap per noise family (right). Backs the
  "fusion separates identity better" and "gate selects branch by
  noise" claims.
- **Fig. 5 — λ heatmap and m line plot** (small inset): backs the
  hyperparameter ablation.

If page envelope tightens, drop Fig. 5 to supplementary and reference
it in the Discussion.

## Impact

- **Affected specs**: `apsipa-paper-claims` (ADDED Requirements
  covering significance reporting, visual evidence, hyperparameter
  ablation, prior-work comparison, fusion-naming consistency,
  abstract length, intro structure).
- **Affected code**: ablation drivers under
  `experiments/v4_multimodal/` (new ablation entry-points),
  visualisation scripts under `experiments/shared/` or a new
  `experiments/visualisations/` (waveform, spectrogram, t-SNE, gate);
  bootstrap significance script.
- **Affected docs**: `docs/apsipa2026/main.tex`,
  `docs/apsipa2026/figures/*` (3 new figures),
  `docs/apsipa2026/main.pdf`, `docs/APSIPA2026_NeuroBiometrics.pdf`
  (rebuild + resync via `apsipa-build-artifacts` invariant).
- **Page envelope**: must stay at 5; add 1 prior-work comparison
  table, 1 ablation row block, ≤3 new figures; tighten Method and
  Discussion to make room.
- **Supersedes**: `apsipa-reviewer1-prose-fix`,
  `apsipa-reviewer1-deep-fix` (both on side worktree). After this
  change merges, archive the two superseded proposals without
  separately deploying them.
