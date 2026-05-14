## Context

Reviewer #1 left a hard reject ranging across prose, missing
experiments, missing visualisations, and missing prior-work
comparisons. The 5-page APSIPA envelope is tight — 4 pages of body
plus ~1 page references. Every new artifact (figure, table, paragraph)
must justify its column inches against an existing artifact.

Two prior side-worktree proposals already covered prose, but stalled
without merge. This change replaces them and adds the experiment /
visualisation / comparison content needed to make a credible
resubmission.

GPU budget: 60h on a single RTX 5090. V4 trains in ~90 min/seed
(user-confirmed). V2 ~60 min/seed, V1 ~45 min/seed.

## Decisions

### D1. Significance test — paired bootstrap on stored embeddings

The reviewer asks whether V4 - V1 ≈ 1-2 pp is meaningful. Retraining
to add seeds is expensive; instead reuse the existing per-seed test
embeddings under `D:\Neuro-Biometrics\out\` (5 zip files,
`03_Model_Outputs_JSON-...zip`).

For each (noise, head) cell, compute paired P@1 differences across
the three seeds, bootstrap-resample queries with replacement
(B=10000), and report the 95% CI on the mean delta. Cells whose CI
excludes zero are flagged in the ablation table caption; cells whose
CI straddles zero are explicitly described as "magnitude observation
within across-seed std" rather than a tested claim.

This is honest, cheap, and directly answers the reviewer without
fabricating a stronger claim than three seeds support.

### D2. λ-grid scope — coarse 3 × 3, single seed, one head, one noise

A full 5 × 5 × 3 × 3 grid would consume ~67h. The defensible minimum
is a 3 × 3 sensitivity around (λ_e, λ_s) = (0.3, 0.2) on the strongest
configuration (ResNet-34+ArcFace + power-line). Single-seed is
acceptable here because the goal is sensitivity (does P@1 vary
smoothly?) not significance.

If the heatmap shows P@1 monotonically rising with λ_e, the existing
λ_e = 0.3 choice will be flagged as conservative; if it shows a clear
peak elsewhere, the paper will say so honestly.

### D3. m-sweep scope — 4 single-seed points, one config

Reviewer claims m = 4 is "small for SSL". Strictly the framework is
not contrastive SSL — it is class-balanced metric learning. Two
actions: (a) one Method sentence clarifying that distinction;
(b) m ∈ {2, 4, 8, 16} sweep on R34+ArcFace + power-line, single seed,
to show m = 4 is a defensible operating point and not arbitrary. Skip
m = 32 because batch size = 256 / 16 = 16 identities ⇒ already at the
boundary of class-balance assumption with 16 non-holdout subjects.

### D4. Component ablation — V4 minus one block per row, 3 seeds

Per-component knockouts to isolate which block delivers the V4 gain
above V2:

| Variant            | Drop                                            |
|--------------------|-------------------------------------------------|
| V4                 | (full)                                          |
| V4 - gate          | replace gate with concat + Linear → 128         |
| V4 - self-attn     | identity (skip MHA), keep gate                  |
| V4 - aux           | λ_e = λ_s = 0                                   |
| V4 - mamba_spec    | replace Mamba sweeps with 2 BatchNorm-Conv1d    |

Three seeds each so the column is comparable to the existing V4
3-seed numbers. Total ~18h (4 variants × 3 seeds × 1.5h).

### D5. Prior-work comparison strategy — cited-only

Audit (task 0.2) confirmed there is no EEGNet baseline file in the
repo, contradicting the initial expectation. User decision:
**do not re-implement** any external baseline within this revision;
build the comparison table from cited numbers only.

Build Table III with at least three rows of cited numbers from
prior EEG biometric papers (MindID [zhang2017mindid],
BrainNet [fallahi2023brainnet], ChronoNet, EEGNet
[lawhern2018eegnet], etc.). Each row carries an explicit footnote
naming the protocol difference (clean vs noise-injected dataset,
closed- vs open-set, dataset identity). The caption states up front
that no external baseline was re-implemented under the same
noise-injection protocol in this paper, so the table is contextual
and does not support a "win" claim against V4.

This is honest under-claim rather than a fabricated comparison; it
also leaves a clear future-work direction (re-implement EEGNet
under same protocol, planned for the journal version).

### D6. Visualisations — three figures, surgical placement

- **Fig. 3** (denoising evidence): 3 × 2 panel grid (noise family
  rows × waveform/spectrogram columns). Use one representative epoch
  per noise type. Annotate the 50 Hz line in spectrograms with an
  arrow. ~half-column wide.
- **Fig. 4** (embedding + gate): t-SNE side by side V1 vs V4 (left),
  gate-vector heatmap by noise (right). Half-column each, share a
  caption. Use seaborn / matplotlib defaults; no fancy 3D.
- **Fig. 5** (hyperparameter): 3 × 3 P@1 heatmap (λ_e × λ_s) and
  inline m sweep mini-plot. If page tight, demote to supplementary.

All figures rendered at 300 DPI as PDF (vector for matplotlib).

### D7. Page envelope — what to cut

5-page hard cap. Adding a comparison table + ablation rows + 2-3
figures requires reclamation. Plan:

- Drop Fig. 1 (`two_stages.tex`). It overlaps Fig. 2 (`v4_arch.tex`).
  Save: ~⅓ column.
- Tighten Method §III.A--D paragraphs from ~1.5 cols to ~1 col by
  consolidating gloss sentences. Save: ~½ col.
- Compress Discussion bullets into 4 short paragraphs (Where
  multimodal helps / Magnitude / Verification gap / Why Mamba alone
  is not enough). Save: ~⅓ col.
- Trim Related Work to 3 paragraphs (EEG biometrics, denoising +
  state-space, metric learning + multimodal fusion). Save: ~¼ col.

If after all of the above the build still overflows, demote Fig. 5
to supplementary first, then drop the m-sweep result inline (keep in
text).

### D8. Naming — `cross-attention` → `gated self-attention fusion`

Rename throughout the manuscript text, abstract, captions,
contributions, conclusion. Keep one parenthetical in Method §III.B
"(referred to as a cross-attention module in our prior work
[vu2026aisei])" so citation discoverability is preserved. Update
Table III row label `+ Spec + CrossAttn` → `+ Spec + Fusion`. Code
identifiers in `experiments/shared/fusion.py` need not change unless
they appear in figure or table renderings.

### D9. Acknowledgement

Reviewer flagged "Thiếu phần này" but the manuscript already has a
one-line Acknowledgment crediting the PhysioNet dataset (line 558).
Reviewer likely expected funding / IRB. We have neither in this
codebase; per user guardrail, do not invent sponsors. Keep the
existing line and note in Limitations that no external funding
information is available.

## Risks / Trade-offs

- **Bootstrap CI may include zero for several cells** → that is the
  honest finding; report it. Do not p-hack by switching to per-query
  bootstrap if seed-bootstrap fails.
- **λ heatmap may peak away from (0.3, 0.2)** → if so, retrain V4
  3-seed at the new optimum (~9h, fits buffer) and update headline
  numbers. Otherwise flag the chosen point as "near optimum".
- **Component ablation may show fusion = concat is comparable** →
  honest finding; downgrade fusion contribution claim and shift
  emphasis to the spec branch / Mamba sweeps. Do not silently drop
  the row.
- **EEGNet baseline may beat V4 on Gaussian** (likely on clean signal
  it does) → keep the row, frame as "noise-robust trade-off:
  V4 wins on power-line and EMG, EEGNet competitive on Gaussian".
- **Page overflow** → mitigation in D7. Last resort is demoting
  Fig. 4's gate heatmap to supplementary (keep t-SNE inline).
- **Out-of-context numbers from cited prior work** → honest footnote
  per row; do not strip the row for being unflattering.

## Migration Plan

This change supersedes two side-worktree proposals. The execution
order is:

1. Audit `out/` artifacts and EEGNet baseline location (Phase 0).
2. Run no-GPU work in parallel with GPU queue start (Phase A + B).
3. Run GPU experiments overnight (Phase C — component ablation
   first, then λ-grid, then m-sweep).
4. Integrate all results into `main.tex` (Phase D).
5. Build, sync PDF, validate (Phase E).
6. Archive the two superseded proposals.
