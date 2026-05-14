## 0. Audit existing artifacts (~30 min, no GPU)

- [x] 0.1 Unzip `D:\Neuro-Biometrics\out\03_Model_Outputs_JSON-*.zip`
      and inventory what is inside (per-seed embeddings? checkpoints?
      metric JSONs only?). If only JSONs are present, retraining V4
      and V1 with embedding dump enabled is needed for Fig. 4 and
      bootstrap (~3 + 9 = ~12h GPU added)
      **FOUND:** 9 JSONs contain stats (mean/std over 3 seeds) +
      best_run (CMC/ROC/DET arrays for 1 seed). NO raw embeddings.
      → must execute task 3.4.1 (retrain V1+V4 with --dump-embeddings,
      ~12h added to GPU budget; total now 49.5h, leaves 10.5h buffer).
- [x] 0.2 Locate the EEGNet baseline output the user mentions (search
      for `eegnet`, `EEGNet`, `lawhern` in repo); record its results
      file path and which noise families / heads it covers
      **NOT FOUND.** No EEGNet implementation in `experiments/`; only
      mentions in citations and "planned for next round" notes in
      `docs/ssrc2026/main.tex`. User decision: drop re-implementation,
      use cited numbers only (see updated D5 in design.md and updated
      "Prior-work comparison table" requirement in spec.md).
- [x] 0.3 Verify `experiments/shared/trainer_bimodal.py` already
      saves test embeddings; if not, add an embedding-dump flag
      (no model change)
      **NOT FOUND.** Trainer does not currently dump per-seed test
      embeddings. Will add a `--dump-embeddings` flag in task 3.4.1
      and re-train V1 + V4 to materialise them.
- [x] 0.4 Confirm `MPerClassSampler` configurable via
      `experiments/v4_multimodal/config.py:m_per_class`
      **CONFIRMED.** `config.py:18-19` exposes
      `use_m_per_class_sampler: bool = True` and
      `m_per_class: int = 4`. `pipeline.py:238-258` wires
      MPerClassSampler from these. CLI flags
      `--no-m-per-class-sampler` and `--m-per-class` already exist
      (`pipeline.py:513`, `541-542`).
- [x] 0.5 Confirm `aux_eeg_loss_weight` / `aux_spec_loss_weight`
      are config-flag-controlled (file: `config.py:23-24`)
      **CONFIRMED.** `config.py:20-21` exposes
      `aux_eeg_loss_weight: float = 0.3` and
      `aux_spec_loss_weight: float = 0.2`. Variant `no_aux` in task
      3.1.4 sets both to 0 via the existing config path.

## 1. Phase A — No-GPU analyses (~1-2 days)

- [ ] 1.1 Write `experiments/analysis/bootstrap_significance.py`:
      load V1 and V4 per-seed embeddings, compute paired P@1 deltas
      per noise family, bootstrap-resample queries (B=10000) and
      output 95% CI on V4 - V1 mean delta to a JSON
- [ ] 1.2 Write `experiments/analysis/viz_denoising.py`: for one
      representative epoch per noise family (Gaussian / power-line /
      EMG), plot waveform pre/post denoiser and STFT pre/post; save
      as `docs/apsipa2026/figures/fig3_denoising.pdf`
- [ ] 1.3 Write `experiments/analysis/viz_embeddings.py`:
      compute t-SNE of V1 vs V4 fused embeddings on the test pool
      (same epochs both models), color by subject ID; save as
      `docs/apsipa2026/figures/fig4a_tsne.pdf`
- [ ] 1.4 Write `experiments/analysis/viz_gate.py`: dump mean gate
      vector per noise family from a V4 forward pass on test set,
      render as a 128-d × 3 heatmap; save as
      `docs/apsipa2026/figures/fig4b_gate.pdf`
- [ ] 1.5 Write `experiments/analysis/viz_lambda.py` (run after
      Phase C completes): render P@1 over (λ_e, λ_s) as a 3×3
      heatmap; save as `docs/apsipa2026/figures/fig5a_lambda.pdf`
- [x] 1.6 Write `experiments/analysis/build_priorwork_table.py`:
      assemble the cited-only prior-work comparison table (≥3 rows
      from MindID, BrainNet, ChronoNet, EEGNet etc.) with a
      per-row footnote stating the protocol difference; emit a LaTeX
      `tabular` body to be pasted into `main.tex`. Caption states
      explicitly that no external baseline was re-implemented under
      the same noise protocol in this paper.
      **DONE THIS SESSION.** Script writes
      `docs/apsipa2026/tables/priorwork.tex` (6 rows: MindID,
      EEG-Triplet, BrainNet, EEGNet, EEG-Mamba, AISEI v0) plus
      a "This work (V4)" block; main.tex now `\input`s it after
      Table~\ref{tab:cross} and references it from Limitations (iv).

## 2. Phase B — Prose updates (~4-6 hours, no GPU)

### 2.1 Abstract

- [x] 2.1.1 Trim Abstract from ~207 words to ≤100 words. Keep the
      headline numbers (P@1 82.4 / 87.7 / 83.8 and SI-SNR
      12.15 / 32.38 / 13.83) and the AUROC ≈ 0.50 verification-gap
      sentence. Drop the full "(i)/(ii)/(iii)" enumeration in favour
      of one summary sentence
      **DONE.** Now 95 words.
- [ ] 2.1.2 Verify line count of compiled abstract ≤ 8 lines after
      `latexmk` rebuild (deferred to Phase 5 build)

### 2.2 Introduction (R#1: problem statement, I/O, V1-V4)

- [x] 2.2.1 Reorder Intro paragraphs: opening problem statement (one
      paragraph stating EEG biometric input/output formally), then
      practical challenges, then the prior-pipeline observations,
      then the contribution list
      **DONE in earlier session.** Para 1 = motivation + I/O +
      challenges; Para 2 = prior-pipeline observations + 3
      motivation-first contributions; Para 3 = V1-V4 intro.
- [x] 2.2.2 Insert a one-paragraph V1-V4 introduction at the end of
      Intro with one line per version (V1 = WaveNet only baseline;
      V2 = + midpoint Mamba; V3 = single-seed H100 quick run, used
      as directional reference only; V4 = full multimodal)
      **DONE.** main.tex lines 108-116.
- [x] 2.2.3 Verify each contribution paragraph in Intro names the
      observation it addresses (motivation-first, not list-first)
      **VERIFIED.** Each (i)/(ii)/(iii) leads with the limitation
      observed on the prior pipeline before naming the change.

### 2.3 Method clarifications (R#1: identity embedding, σ, W_g, etc.)

- [x] 2.3.1 Replace remaining `identity embedding` mentions with
      `biometric embedding` **DONE in earlier session.**
- [x] 2.3.2 σ + W_g definition and gate role
      **DONE.** main.tex lines 256-259.
- [x] 2.3.3 L2-normalization layer explanation
      **DONE.** main.tex lines 173 + 244-245.
- [x] 2.3.4 `cross-attention` → `gated self-attention fusion`,
      keep one parenthetical for citation continuity
      **DONE.** Single parenthetical retained in Method §III.B.
- [x] 2.3.5 Table III row label `+ Spec + Fusion`
      **DONE.** main.tex line 488.
- [x] 2.3.6 SI-SNR scale-invariance purpose sentence
      **DONE.** main.tex line 213.
- [x] 2.3.7 Spectrogram Mamba sweep purpose (AST analogy)
      **DONE THIS SESSION.** Sentence "Following the
      audio-spectrogram-transformer intuition…" added before
      the two sweeps.
- [ ] 2.3.8 Projection-head purpose clause (representation vs
      metric-space) — minor, keep current sentence
- [x] 2.3.9 Move λ_e, λ_s, m=4 numerical values to Experimental
      Setup with forward reference to Tables ref{tab:lambda} and
      ref{tab:msweep}
      **DONE THIS SESSION.** Method now defers; Setup
      "Hyperparameters" paragraph holds the values.

### 2.4 Experimental Setup (R#1: AUROC/EER, V3 fairness, V1-V4 visual)

- [x] 2.4.1 AUROC + EER formula sentences
      **DONE in earlier session.** main.tex lines 372-377.
- [x] 2.4.2 Score vs metric distinction
      **DONE.** main.tex lines 366-368.
- [x] 2.4.3 V1-V4 itemized list as visual anchor in Setup
      **DONE.** Itemized list lines 340-353 (table form skipped to
      save column inches).
- [x] 2.4.4 V3 reframed as H100 directional reference
      **DONE.** lines 346-349.
- [x] 2.4.5 Redundant `P@5 ≤ P@1` sentence removed
      **DONE in earlier session.**
- [x] 2.4.6 Seeds {0, 1, 2} disclosed
      **DONE.** line 331.

### 2.5 Results & Discussion (R#1: deep analysis, viz callouts, hedge)

- [x] 2.5.1 V3 commentary now fact-based with hardware/seed
      disclaimer
      **DONE THIS SESSION.**
- [ ] 2.5.2 Bootstrap CI interpretation paragraph (depends on
      Phase 1.1 + Phase 3.4.1 retrain)
- [ ] 2.5.3 Fig. 3 callout (depends on Phase 1.2 figure)
- [ ] 2.5.4 Fig. 4 callout (depends on Phase 1.3 + 1.4 figures)
- [ ] 2.5.5 Fig. 5 callout (depends on Phase 1.5 figure)
- [x] 2.5.6 Limitations paired with future work
      **DONE THIS SESSION.** Four numbered limitations, each with
      paired remedy: dataset size → multi-cohort; synthetic noise
      → live-environment; AUROC ≈ 0.5 → Platt calibration;
      no same-protocol re-implementation / cross-dataset →
      journal-version scope.

### 2.6 Acknowledgement

- [x] 2.6.1 Funding clause added
      **DONE THIS SESSION.** "no external funding to disclose".

## 3. Phase C — GPU experiments (~37.5h sequential, leaves 22.5h buffer)

### 3.1 Component ablation (~18h: 4 variants × 3 seeds × 1.5h)

- [ ] 3.1.1 Add a `--ablation` flag to
      `experiments/v4_multimodal/main.py` accepting one of
      `{full, no_gate, no_self_attn, no_aux, no_mamba_spec}`
- [ ] 3.1.2 Implement variant `no_gate` in
      `experiments/shared/fusion.py`: replace gate by `Concat → Linear`
      to 128-d (no soft selector)
- [ ] 3.1.3 Implement variant `no_self_attn` in
      `experiments/shared/fusion.py`: bypass MHA, feed raw concat
      directly into the gate
- [ ] 3.1.4 Implement variant `no_aux` by setting
      `aux_eeg_loss_weight = aux_spec_loss_weight = 0`
- [ ] 3.1.5 Implement variant `no_mamba_spec` in
      `experiments/shared/model_spectrogram.py`: replace the two
      Mamba sweeps with `Conv1d → BN → ReLU → Conv1d → BN`
- [ ] 3.1.6 Run all 4 variants × 3 seeds (R34 + ArcFace, all 3 noise
      families); persist outputs as
      `experiments/v4_multimodal/output_ablation_<variant>.json`
- [ ] 3.1.7 Verify each output JSON contains `p@1_mean`, `p@1_std`,
      `p@1_ci95` per noise type for the R34+Arc head

### 3.2 λ-grid coarse sensitivity (~13.5h: 9 cells × 1 seed × 1.5h)

- [ ] 3.2.1 Add a `--lambda-sweep` flag to
      `experiments/v4_multimodal/main.py` accepting two CSV lists for
      λ_e and λ_s
- [ ] 3.2.2 Run the 3 × 3 grid (λ_e ∈ {0.1, 0.3, 0.5} × λ_s ∈
      {0.0, 0.2, 0.5}) on R34+ArcFace + power-line, single seed
- [ ] 3.2.3 Persist outputs as
      `experiments/v4_multimodal/output_lambda_grid.json`
- [ ] 3.2.4 If the grid peak is far from (0.3, 0.2) (delta > 2 pp),
      retrain V4 3-seed at the new optimum and update headline
      Table II numbers

### 3.3 m-per-class sweep (~6h: 4 points × 1 seed × 1.5h)

- [ ] 3.3.1 Add a `--m-per-class` override flag in main.py (or
      reuse existing config.py flag)
- [ ] 3.3.2 Run m ∈ {2, 4, 8, 16} on R34+ArcFace + power-line,
      single seed
- [ ] 3.3.3 Persist outputs as
      `experiments/v4_multimodal/output_m_sweep.json`

### 3.4 (Conditional) Re-train V1 + V4 with embedding dump (~12h)

- [ ] 3.4.1 IF task 0.1 finds no per-seed embeddings, add an
      `--dump-embeddings` flag to V1 and V4 main.py and re-train
      both with 3 seeds (V1: ~3h, V4: ~9h)
- [ ] 3.4.2 ELSE skip; reuse existing embeddings under
      `D:\Neuro-Biometrics\out\`

## 4. Phase D — Integrate into manuscript (~half day)

- [ ] 4.1 Replace Table III ablation with new component ablation
      results (5-row block, each with mean±std and 95% CI on
      P@1 delta vs V1 from task 1.1)
- [ ] 4.2 Insert new prior-work comparison table (Table IV) using
      the LaTeX body emitted by task 1.6
- [ ] 4.3 Insert Fig. 3 (denoising), Fig. 4 (t-SNE + gate), and
      Fig. 5 (λ heatmap + m sweep) at the placeholders set in tasks
      2.5.3 / 2.5.4 / 2.5.5
- [ ] 4.4 Update Conclusion to reference the new ablation findings
      and prior-work comparison
- [ ] 4.5 Drop Fig. 1 (`two_stages.tex`) — overlaps Fig. 2; reclaim
      ~⅓ column. Update text references that point to Fig. 1 to
      point to Fig. 2 instead

## 5. Phase E — Build, sync, validate (~1h)

- [ ] 5.1 Run `latexmk -pdf -interaction=nonstopmode -halt-on-error
      main.tex` in `docs/apsipa2026/`
- [ ] 5.2 Verify the producing log shows
      `Output written on main.pdf (5 pages, ...)`, 0 overfull,
      0 undefined refs/citations, 0 errors
- [ ] 5.3 If page count exceeds 5, apply D7 mitigations in this
      order: tighten Method, then compress Discussion, then trim
      Related Work, then demote Fig. 5 to supplementary, then drop
      m-sweep inline plot
- [ ] 5.4 Copy `docs/apsipa2026/main.pdf` over
      `docs/APSIPA2026_NeuroBiometrics.pdf` (canonical-to-shipped
      direction only, per `apsipa-build-artifacts`)
- [ ] 5.5 Recompute SHA-256 of both PDFs and confirm equal
- [ ] 5.6 Run `openspec validate apsipa-reviewer1-revision --strict`
      and confirm no errors

## 6. Cleanup

- [ ] 6.1 Archive the superseded
      `apsipa-reviewer1-prose-fix` proposal (move from worktree to
      `openspec/changes/archive/`) with a note in its archive entry
      pointing to this change
- [ ] 6.2 Archive the superseded
      `apsipa-reviewer1-deep-fix` proposal similarly
- [ ] 6.3 Run `openspec status --change apsipa-reviewer1-revision`
      and confirm `isComplete: true`
