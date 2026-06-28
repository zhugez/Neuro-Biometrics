## Context

The existing evaluation pipeline partitions the 20-subject corpus into
16 known (enrolled) subjects and 4 holdout subjects [2, 5, 7, 12]
inside `build_dataset_with_novelty` (datapreprocessor.py:337). The 16
known subjects are then further split by subject into train (~10),
val (3), and test (3) inside `_create_split_dataloaders`
(pipeline.py:168). The result is a model trained on only ~10 subjects,
a gallery with ~6 zero-vector centroids, a threshold calibrated on
subjects who have no centroid, and AUROC that measures two-unseen-group
separation rather than enrolled-vs-novel separation.

The 16:4 open-set change adds a new parallel evaluation path
(`run_evaluation_suite_openset`) that fixes all four failures without
touching the existing path. Key constraints: V4/V5 published numbers
must remain reproducible; `build_dataset_with_novelty` and all model
files are not modified; the open-set protocol is invoked only from the new
`experiments/v6_openset/main.py` entry point, so the V4/V5 CLIs and existing
CI jobs are unaffected.

Window statistics (confirmed from config and dataset counts):
window_size=800, step_size=400, sfreq=200 Hz, n_channels=4.
~708 windows/subject. 16 known × 708 = ~11,328 known windows total.
15% val fraction → ~106 val windows/subject → ~1,699 val windows total.
One boundary gap window per subject → 16 fewer train windows (negligible).
Train windows: ~9,629. 4 holdout × 708 = ~2,832 unknown windows.

## Decisions

### D1. Val split — temporal boundary with a one-window gap, not random

The 50% overlap between adjacent windows (step_size=400,
window_size=800) means a random train/val split places temporally
adjacent windows in both partitions, sharing 400 of 800 raw samples at
every boundary pair. A purely temporal cut (last 15% of windows by
index = last in time) reduces this to a single shared-sample pair at
the boundary per subject. That pair is eliminated by dropping
`n_gap = ceil(window_size / step_size) - 1 = 1` window immediately
before the val start:

```
train_idx = indices[:n_train - n_gap]   # ~85% minus 1 window
val_idx   = indices[n_train:]           # last 15%
```

This removes all raw-sample sharing across the split boundary at a cost
of 16 windows (one per subject, ~0.1% of training data). The
within-val overlap between consecutive val windows cannot be eliminated
without a random split (which reintroduces boundary leakage); it is
accepted as a known limitation and documented explicitly. Val P@1 must
be treated as a training-health indicator only, not a generalization
metric.

The alternative — random window-level split — eliminates within-val
overlap but reintroduces temporal leakage at every pair of
randomly-adjacent windows. Temporal boundary is preferred.

### D2. Threshold calibration on train_dl, TAR on val_dl

`compute_threshold(model, train_dl, centroids, percentile=95)` is called
with `train_dl` instead of `val_dl`. This decouples the calibration
distribution from the measurement distribution, so TAR (fraction of
val_dl windows with dist < threshold) is a genuine empirical acceptance
rate rather than a tautological constant. The 95th-percentile train
distance is slightly stricter (larger) than a val-based threshold
because train windows are closest to their own centroids; this is
conservative and appropriate for a verification system.

The original design used val_dl for both operations. Critic audit
confirmed this makes TAR a mathematical constant ≈ 0.95 regardless of
model quality.

### D3. Open-set EER via threshold sweep, distinct from closed-set EER

`evaluate_novelty_comprehensive` is extended to sweep thresholds from
`min(known_dists + unknown_dists)` to `max(known_dists + unknown_dists)`
over 1000 points, compute `FAR_t` (fraction of unknown_dists below t)
and `FRR_t` (fraction of known_dists at or above t) at each point, and
return the crossing as `'open_set_eer'`. This value is added to
`keys_nov` in `_aggregate_results`.

The existing `'eer'` field from `evaluate_comprehensive` (closed-set
pairwise cosine EER) is relabelled `'closed_set_eer'` in `keys_test`.
Reporting the closed-set EER as if it were an open-set novelty EER is
a category error that is corrected here.

### D4. Early stopping in openset mode — val loss, not P@1

When `config.eval_protocol == 'openset_16_4'`, Stage-2 early stopping
in `_train_stage2` (trainer.py:330) monitors val loss (ArcFace or
MultiSimilarity loss on val_dl) instead of val P@1. Rationale: val P@1
among enrolled subjects can spike to 0.8+ by epoch 2 due to
within-class window temporal overlap, causing early termination before
the open-set distance margin has stabilized. Val loss provides a
gradient-aligned signal that does not saturate as quickly. The patience
and delta hyperparameters remain unchanged; only the monitored quantity
differs. A config option `early_stop_metric: str = 'p1' | 'val_loss'`
with default `'p1'` (preserving existing behavior) exposes this.

### D5. Checkpoint naming — propagate eval_protocol to weight path

`_train_stage2` (trainer.py:342) names weight files using
`version_tag` and `noise_type`. When `config.eval_protocol == 'openset_16_4'`,
the name gets an `_openset16_4` suffix before `.pth`:

```python
if getattr(self.config, 'eval_protocol', 'standard') == 'openset_16_4':
    weight_path = weight_path.replace('.pth', '_openset16_4.pth')
```

Without this, an openset run overwrites the standard-run checkpoint
for the same version/noise/model/seed combination.

### D6. AUPR random baseline — always reported in output JSON

For every call to `evaluate_novelty_comprehensive`, compute and return
`aupr_random_baseline = n_unknown / (n_known + n_unknown)`. Add it to
the output JSON via `_aggregate_results`. Any AUPR claim in a paper
must be compared against this value; models where AUPR ≤ random
baseline are performing at or below chance and MUST NOT be presented as
functioning open-set detectors.

For the expected window counts (val_dl ~1,699 known, X_n_unk ~2,832
unknown), the random baseline is 2832/4531 ≈ 0.625.

### D7. CI95 — bootstrap resampling, not Student's t with df=2

`_aggregate_results` currently computes `scipy.stats.t.interval` with
df=n_seeds-1=2, which can produce CI95 bounds outside [0, 1] for
bounded metrics (confirmed in V5 output: TAR upper CI = 1.023,
AUROC upper CI = 1.101). These values must not appear in any paper
submission.

In `run_evaluation_suite_openset` (and as a correction to
`run_evaluation_suite`), replace with bootstrap CI95: draw B=1000
bootstrap resamples from the N seed results; report
`np.percentile(bootstrap_samples, [2.5, 97.5])`. At N=3 seeds the
CI is still wide; the paper must state N=3 explicitly and not use the
CI to make significance claims.

### D8. Per-holdout-subject AUROC — partition y_unk_arr inside evaluate_novelty_comprehensive

`evaluate_novelty_comprehensive` receives `X_n_unk` (raw tensor) and
already has access to `y_unk_arr` (raw subject IDs 2, 5, 7, 12) via
the existing return from `build_dataset_with_novelty`. Pass `y_unk_arr`
as an optional argument; for each unique holdout subject ID `s`, compute
subject-specific AUROC by pooling that subject's distances against all
`known_dists`. Return a dict `per_subject_auroc` keyed by subject ID.
The inter-subject standard deviation over the 4 values is an honest
estimate of holdout sensitivity; any paper claim about rejection
generalization must acknowledge this 4-subject variance.

### D9. V4 bimodal — 4-element TensorDataset arity enforced by assertion

`_create_openset_loaders_bimodal` constructs both `train_dl` and
`val_dl` as `DataLoader(TensorDataset(Xn, Xc, y, Xs), ...)`. A
one-line guard immediately after creation:
```python
assert len(next(iter(val_dl))) == 4, "bimodal val_dl must unpack as (noisy, clean, label, spec)"
```
prevents silent crashes in `_compute_threshold_bimodal` (line 276)
and `_evaluate_novelty_bimodal` (line 290), which both unpack 4-tuples.

### D10. V6 entry point + output naming — a separate experiment line, not a V5 flag

The 16:4 protocol is delivered as a dedicated **V6** line
(`experiments/v6_openset/main.py`), not a `--openset` flag on the V5/V4 CLIs.
The V6 entry point sets `config.log_file` / `config.output_file` to
`experiments/v6_openset/output_v6_openset_baselines.json` (prior-work) and
`output_v6_openset_bimodal.json` (flagship V4), then calls the shared
`run_evaluation_suite_openset`. The `_save_results_openset` envelope records:

```json
{
  "experiment": "V6: 16:4 Open-Set Evaluation (all 16 known in train)",
  "split_protocol": "train=all_16_known_temporal_85pct,val=window_level_15pct,test=4_holdout",
  "split_mode": "openset_16_4",
  ...
}
```

`scripts/ssrc_make_charts.py` hardcodes the path to
`output_v5_baselines_slim.json` and will not read the new files.
A separate `scripts/openset_make_charts.py` reads the two V6 JSONs
(`experiments/v6_openset/output_v6_openset_{baselines,bimodal}.json`) and
merges them into one "ours vs prior-work" chart.

## Data Flow

```
set_seed(seed)
        │
        ▼
build_dataset_with_novelty(clean_df, noise)
        │
        ├─── known tuple: (X_n_k [~11328×4×800], X_c_k, y_k [0..15], n_cls=16)
        └─── unknown tuple: (X_n_unk [~2832×4×800], X_c_unk, y_unk_arr [2,5,7,12])

set_seed(seed)   ← second call resets RNG before per-subject split
        │
        ▼
_create_openset_loaders(X_n_k, X_c_k, y_k, val_frac=0.15)
   for each of 16 subjects (label 0..15):
     indices = windows for this subject (sequential = temporal order)
     n_val = int(n_windows * 0.15)
     n_gap = 1
     train_idx ← indices[:n_windows - n_val - n_gap]
     val_idx   ← indices[n_windows - n_val:]
   concatenate across subjects →
        │
        ├─── train_dl: TensorDataset shuffle=True  (~9,629 windows)
        └─── val_dl:   TensorDataset shuffle=False (~1,699 windows)

        │
        ▼
create_metric_model(backbone, embed_dim=256, n_cls=16)

        │
        ▼
trainer.train(model, train_dl, val_dl, n_cls=16,
              loss_type, noise_type, model_name, seed)
   Stage 1: denoiser on all 16-subject (noisy, clean) pairs
   Stage 2: ArcFace[16×256], early_stop on val_loss
             (not val P@1) when eval_protocol == 'openset_16_4'
             weight saved to weights/..._openset16_4.pth
        │
        ▼
compute_centroids(model, train_dl, n_cls=16)
   → centroids [16×256], assertion: all norms > 1e-3
        │
        ▼
compute_threshold(model, train_dl, centroids, percentile=95)
   ← calibrated on TRAIN windows (NOT val_dl)
   → scalar threshold t*
        │
        ▼
evaluate_novelty_comprehensive(
    model, known_dl=val_dl,        ← enrolled-subject probes
    unknown_noisy=X_n_unk,         ← 4 holdout subjects, raw tensor
    centroids, threshold=t*,
    y_unk_arr=y_unk_arr            ← for per-subject AUROC
)
   known_dists  : ~1,699 min-L2 distances (small, enrolled)
   unknown_dists: ~2,832 min-L2 distances (large, novel)
   →  AUROC, AUPR, open_set_eer (sweep), TAR, TRR, FAR, FRR
      per_subject_auroc {2:…, 5:…, 7:…, 12:…}
      aupr_random_baseline = 2832/4531 ≈ 0.625
        │
        ▼
evaluate_comprehensive(model, val_dl, train_dl, n_cls=16)
   → closed_set_eer, P@1, P@5 (enrolled-class retrieval, NOT generalization)
        │
        ▼
_save_results_openset → experiments/v6_openset/output_v6_openset_{baselines,bimodal}.json
```

## Val Strategy

Within each of the 16 known subjects independently, the last 15% of
windows by index (= last in time, given sequential sliding) form the
val split; the first 85% minus one boundary-gap window form the train
split. Temporal ordering is identical to sliding-window generation
order in `_process_subject` (datapreprocessor.py:403).

Rationale for temporal over random: a random split would place
temporally adjacent windows in both partitions, sharing 400/800 = 50%
of raw samples at every boundary pair in the dataset — not just at the
cut. The temporal boundary confines raw-sample sharing to exactly one
window pair per subject (boundary pair), which the one-window gap
eliminates entirely.

The val split serves three purposes:
1. Early stopping via val loss in Stage 2 (training-health indicator).
2. Input to `evaluate_novelty_comprehensive` as the enrolled-subject
   probe set (known_dists); val subjects have real centroids so
   distances are realistic.
3. Input to `evaluate_comprehensive` for closed-set P@1/P@5 (clearly
   labelled as enrolled-class retrieval, not generalization accuracy).

Val P@1 is expected to be high (same identities as train) and MUST NOT
be reported as an open-set generalization metric. The primary paper
metrics are AUROC and open-set EER on the holdout subjects.

## Enroll/Probe Strategy for Holdout Subjects

The 4 holdout subjects [2, 5, 7, 12] are NOT enrolled. ALL of their
windows (~2,832 total) are passed as `unknown_noisy` to
`evaluate_novelty_comprehensive`. The evaluation question is open-set
rejection: does every holdout window score above the rejection
threshold derived from train_dl? No enrollment split is needed for
holdout subjects in this protocol.

If a future enrollment-style extension is required (enroll K windows of
each holdout subject, probe the rest against a 20-centroid gallery),
that is a separate protocol addable post-hoc by filtering `y_unk_arr`
inside `evaluate_novelty_comprehensive` — no architectural change is
needed. That variant is deferred to a future change.

## Metrics

| Metric | Primary | How computed |
|--------|---------|--------------|
| AUROC | Yes | `roc_auc_score` on (val_dl known_dists labeled 0, X_n_unk unknown_dists labeled 1); threshold-free, unaffected by TAR circularity |
| Open-set EER | Yes | FAR==FRR crossing over 1000-point threshold sweep; NEW in this change |
| TAR@FAR | Yes | Reported at FAR operating points 0.01 and 0.001 via `roc_curve` interpolation; added to `evaluate_novelty_comprehensive` |
| FRR / FAR at t* | Secondary | At the 95th-percentile train-dist threshold; FRR = 1-TAR is now meaningful because TAR comes from val_dl not the calibration set |
| AUPR | Secondary | `average_precision_score`; must be compared against `aupr_random_baseline` ≈ 0.625 |
| Per-subject AUROC | Secondary | One AUROC per holdout subject ID (4 values); std across these 4 is the honest holdout-sensitivity estimate |
| Closed-set P@1 / P@5 | Caveat | From `evaluate_comprehensive` on val_dl; val subjects are enrolled, label clearly as "enrolled-class retrieval upper bound" |

## Leakage Mitigations

- **Boundary raw-sample overlap**: eliminated by one-window boundary
  gap in `_create_openset_loaders` (D1).
- **TAR circularity**: broken by calibrating threshold on train_dl
  and measuring TAR on val_dl (D2).
- **Early-stop metric inflation**: switched from val P@1 to val loss
  in openset mode (D4).
- **Checkpoint collision**: eval_protocol propagated to weight-file
  name (D5).
- **Zero-vector centroids**: assertion after `compute_centroids` with
  per-class norm check (confirmed by train covering all 16 labels).
- **RNG state divergence**: `set_seed(seed)` called a second time
  immediately before `_create_openset_loaders` to reset numpy and
  torch RNG after the RNG-consuming `build_dataset_with_novelty` call.
- **CI95 out-of-bounds**: Student's t replaced with bootstrap CI95 (D7).
- **AUPR random-baseline violation**: `aupr_random_baseline` always
  included in output JSON (D6).
- **Output file collision**: `_openset16_4` suffix on all output paths (D10).

## Alternatives Considered

- **Holdout-as-val (enroll K windows of holdout subjects)**:
  Requires per-holdout centroid building and a more complex enroll/probe
  split on `X_n_unk`. Deferred; not needed for the open-set rejection
  question answered here.
- **Fixed epoch count, no early stopping**:
  Conservative and avoids inflated-P@1 bias entirely. Rejected because
  val loss early stopping is available and avoids wasted compute on
  overfit seeds. Configurable via `early_stop_metric` in Config.
- **Cross-validated holdout composition (rotating 4-subject holdout)**:
  Would provide honest holdout-sensitivity estimates but requires
  rerunning all experiments 5× and retraining from scratch for each
  composition. Deferred; recommended for the journal version. Current
  change documents this limitation explicitly in the output JSON and
  paper.
- **Larger val fraction (20%)**:
  At ~708 windows/subject, 15% gives ~106 val windows/subject
  (~1,699 total), sufficient for 95th-percentile threshold estimation
  and stable AUROC. Increasing to 20% reduces train coverage by
  ~354 windows (3.5%) with marginal threshold stability gain. 15% is
  kept as default; `openset_val_frac` in Config allows override.

## Open Questions

1. **TAR@FAR operating points**: the change adds TAR at FAR=0.01 and
   FAR=0.001. Are additional operating points required for the target
   submission venue?
2. **Per-subject EER for holdout subjects**: is a per-holdout EER table
   required, or is the aggregate open-set EER over all 4 holdout
   subjects sufficient?
3. **V4 bimodal priority**: should the V4 bimodal openset path be
   implemented and run before the V5 unimodal path, given that the
   code map identifies V4 as the highest-priority target for AUROC
   improvement?
4. **Cross-validation over holdout composition**: for the journal
   version, should at least one additional 4-subject holdout set be
   tested to demonstrate that results are not specific to subjects
   [2, 5, 7, 12]?
5. **Val fraction sensitivity**: is 15% per subject sufficient for
   stable threshold calibration, or should a sensitivity check over
   {10%, 15%, 20%} be included in the output?
6. **CI reporting strategy at N=3**: if N=3 seeds is kept, should CI95
   be reported at all in the paper, or only mean ± std with an
   explicit N=3 footnote?

## Risks / Trade-offs

- **Within-val window overlap inflates P@1 and eval_comprehensive
  metrics** → accepted and documented; val P@1 is labelled "enrolled-class
  retrieval upper bound" everywhere; primary metric is open-set AUROC
  on holdout subjects, which is threshold-free and unaffected.
- **Threshold calibrated on train_dl may be stricter than val-based
  threshold, reducing TAR** → conservative direction; appropriate for
  a verification system; TAR values will now be genuinely informative
  rather than tautological.
- **4 fixed holdout subjects: AUROC variance across holdout compositions
  is unobserved** → mitigated by reporting per-subject AUROC (4 values)
  so inter-individual variation is visible; cross-composition validation
  deferred to journal version.
- **open_set_eer sweep over 1000 points adds ~1ms per eval call** → negligible.
- **Bootstrap CI with N=3 seeds is still wide** → reported honestly as
  mean ± std (n=3); no significance claims from CI alone.
- **Propagating eval_protocol to weight-file names requires the trainer
  to read Config** → Config is already available to `TwoStageTrainer`
  via `self.config`; the change is additive.

## Migration Plan

1. Add `eval_protocol` and `openset_val_frac` fields to Config
   (datapreprocessor.py); verify all existing Config instantiations
   still work (default `'standard'`).
2. Add open-set EER sweep, per-subject AUROC, `aupr_random_baseline`,
   and `y_unk_arr` parameter to `evaluate_novelty_comprehensive`;
   add `'open_set_eer'` to `keys_nov`; relabel `'eer'` to
   `'closed_set_eer'` in `keys_test`; replace CI95 with bootstrap CI95
   in `_aggregate_results`.
3. Add `early_stop_metric` config field; branch `_train_stage2` early
   stopping on config value; add eval_protocol suffix to weight-file
   naming.
4. Implement `_create_openset_loaders` in shared pipeline; implement
   `run_evaluation_suite_openset` calling train→centroid→threshold
   (train_dl)→novelty(val_dl)→save; `_save_results_openset` writes to
   `self.config.log_file`.
5. Implement bimodal parallel (`_create_openset_loaders_bimodal`,
   `run_evaluation_suite_openset`) in V4 pipeline with 4-tuple arity guard.
6. Create the V6 entry point `experiments/v6_openset/main.py` (runs prior-work
   baselines + flagship V4 bimodal under 16:4, writing into
   `experiments/v6_openset/`). No `--openset` flag on the V4/V5 CLIs.
7. Write `scripts/openset_make_charts.py` (reads the two V6 JSONs).
8. Run V6 open-set (`python -m experiments.v6_openset.main`); verify
   AUROC > 0.625 (above random baseline) and open_set_eer is non-degenerate;
   compare flagship vs prior-work.
9. Update README.md with new section; update one .tex file with new
   open-set results table or subsection.
