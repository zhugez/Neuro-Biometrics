## ADDED Requirements

### Requirement: The open-set evaluation path SHALL train on all 16 enrolled subjects

When `eval_protocol == 'openset_16_4'`, the training DataLoader SHALL
include windows from all 16 known subjects (labels 0..15 as encoded by
`build_dataset_with_novelty`). No subject SHALL be excluded from the
training set. The ArcFace weight matrix SHALL be initialized with
`num_classes = 16` and all 16 rows SHALL receive gradient updates during
Stage-2 training.

#### Scenario: All 16 enrolled subjects are present in train_dl
- **WHEN** `_create_openset_loaders` constructs `train_dl` from the
  known-subject arrays returned by `build_dataset_with_novelty`
- **THEN** `torch.unique(y_train)` returns a tensor of length 16 with
  values 0 through 15, and `len(train_dl.dataset)` is approximately
  `round(16 * n_windows_per_subject * 0.85) - 16`

#### Scenario: Gallery centroids cover all 16 enrolled identities
- **WHEN** `compute_centroids(model, train_dl, n_cls=16)` is called
  after training
- **THEN** `centroids.shape[0] == 16` and
  `all(torch.norm(centroids[c]) > 1e-3 for c in range(16))` holds
  (no zero-vector centroid for any enrolled identity)

### Requirement: The val split SHALL use a per-subject temporal boundary with a one-window gap

`_create_openset_loaders` SHALL partition each subject's windows by
temporal order (window index ascending, equal to generation order from
`_process_subject`). The last `int(n_windows * val_frac)` windows SHALL
form the val split. The window immediately before the val boundary SHALL
be excluded from train to eliminate the 50%-overlap raw-sample sharing
between the last train window and first val window (step_size=400,
window_size=800). A random shuffle of windows within the val or train
splits is NOT performed; temporal order is preserved within each split.

#### Scenario: Temporal boundary eliminates cross-split raw-sample overlap
- **WHEN** `_create_openset_loaders` runs with default `val_frac=0.15`
  on a subject with `n` windows
- **THEN** `max(train_idx) < min(val_idx)` (temporal cut is strict),
  `min(val_idx) - max(train_idx) >= 2` (gap of at least one window index),
  and no index appears in both `train_idx` and `val_idx`

#### Scenario: All 16 enrolled subjects appear in val_dl
- **WHEN** `_create_openset_loaders` builds `val_dl`
- **THEN** `torch.unique(y_val)` returns length 16 with values 0..15
  (val split is window-level, not subject-level; every subject contributes
  approximately `int(n_windows_per_subject * val_frac)` val windows)

### Requirement: The rejection threshold SHALL be calibrated on train_dl, not val_dl

`run_evaluation_suite_openset` SHALL call
`compute_threshold(model, train_dl, centroids, percentile=95)` — passing
`train_dl` as the calibration loader — so that the calibration distance
distribution is independent of the val_dl windows on which TAR is
subsequently measured.

#### Scenario: TAR is not a tautological constant
- **WHEN** the rejection threshold `t*` is computed from `train_dl` and
  TAR is computed from `val_dl` via `evaluate_novelty_comprehensive`
- **THEN** TAR is not guaranteed to equal 0.95 and MUST vary across
  models with genuinely different embedding geometries (a random-weight
  model SHALL produce TAR and AUROC that differ from a trained model)

#### Scenario: Threshold calibration loader and TAR measurement loader are distinct
- **WHEN** `run_evaluation_suite_openset` calls `compute_threshold`
  then calls `evaluate_novelty_comprehensive`
- **THEN** the DataLoader passed to `compute_threshold` is `train_dl`
  and the `known_dl` argument to `evaluate_novelty_comprehensive` is
  `val_dl`; these are not the same object

### Requirement: evaluate_novelty_comprehensive SHALL return open-set EER derived by FAR==FRR sweep

`evaluate_novelty_comprehensive` SHALL compute the open-set EER as the
threshold value at which
`FAR_t = fraction(unknown_dists < t)` equals
`FRR_t = fraction(known_dists >= t)`,
found by sweeping 1000 evenly-spaced thresholds between
`min(all_dists)` and `max(all_dists)` and minimising `|FAR_t - FRR_t|`.
The result SHALL be returned as `'open_set_eer'` in the metric dict.
This metric SHALL be included in `keys_nov` in `_aggregate_results` and
SHALL be reported as a primary open-set metric. It is distinct from the
existing closed-set pairwise EER from `evaluate_comprehensive`, which
SHALL be labelled `'closed_set_eer'` in `keys_test`.

#### Scenario: open_set_eer is between 0 and 0.5 for a functioning detector
- **WHEN** `evaluate_novelty_comprehensive` is called with known_dists
  from enrolled-subject val windows and unknown_dists from holdout windows
- **THEN** `result['open_set_eer']` is a float in [0.0, 0.5] for a model
  whose AUROC > 0.5; a random-weight model MAY return values near 0.5

#### Scenario: open_set_eer and closed_set_eer appear in the aggregated result JSON
- **WHEN** `_aggregate_results` processes per-seed dicts
- **THEN** the output dict contains both `'open_set_eer_mean'` and
  `'closed_set_eer_mean'` as distinct numeric fields, and neither
  field name is `'eer_mean'`

### Requirement: evaluate_novelty_comprehensive SHALL report TAR at fixed FAR operating points

The function SHALL compute and return TAR at FAR=0.01 and FAR=0.001 as
`'tar_at_far_0_01'` and `'tar_at_far_0_001'`, derived by interpolating
`sklearn.metrics.roc_curve` on the binary label assignment
(known_dists labeled 0, unknown_dists labeled 1).

#### Scenario: TAR@FAR=0.01 is reported and comparable across models
- **WHEN** a reader compares two models' open-set results in the output JSON
- **THEN** both results contain `tar_at_far_0_01_mean` with values
  that differ across models with different embedding quality, and
  neither value is constrained to be ≈ 0.95 by construction

### Requirement: The output JSON SHALL include aupr_random_baseline

Every call to `evaluate_novelty_comprehensive` in openset mode SHALL
compute `aupr_random_baseline = n_unknown / (n_known + n_unknown)` and
include it in the returned metric dict. `_aggregate_results` SHALL
propagate this value (it is constant across seeds for fixed class sizes,
so mean == each seed value). Any AUPR reported in the output JSON MUST
be interpretable against this baseline; models with
`aupr_mean <= aupr_random_baseline` MUST NOT be represented as
functioning open-set detectors in any paper or README table.

#### Scenario: aupr_random_baseline appears in the output JSON
- **WHEN** `_save_results_openset` writes the result file
- **THEN** each model/noise record contains `'aupr_random_baseline'`
  (a float in (0, 1), approximately 0.625 for the expected window counts)

#### Scenario: AUPR is compared against the random baseline in documentation
- **WHEN** a model result shows `aupr_mean <= aupr_random_baseline`
- **THEN** the README and any paper table for that row include a note
  that the model performs at or below random on AUPR

### Requirement: The openset run SHALL NOT overwrite standard-run weight files or output JSONs

When `eval_protocol == 'openset_16_4'`, the weight file path SHALL include
the suffix `_openset16_4` before `.pth` (set in `_train_stage2`).
The open-set runs SHALL be launched from the dedicated V6 entry point
(`experiments/v6_openset/main.py`), which SHALL write its results under
`experiments/v6_openset/` (`output_v6_openset_baselines.json` and
`output_v6_openset_bimodal.json`). Standard-run V4/V5 artifacts SHALL remain
unmodified after an open-set run.

#### Scenario: Standard-run weight files are preserved after an openset run
- **WHEN** `run_evaluation_suite_openset` completes for the same
  version, noise type, model name, and seed as a prior standard run
- **THEN** the weight file `weights/best_{tag}_{noise}_{model}_seed{N}.pth`
  still exists and its modification timestamp predates the openset run;
  the openset weight is at
  `weights/best_{tag}_{noise}_{model}_seed{N}_openset16_4.pth`

#### Scenario: Standard-run output JSON is preserved after a V6 open-set run
- **WHEN** `python -m experiments.v6_openset.main` completes
- **THEN** `output_v5_baselines.json`, `output_v5_baselines_slim.json` and
  `output_v4_multimodal.json` have the same content as before the run; the new
  files `experiments/v6_openset/output_v6_openset_baselines.json` and
  `experiments/v6_openset/output_v6_openset_bimodal.json` have
  `split_mode == "openset_16_4"`

### Requirement: CI95 in aggregated results SHALL be computed by bootstrap, never by Student's t with df <= 2

`_aggregate_results` SHALL replace `scipy.stats.t.interval` with a
bootstrap CI95 computed by drawing B >= 1000 resamples with replacement
from the N seed values and taking the 2.5th and 97.5th percentiles.
No CI95 bound for a metric bounded to [0, 1] SHALL exceed 1.0 or fall
below 0.0 in any output JSON.

#### Scenario: CI95 bounds are within the valid range for bounded metrics
- **WHEN** any `*_ci95` field is written to the output JSON
- **THEN** both the lower and upper bounds are in [0.0, 1.0] for metrics
  whose domain is [0, 1] (AUROC, AUPR, TAR, FAR, FRR, TRR, P@1, P@5)

### Requirement: The V4 bimodal openset path SHALL construct 4-element TensorDatasets

`_create_openset_loaders_bimodal` SHALL produce both `train_dl` and
`val_dl` as `DataLoader` over `TensorDataset(X_noisy, X_clean, y, X_spec)`
(4 elements per batch), matching the 4-tuple unpack contract of
`_compute_threshold_bimodal` and `_evaluate_novelty_bimodal`. A runtime
assertion `assert len(next(iter(val_dl))) == 4` SHALL appear immediately
after `val_dl` construction.

#### Scenario: bimodal val_dl unpacks without error in threshold computation
- **WHEN** `_compute_threshold_bimodal` iterates over `val_dl`
- **THEN** each batch unpacks cleanly as `(noisy, clean, labels, spectrograms)`
  with no `ValueError: not enough values to unpack`

### Requirement: The output JSON envelope SHALL self-document the open-set split protocol

`_save_results_openset` SHALL include the keys `split_protocol` and
`split_mode` at the top level of the JSON envelope. The value of
`split_mode` SHALL be `"openset_16_4"`. The value of `split_protocol`
SHALL describe the train, val, and test assignment in human-readable
form (e.g. `"train=all_16_known_temporal_85pct,val=window_level_15pct_temporal,test=4_holdout"`).

#### Scenario: A reader can distinguish openset and standard result files
- **WHEN** a reader opens any `experiments/v6_openset/output_v6_openset_*.json` file
- **THEN** the file contains `"split_mode": "openset_16_4"` at the top
  level and DOES NOT contain `"split_mode": "standard"` or any field
  that would be ambiguous with the standard 10/3/3 subject split
