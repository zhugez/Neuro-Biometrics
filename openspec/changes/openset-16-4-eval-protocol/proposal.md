## Why

The current `run_evaluation_suite` path trains on only ~10 of the 16
enrolled subjects because `_create_split_dataloaders`
(`experiments/shared/pipeline.py:168`) re-partitions the 16 known
subjects into train (~10), val (3), and test (3) sub-groups by subject.
Four correctness failures cascade from this:

1. **Under-trained gallery.** `compute_centroids` builds only ~10
   centroids for the 16-slot gallery. The 6 remaining enrolled subjects
   receive zero-vector centroids that falsely attract novel embeddings
   under L2 distance.
2. **Inflated threshold.** `compute_threshold` is calibrated on 3 val
   subjects who have no gallery centroid, so their min-distances to the
   nearest of the ~10 training centroids are systematically large,
   pushing the 95th-percentile rejection threshold into an overly
   permissive regime.
3. **Invalid AUROC.** `evaluate_novelty_comprehensive` receives
   `known_dl = test_dl` (3 known-but-unseen subjects) and
   `unknown_noisy = X_n_unk` (4 holdout subjects). Both groups are
   equally unknown to the model; AUROC therefore measures separation
   between two groups of unseen subjects, not enrolled vs novel.
4. **Degenerate TAR.** When the threshold is the 95th percentile of the
   val_dl distance distribution and TAR is the fraction of the same
   val_dl distances below that threshold, TAR ≈ 0.95 by construction
   regardless of model quality. This is a mathematical tautology, not
   an empirical acceptance rate.

The V5 output `output_v5_baselines_slim.json` confirms the pathology:
TAR spans 0.930–0.961 across all models and noise types (range too
narrow to discriminate models), AUPR for MindID_ArcFace gaussian is
0.591 (below the 0.625 random-classifier baseline for this class ratio),
and CI95 upper bounds reach 1.023 and 1.101 — physically impossible
for bounded metrics, an artifact of Student's t with df=2 on N=3 seeds.

The intended 16:4 open-set protocol is: train on **all 16** enrolled
subjects so that gallery centroids are complete, calibrate the rejection
threshold on a held-out window slice of those same 16 subjects
(not the same windows used for TAR), evaluate whether enrolled-subject
probes are accepted and whether the 4 fixed holdout subjects
[2, 5, 7, 12] are rejected, and report threshold-free metrics (AUROC,
open-set EER derived by FAR==FRR sweep) as the primary claims.

## What Changes

> **Amendment:** the 16:4 open-set protocol is delivered as its own **V6**
> experiment line (`experiments/v6_openset/`), NOT as a `--openset` flag on the
> V5/V4 entry points. The evaluation engine stays shared
> (`run_evaluation_suite_openset`); V6 is the entry point + output naming.
> Outputs: `experiments/v6_openset/output_v6_openset_baselines.json`
> (prior-work MindID/BrainNet) and `output_v6_openset_bimodal.json` (flagship
> V4). Existing V4/V5 outputs are untouched.

### New evaluation path in `experiments/shared/pipeline.py`

- Add `EEGPipeline.run_evaluation_suite_openset` (after line 127):
  outer loop identical to `run_evaluation_suite` (noise × model × seed)
  but calls `_create_openset_loaders` instead of
  `_create_split_dataloaders`. Calls `compute_threshold(train_dl, ...)` —
  not `val_dl` — to decouple calibration from TAR measurement. Passes
  `val_dl` as `known_dl` to `evaluate_novelty_comprehensive`.
- Add `EEGPipeline._create_openset_loaders` (after line 210):
  per-subject temporal window split with a one-window boundary gap
  (`n_gap = 1`) to eliminate the 50%-overlap leakage between the last
  train window and first val window. All 16 known subjects appear in
  both train and val splits (window-level, not subject-level).
- Add `EEGPipeline._save_results_openset` (after line 227): writes to
  `self.config.log_file` (the V6 entry point points this at
  `experiments/v6_openset/output_v6_openset_baselines.json`), with envelope
  fields `split_protocol` and `split_mode` so the file is self-documenting.
- New `experiments/v6_openset/main.py` entry point dispatches to
  `run_evaluation_suite_openset` for both the prior-work baselines and the V4
  bimodal flagship, writing into `experiments/v6_openset/`. No `--openset` flag
  is added to `run_cli`, so the V4/V5 CLIs and their outputs are unchanged.

### Config additions in `experiments/shared/datapreprocessor.py`

- Two new optional fields with defaults in the `Config` dataclass
  (after line 98): `eval_protocol: str = 'standard'` and
  `openset_val_frac: float = 0.15`. All existing instantiations remain
  valid; no breaking change.

### Trainer additions in `experiments/shared/trainer.py`

- Extend `evaluate_novelty_comprehensive` (line 603) to compute open-set
  EER via a threshold sweep (FAR==FRR crossing) and return it as
  `'open_set_eer'`. This is distinct from the existing closed-set pairwise
  EER in `evaluate_comprehensive` (line 548), which is relabelled
  `'closed_set_eer'` in keys_test to eliminate the category error of
  reporting it as an open-set metric.
- Add `'aupr_random_baseline'` to each result record (positive rate =
  `n_unknown / (n_known + n_unknown)`) so any AUPR claim in a paper can
  be benchmarked against the random classifier.
- Add centroid-health assertion after `compute_centroids` in
  `run_evaluation_suite_openset`: assert all 16 centroid norms exceed
  1e-3, raising a clear error if any centroid is zero-vector.
- Propagate `config.eval_protocol` into the weight-file name suffix
  inside `_train_stage2` (line 342) to prevent openset runs from
  overwriting standard-run checkpoint files.
- Replace Student's t CI95 with bootstrap CI95 (B = 1000 resamples from
  seed results) in `_aggregate_results` so reported intervals never
  exceed [0, 1] for bounded metrics. Early stopping in `_train_stage2`
  switches to val loss (ArcFace/MultiSimilarity) rather than val P@1
  when `config.eval_protocol == 'openset_16_4'`, because val P@1
  among enrolled-only subjects is artificially elevated by within-class
  window overlap and does not align with the open-set AUROC target.

### V4 bimodal parallel path in `experiments/v4_multimodal/pipeline.py`

- Add `_create_openset_loaders_bimodal` (after line 269): identical
  to the shared variant but slices `X_spec` using the same
  `train_idx` / `val_idx` arrays and constructs 4-element
  `TensorDataset(Xn, Xc, y, Xs)` to satisfy the bimodal trainer's
  4-tuple unpack contract in `_compute_threshold_bimodal` (line 276) and
  `_evaluate_novelty_bimodal` (line 290). A one-line assertion
  `assert len(next(iter(val_dl))) == 4` guards the arity requirement.
- Add `MultimodalEEGPipeline.run_evaluation_suite_openset` (after line 154),
  invoked by the V6 entry point (no `--openset` flag on the V4 `run_cli`).

### No changes to existing paths

`run_evaluation_suite`, `_create_split_dataloaders`,
`TwoStageTrainer.train`, `build_dataset_with_novelty`, and all model
files are left untouched. All published V4/V5 numbers remain
reproducible from the existing output JSONs.

## Impact

- **Affected specs**: `open-set-eval` (new capability; ADDED
  Requirements covering full-16-subject training, temporal window val
  split with boundary gap, threshold calibration on train set, open-set
  EER sweep, per-holdout-subject AUROC, AUPR random-baseline reporting,
  and checkpoint non-collision).
- **Affected code**: `experiments/shared/pipeline.py`,
  `experiments/shared/datapreprocessor.py`,
  `experiments/shared/trainer.py`,
  `experiments/shared/trainer_bimodal.py`,
  `experiments/v4_multimodal/pipeline.py`,
  and new `experiments/v6_openset/main.py`.
- **Affected docs**: `README.md` (new `### V6: 16:4 Open-Set` section
  below the existing V5 tables, hand-written or scripted from the new
  JSON); `docs/apsipa2026/main.tex` and `docs/ssrc2026/main.tex` (new
  `\subsection` or table row for open-set results; existing `tab:v5`
  rows are unchanged). `docs/apsipa2026/tables/priorwork.tex` gains one
  optional `This work (V4/V5 open-set)` row once results are final.
- **New output files**: `experiments/v6_openset/output_v6_openset_baselines.json`
  and `experiments/v6_openset/output_v6_openset_bimodal.json`. The existing
  `output_v5_baselines*.json` and `output_v4_multimodal*.json` are never
  overwritten (V6 writes to its own directory).
- **New chart script**: `scripts/openset_make_charts.py` (mirrors
  `ssrc_make_charts.py`; reads the new openset JSON path only).
  `ssrc_make_charts.py` is not modified; the SSRC slide PNGs remain
  stable.
- **Supersedes**: nothing. The standard protocol remains the primary
  published comparison path; the openset path is a parallel evaluation
  presented alongside it.
