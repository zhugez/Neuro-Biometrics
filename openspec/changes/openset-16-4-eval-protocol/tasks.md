## 0. Audit and pre-flight (~1h, no GPU)

- [x] 0.1 Confirm `config.holdout_subjects = [2, 5, 7, 12]` is the sole
      novelty-subject source: grep for `holdout_subjects` in
      `experiments/shared/datapreprocessor.py` and
      `experiments/shared/pipeline.py`; record line numbers.
- [x] 0.2 Confirm `build_dataset_with_novelty` (datapreprocessor.py:337)
      returns known tuple with labels encoded 0..15 via LabelEncoder
      fitted ONLY on the 16 known subjects; confirm holdout labels are
      raw IDs never passed through LabelEncoder.
- [x] 0.3 Confirm `evaluate_novelty_comprehensive` (trainer.py:603)
      accepts `X_n_unk` as a raw tensor (not a DataLoader) for the
      `unknown_noisy` argument; confirm batching loop at line 616.
- [x] 0.4 Confirm `_train_stage2` (trainer.py:342) weight-file name
      format: `weights/best_{version_tag}_{noise_type}_{model_name}_seed{N}.pth`;
      confirm `version_tag` derivation logic does not already include
      a protocol suffix.
- [x] 0.5 Confirm `_aggregate_results` (pipeline.py:141) currently uses
      `scipy.stats.t.interval(alpha=0.95, df=n-1, ...)` for CI95;
      record all metric keys in `keys_test` and `keys_nov`.
- [x] 0.6 Confirm V4 bimodal batch unpack in `_compute_threshold_bimodal`
      (v4_multimodal/pipeline.py:276) and `_evaluate_novelty_bimodal`
      (v4_multimodal/pipeline.py:290) requires exactly 4 elements per
      batch; record the unpack variable names.
- [x] 0.7 Verify window count arithmetic: load
      `experiments/analysis/dataset_signal_counts.json`, confirm
      ~708 windows/subject; compute expected val size at 15% and
      expected unknown size for the 4 holdout subjects.

## 1. Phase A — Config and trainer extensions (no GPU)

### 1.1 Config additions

- [x] 1.1.1 In `experiments/shared/datapreprocessor.py` after line 98
      (inside the `Config` dataclass), add:
      ```python
      eval_protocol: str = 'standard'   # 'standard' | 'openset_16_4'
      openset_val_frac: float = 0.15    # fraction of per-subject windows held for val
      early_stop_metric: str = 'p1'     # 'p1' | 'val_loss'
      ```
- [ ] 1.1.2 Run `python -c "from experiments.shared.datapreprocessor import Config; c = Config(); print(c.eval_protocol)"`;
      confirm output is `standard` and no TypeError is raised.

### 1.2 evaluate_novelty_comprehensive extensions

- [x] 1.2.1 Add optional `y_unk_arr=None` parameter to
      `evaluate_novelty_comprehensive` signature (trainer.py:603).
- [x] 1.2.2 Inside the function, after computing `unknown_dists`, add a
      sweep over 1000 threshold values from
      `min(all_dists)` to `max(all_dists)` where
      `all_dists = known_dists + unknown_dists`; compute
      `FAR_t = fraction of unknown_dists < t` and
      `FRR_t = fraction of known_dists >= t` at each t;
      find crossing `open_set_eer = t where |FAR_t - FRR_t|` is
      minimised; add `'open_set_eer'` to the returned dict.
- [x] 1.2.3 Compute and return
      `aupr_random_baseline = len(unknown_dists) / (len(known_dists) + len(unknown_dists))`.
- [x] 1.2.4 If `y_unk_arr is not None`, compute per-holdout-subject
      AUROC: for each unique ID `s` in `y_unk_arr`, pool
      `subject_unknown_dists` against all `known_dists`; return as
      `per_subject_auroc` dict keyed by raw subject ID.
- [x] 1.2.5 Add TAR at fixed FAR operating points: use
      `sklearn.metrics.roc_curve(y_true, scores)` on the binary
      problem (known=0, unknown=1) to obtain (fpr, tpr, thresholds);
      interpolate `TAR@FAR=0.01` and `TAR@FAR=0.001`; add both to
      the returned dict.

### 1.3 _aggregate_results — replace CI95 with bootstrap, add new keys

- [x] 1.3.1 In `_aggregate_results` (pipeline.py:141), replace the
      `scipy.stats.t.interval` call with a bootstrap CI95 function:
      draw B=1000 bootstrap resamples (with replacement) from the
      N seed values; return `np.percentile(samples, [2.5, 97.5])`.
      Apply this to all existing metric keys.
- [x] 1.3.2 Add `'open_set_eer'`, `'aupr_random_baseline'`,
      `'tar_at_far_0_01'`, and `'tar_at_far_0_001'` to `keys_nov`.
- [x] 1.3.3 Rename `'eer'` in `keys_test` to `'closed_set_eer'` so
      it no longer conflicts semantically with open-set EER.
      Update `_print_summary` to use `'closed_set_eer'`.
- [ ] 1.3.4 Verify the existing V5 standard-run smoke test still
      passes with the relabelled key (the standard path populates
      `'closed_set_eer'` from `evaluate_comprehensive`; the openset
      path populates `'open_set_eer'` from novelty eval). Both paths
      must produce valid JSON without KeyError.

### 1.4 _train_stage2 — checkpoint naming and early-stop branching

- [x] 1.4.1 In `_train_stage2` (trainer.py:342), after deriving
      `weight_path`, add:
      ```python
      if getattr(self.config, 'eval_protocol', 'standard') == 'openset_16_4':
          weight_path = weight_path.replace('.pth', '_openset16_4.pth')
      ```
- [x] 1.4.2 In the early-stopping block (trainer.py:330), add a branch:
      ```python
      if getattr(self.config, 'early_stop_metric', 'p1') == 'val_loss':
          current_metric = val_loss_value  # track val loss from the loss backward
          improving = current_metric < best_metric - delta
      else:
          current_metric = val_p1          # existing behavior
          improving = current_metric > best_metric + delta
      ```
      Default (`early_stop_metric='p1'`) preserves existing behavior
      exactly.

## 2. Phase B — pipeline.py additions (no GPU)

### 2.1 _create_openset_loaders

- [x] 2.1.1 Add method `EEGPipeline._create_openset_loaders(self, X_n,
      X_c, y, val_frac=None)` after line 210 (after
      `_create_split_dataloaders`). `val_frac` defaults to
      `self.config.openset_val_frac`.
- [x] 2.1.2 Implement per-subject temporal split:
      ```python
      for label in torch.unique(y).tolist():
          mask = (y == label).nonzero(as_tuple=True)[0]
          n = len(mask)
          n_val = int(n * val_frac)
          n_gap = 1  # eliminate 50%-overlap boundary pair
          train_mask.append(mask[:n - n_val - n_gap])
          val_mask.append(mask[n - n_val:])
      train_idx = torch.cat(train_mask)
      val_idx   = torch.cat(val_mask)
      ```
- [x] 2.1.3 Build `TensorDataset(X_n[train_idx], X_c[train_idx], y[train_idx])`
      → `train_dl` (shuffle=True, batch_size=config.batch_size,
      pin_memory=True, num_workers=config.num_workers).
- [x] 2.1.4 Build `TensorDataset(X_n[val_idx], X_c[val_idx], y[val_idx])`
      → `val_dl` (shuffle=False).
- [x] 2.1.5 Return `(train_dl, val_dl)`.
- [x] 2.1.6 Write a unit test with synthetic data (16 labels × 20 windows
      each) confirming: train size = 16 × (int(20*0.85) - 1),
      val size = 16 × int(20*0.15), no label appears in val but not
      in train (all 16 labels in both), no shared tensor index between
      train_idx and val_idx.

### 2.2 run_evaluation_suite_openset

- [x] 2.2.1 Add method `EEGPipeline.run_evaluation_suite_openset(self,
      n_seeds, models)` after line 127. Copy the outer triple-loop
      structure from `run_evaluation_suite` (noise × model × seed).
- [x] 2.2.2 Inside the seed loop, after `build_dataset_with_novelty`,
      call `self.set_seed(seed)` a second time to reset RNG before
      `_create_openset_loaders`.
- [x] 2.2.3 Call `_create_openset_loaders(X_n, X_c, y)` → `(train_dl, val_dl)`.
- [x] 2.2.4 Set `self.config.eval_protocol = 'openset_16_4'` and
      `self.config.early_stop_metric = 'val_loss'` before calling
      `trainer.train(...)` so Stage-2 checkpointing and early stopping
      use the openset variants.
- [x] 2.2.5 Call `trainer.train(model, train_dl, val_dl, n_cls=n_cls,
      ...)` (n_cls=16 from `build_dataset_with_novelty` return).
- [x] 2.2.6 Call `compute_centroids(model, train_dl, n_cls=n_cls)`;
      add assertion `all(torch.norm(centroids[c]) > 1e-3 for c in range(n_cls))`.
- [x] 2.2.7 Call `compute_threshold(model, train_dl, centroids,
      percentile=95)` — train_dl, NOT val_dl.
- [x] 2.2.8 Call `evaluate_novelty_comprehensive(model, known_dl=val_dl,
      unknown_noisy=X_n_unk, centroids=centroids, threshold=threshold,
      y_unk_arr=y_unk_arr)` and collect the result dict.
- [x] 2.2.9 Call `evaluate_comprehensive(model, val_dl, train_dl, n_cls)`
      for secondary metrics (closed-set P@1/P@5, SI-SNR, closed_set_eer).
- [x] 2.2.10 Append `{seed, test: comprehensive_res, novelty: novelty_res}`
      to the per-noise/model seed list; call `_aggregate_results`
      (unchanged).

### 2.3 _save_results_openset

- [x] 2.3.1 Add method `EEGPipeline._save_results_openset(self, results)`
      after line 227. Identical to `_save_results` except:
      - `experiment` key = `"16:4 Open-Set Evaluation (all 16 known in train)"`.
      - Add `split_protocol` key = `"train=all_16_known_temporal_85pct,val=window_level_15pct_temporal,test=4_holdout"`.
      - Add `split_mode` key = `"openset_16_4"`.
- [x] 2.3.2 Verify the method writes to `self.config.log_file` (which the
      V6 entry point sets to the `experiments/v6_openset/output_v6_openset_*.json`
      path before calling `run_evaluation_suite_openset`).

### 2.4 V6 entry point (experiments/v6_openset/main.py)

  NOTE (amended): the 16:4 protocol is its own **V6** experiment line, NOT a
  `--openset` flag bolted onto the V5/V4 entry points. The engine
  (`run_evaluation_suite_openset`) stays shared; V6 is the entry point + output
  naming. No `--openset` flag is added to `run_cli`.

- [x] 2.4.1 Create `experiments/v6_openset/main.py` with argparse
      (`--only {both,baselines,bimodal}`, `--seeds`, `--epochs`,
      `--batch-size`, `--num-workers`, `--optimize-h100`). Insert
      `experiments/` and `experiments/v4_multimodal/` on `sys.path` so it runs
      both as a script and via `python -m`.
- [x] 2.4.2 `run_baselines()`: build `Config` with
      `log_file=experiments/v6_openset/output_v6_openset_baselines.json`,
      then `EEGPipeline(use_mamba=False, use_denoiser=False).run_evaluation_suite_openset(models=BASELINE_MODELS)`.
- [x] 2.4.3 `run_bimodal()`: build `V4Config`, set
      `output_file=experiments/v6_openset/output_v6_openset_bimodal.json`,
      then `MultimodalEEGPipeline(use_mamba=True).run_evaluation_suite_openset()`.

## 3. Phase C — V4 bimodal parallel path (no GPU)

- [x] 3.1 Add `_create_openset_loaders_bimodal(self, X_n, X_c, y,
      X_spec, val_frac=None)` after `_create_split_dataloaders` in
      `experiments/v4_multimodal/pipeline.py` (after line 269).
      Use the same per-subject temporal split logic as the shared
      variant; additionally slice `X_spec[train_idx]` and
      `X_spec[val_idx]`. Construct both DataLoaders as
      `TensorDataset(Xn, Xc, y, Xs)` (4 elements).
- [x] 3.2 Add one-line assertion after `val_dl` creation:
      `assert len(next(iter(val_dl))) == 4`.
- [x] 3.3 Add `MultimodalEEGPipeline.run_evaluation_suite_openset(self, n_seeds)`
      after line 154 in `v4_multimodal/pipeline.py`. Mirror the
      shared `run_evaluation_suite_openset` but call
      `_create_openset_loaders_bimodal` (carrying `X_spec`) and use
      the bimodal trainer methods
      (`compute_centroids_bimodal`, `_compute_threshold_bimodal`,
      `_evaluate_novelty_bimodal`).
- [x] 3.4 The V4 bimodal open-set run is launched from the V6 entry point
      (`experiments/v6_openset/main.py run_bimodal()`), NOT a `--openset` flag
      on the V4 `run_cli`. The bimodal checkpoint also gets the
      `_openset16_4` suffix (in `trainer_bimodal._train_stage2_bimodal`).

## 4. Phase D — chart script (no GPU)

- [x] 4.1 Write `scripts/openset_make_charts.py` that reads
      `experiments/v6_openset/output_v6_openset_baselines.json` and
      `experiments/v6_openset/output_v6_openset_bimodal.json` and renders:
      - A grouped-bar chart of AUROC by model × noise type
        (mirroring `ssrc_make_charts.py` style).
      - A table row for open-set EER and TAR@FAR=0.01 per model.
      Save PNGs to `docs/ssrc2026/figures/` with distinct names
      (e.g. `openset_auroc_compare.png`).
- [x] 4.2 Do NOT modify `scripts/ssrc_make_charts.py`.

## 5. Phase E — GPU runs (V6)

- [ ] 5.1 Run V6 open-set (both families):
      `python -m experiments.v6_openset.main --seeds 3 --epochs 30`;
      confirm outputs `experiments/v6_openset/output_v6_openset_baselines.json`
      and `experiments/v6_openset/output_v6_openset_bimodal.json`.
- [ ] 5.2 Verify in each output JSON:
      - `split_mode == "openset_16_4"`.
      - `auroc_mean > 0.625` for at least one model/noise combination
        (above the random baseline).
      - `open_set_eer_mean` is between 0.0 and 0.5 (non-degenerate).
      - `tar_mean` is NOT uniformly ~0.95 across all models.
      - All `ci95` bounds are within [0, 1] for bounded metrics.
      - All 16 centroid norms > 1e-3 (no assertion fire logged).
      - Weight files for openset runs exist at
        `weights/..._openset16_4.pth`; standard-run weight files
        at `weights/best_*.pth` are unchanged (no clobbering).
- [ ] 5.3 Optionally run a single family:
      `python -m experiments.v6_openset.main --only bimodal` /
      `--only baselines`; verify same JSON contract as 5.2.

## 6. Phase F — documentation

- [ ] 6.1 Add a `### V6: 16:4 Open-Set Evaluation` section in `README.md`
      below the existing V5 summary tables; include AUROC, open-set EER,
      TAR@FAR=0.01 per model × noise; add a one-sentence note that
      val P@1 is an enrolled-class retrieval upper bound, not a
      generalization metric.
- [ ] 6.2 Add a new `\subsection{16:4 Open-Set Protocol}` or a new table
      in `docs/apsipa2026/main.tex` for open-set AUROC and EER results;
      do NOT modify existing `tab:v5` rows.
- [ ] 6.3 Mirror the same table/subsection in `docs/ssrc2026/main.tex`.
- [ ] 6.4 Optionally add a `This work (16:4 open-set)` row to
      `docs/apsipa2026/tables/priorwork.tex` via
      `experiments/analysis/build_priorwork_table.py`; add a protocol
      note distinguishing the all-16-known train regime.

## 7. Verification / Repro task

- [ ] 7.1 Confirm exact repro: delete
      `experiments/v6_openset/output_v6_openset_baselines.json`,
      re-run `python -m experiments.v6_openset.main --only baselines --seeds 3`;
      verify the new JSON is bit-identical to the one produced in step 5.1
      (same seed → same result; if not, investigate RNG state divergence
      in the second `set_seed(seed)` call before `_create_openset_loaders`).
- [ ] 7.2 Run the existing standard-run smoke: `python -m experiments.v5_baselines.main --smoke`;
      confirm SMOKE_OK and confirm `output_v5_baselines.json` is
      NOT overwritten by the openset run.
- [ ] 7.3 Run `python -m experiments.v5_baselines.main --mini-train`;
      confirm the mini-train path (which uses `_create_split_dataloaders`
      on synthetic data) is unaffected.
