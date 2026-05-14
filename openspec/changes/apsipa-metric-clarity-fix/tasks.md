## 1. Audit metric and protocol implementation

- [x] 1.1 Read `experiments/shared/trainer.py` `_calculate_retrieval_metrics`
      (around lines 543--563) and confirm `p5 = match_matrix[:, :5].float().mean()`
      is the fraction of top-5 cells that match (i.e. precision@5),
      not `any(dim=1).mean()` (which would be CMC@5)
- [x] 1.2 Read `experiments/shared/trainer_bimodal.py` `evaluate_bimodal`
      (around lines 378--439) and confirm AUROC/EER come from pairwise
      cosine similarity over test-pool embeddings (`embs @ embs.T`)
- [x] 1.3 Read `experiments/v4_multimodal/pipeline.py`
      `_compute_threshold_bimodal` and `_evaluate_novelty_bimodal`
      (around lines 270--320) and confirm the held-out-subject scoring
      uses Euclidean min-distance to centroids with a val-95th-percentile
      threshold

## 2. Edit Metrics paragraph (D1)

- [x] 2.1 Open `docs/apsipa2026/main.tex` and locate the Metrics
      paragraph (near line 341--344)
- [x] 2.2 Insert one sentence after the existing P@1/P@5 mention
      defining P@5 as the precision-at-K value: fraction of top-5
      retrieved gallery epochs (excluding self) that share the query
      identity, averaged across queries; consequently P@5 ≤ P@1 by
      construction

## 3. Edit Evaluation-Protocol paragraph (D2)

- [x] 3.1 Locate the open-set-verification clause in the
      Evaluation-Protocol paragraph (near line 302--309)
- [x] 3.2 Replace the trailing "scored against enrolled prototypes
      (held-out-subject open-set verification)" wording with the
      sentence group from `design.md` D2 that names: prototype = mean
      of training-set fused embeddings per subject, distance =
      Euclidean min-distance, threshold = 95th percentile of validation
      min-distances, and explicitly attributes Table~II AUROC/EER to
      pairwise cosine similarity on the test pool (closed-set pairwise,
      session/epoch-disjoint), keeping the held-out-subject AUROC
      separate at ≈ 0.50

## 4. Edit Discussion verification-gap paragraph (D3)

- [x] 4.1 Locate the "Verification gap" paragraph in
      Section~V Discussion (near line 481--491)
- [x] 4.2 Replace the "AUROC of the known-vs-unknown distance score"
      phrasing so the paragraph attributes the near-chance AUROC
      explicitly to the held-out-subject scoring described in D2, and
      does not equate Table~II's pairwise AUROC with the held-out
      AUROC

## 5. Edit Limitations paragraph (D4)

- [x] 5.1 Locate the Limitations paragraph (near line 500--505)
- [x] 5.2 Append one sentence stating that no external EEG-biometric
      baselines under the same protocol are reported (within-paper
      ablations only) and that noise is injected synthetically rather
      than captured from the environment

## 6. Rebuild canonical and resync shipped PDF

- [x] 6.1 Run `latexmk -pdf -interaction=nonstopmode -halt-on-error
      main.tex` in `docs/apsipa2026/`
- [x] 6.2 Verify the producing log shows `Output written on main.pdf
      (5 pages, ...)`, 0 overfull, 0 undefined refs/citations, 0
      errors
- [x] 6.3 If page count exceeds 5, trim wording per the D2 mitigation
      (collapse "session- and epoch-disjoint" or similar redundancies)
      and rebuild
- [x] 6.4 Copy `docs/apsipa2026/main.pdf` over
      `docs/APSIPA2026_NeuroBiometrics.pdf` (canonical-to-shipped
      direction only, per `apsipa-build-artifacts`)
- [x] 6.5 Recompute SHA-256 of both PDFs and confirm they are equal

## 7. Verify envelope and close out

- [x] 7.1 Confirm the shipped PDF still reports 5 pages, A4, fonts
      embedded/subset, not encrypted
- [x] 7.2 Visually spot-check the Metrics paragraph, Evaluation-Protocol
      paragraph, Discussion verification-gap paragraph, and Limitations
      paragraph in the rebuilt PDF for typos, dropped braces, or
      duplicated phrases
- [x] 7.3 Run `openspec status --change apsipa-metric-clarity-fix` and
      confirm `isComplete: true`
