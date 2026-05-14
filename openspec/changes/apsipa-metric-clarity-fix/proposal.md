## Why

A read-only metric-and-protocol audit of the APSIPA 2026 manuscript
(`docs/apsipa2026/main.tex`) cross-checked against the V4 implementation
(`experiments/shared/trainer.py`,
`experiments/shared/trainer_bimodal.py`,
`experiments/v4_multimodal/pipeline.py`) surfaced two reviewer-bite risks
that are not number errors but prose-level under-specifications.

1. **P@5 is not defined.** The Metrics section names "Precision@1
   (P@1) and Precision@5 (P@5)" but never disambiguates P@5 from a
   top-5 hit rate (CMC@5). The implementation
   (`trainer.py:_calculate_retrieval_metrics`,
   `match_matrix[:, :5].float().mean()`) is the *true precision@5*: it
   averages the binary match matrix over the top-5 columns, i.e. the
   fraction of all top-5 retrieved epochs that share the query identity.
   This makes P@5 ≤ P@1 by construction. A reviewer reading P@5 numbers
   slightly below P@1 may suspect a labeling bug; the prose fix is one
   sentence.

2. **Verification protocol is under-specified, and the AUROC source is
   ambiguous.** The Evaluation-Protocol paragraph mentions "scored
   against enrolled prototypes" but does not say (a) how prototypes are
   formed, (b) which distance is used, (c) how the operating threshold
   is chosen, or (d) which AUROC/EER is in Table~II. The implementation
   actually runs two distinct scorings:
   - *Closed-set pairwise verification* (`evaluate_bimodal`, cosine
     similarity over all test-pool embedding pairs) — these are the
     AUROC/EER values printed in Table~II.
   - *Open-set held-out-subject novelty scoring*
     (`_evaluate_novelty_bimodal`, Euclidean min-distance to per-subject
     centroids, threshold = 95th percentile of validation min-distances)
     — these AUROC/AUPR values are not in Table~II.

   The Discussion currently frames the Table~II AUROC as "known-vs-unknown
   distance score", which conflates the two. Prose fix is 1--3 sentences
   in the Evaluation-Protocol and Metrics paragraphs.

3. **Limitations omit a conservative hedge.** The current Limitations
   block lists dataset size, channel count, and synthetic-noise injection
   but does not explicitly state that no strong external baselines under
   the same protocol are reported. One conservative sentence covers it.

## What Changes

- Add a single clarifying sentence in the Metrics paragraph of
  `docs/apsipa2026/main.tex` defining P@5 as the fraction of top-5
  retrieved epochs that share the query identity, and noting that
  P@5 ≤ P@1 by construction.
- Replace the under-specified verification clause in the
  Evaluation-Protocol paragraph with a 2--3 sentence block that names
  prototype construction (per-subject mean of training-set embeddings),
  distance metric (Euclidean for held-out novelty, cosine for closed-set
  pairwise), threshold rule (95th percentile of validation min-distances),
  and which scoring populates Table~II.
- Adjust the Discussion "Verification gap" paragraph to attribute the
  Table~II AUROC/EER to closed-set pairwise verification on the test
  pool, and to keep the held-out-subject novelty AUROC framing
  separately and clearly.
- Append one conservative sentence to Limitations stating that the paper
  does not include external baselines under the same protocol and that
  noise is injected synthetically.
- Rebuild `docs/apsipa2026/main.pdf` and resync
  `docs/APSIPA2026_NeuroBiometrics.pdf` so the two PDFs remain
  byte-for-byte equal (per `apsipa-build-artifacts`).

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `apsipa-paper-claims`: extend the claim-discipline established by
  `apsipa-claim-polish` and `apsipa-aep-component-fix` to cover
  metric-definition and protocol-definition prose. The Metrics and
  Evaluation-Protocol paragraphs MUST disclose enough detail that a
  reader can re-derive the metric formulas and the verification scoring
  procedure without reading the source code.

## Impact

- Single-file textual edits: `docs/apsipa2026/main.tex` (Metrics
  paragraph, Evaluation-Protocol paragraph, Discussion verification-gap
  paragraph, Limitations paragraph). Three to five sentences total,
  no new figures, no new tables, no new bib entries.
- No code changes, no number changes, no protocol changes. The audit
  confirmed the implementation already matches the *intent* the paper
  describes; the fix is prose-level.
- Rebuild artifacts: `docs/apsipa2026/main.pdf` and the shipped
  `docs/APSIPA2026_NeuroBiometrics.pdf` are regenerated and re-synced
  byte-for-byte.
- Build envelope unchanged target: 5 pages, A4, fonts embedded/subset,
  no overfull, no undefined refs. If a clarifying sentence pushes the
  paper past 5 pages, the change MUST trim wording elsewhere rather
  than accept a longer page count.
