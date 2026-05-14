## Context

The APSIPA 2026 manuscript currently passes format and most content
audits, but a focused reviewer pass identified two under-specified
descriptions and one under-hedged limitation:

### Audit 1: P@5 definition

The Metrics paragraph (`main.tex`, near line 341--344) reads:

```tex
For closed-set retrieval we report Precision@1 (P@1) and Precision@5
(P@5). For signal quality we report SI-SNR in decibels. For open-set
verification we report AUROC and equal-error rate (EER) measured on
the held-out subject pool.
```

The reference implementation
(`experiments/shared/trainer.py:543--563`,
`_calculate_retrieval_metrics`):

```python
sim = embs @ embs.T
sim.fill_diagonal_(-1e9)  # exclude self-match
_, indices = sim.sort(dim=1, descending=True)
match_matrix = (lbls[indices] == lbls.unsqueeze(1))
cmc = [match_matrix[:, :k].any(dim=1).float().mean()
       for k in range(1, k_max+1)]
p1 = cmc[0]                                  # CMC@1 == P@1
p5 = match_matrix[:, :5].float().mean()      # mean over n*5 cells
```

Two consequences:

1. P@5 is **true precision@5** (mean of all n×5 binary matches), not a
   top-5 hit rate (`match_matrix[:, :5].any(dim=1).float().mean()`,
   which would equal CMC@5).
2. P@5 ≤ P@1 by construction whenever positions 2..5 are not
   guaranteed to also be same-identity. With n_test queries and k≈5
   gallery items per identity per noise type, P@5 numbers slightly
   below P@1 (e.g. .855 vs .877 for power-line R34+Arc) are *not* a
   bug — they are the expected precision-at-K decay.

The paper does not state this. A reviewer reading the table without
the source code may flag it.

### Audit 2: Verification protocol and AUROC/EER source

The Evaluation-Protocol paragraph (`main.tex`, near line 297--309)
currently says:

```tex
For open-set verification (AUROC, EER), the held-out subjects
S_{holdout} = {2,5,7,12} are removed from training entirely and
reused only at evaluation as "unknown" identities scored against
enrolled prototypes (held-out-subject open-set verification).
```

The Discussion verification-gap paragraph (`main.tex`, near
line 481--491) extends this:

```tex
Across all V4 configurations, the AUROC of the known-vs-unknown
distance score remains close to chance (≈ 0.50), and EER stays in
the 0.35--0.41 range (Table~II).
```

The reference implementation actually has **two separate scorings**:

| Scoring | Code path | Distance | Pairs | Reported in |
| ------- | --------- | -------- | ----- | ----------- |
| Closed-set pairwise verification | `evaluate_bimodal` | cosine, `embs @ embs.T` | all test-pool pairs | Table~II AUROC, EER |
| Open-set novelty (held-out) | `_evaluate_novelty_bimodal` | Euclidean to centroids | known vs held-out | not in Table~II |

The Table~II AUROC/EER values are therefore **closed-set pairwise
verification on the test pool**, where queries come from subjects the
embedder already saw at training time but from disjoint sessions or
epochs. They are NOT scored against held-out subjects. The
held-out-subject open-set scoring lives in `novelty_res` and is
described in the AISEI~2026 line of work as TAR/TRR/FAR/FRR plus its
own AUROC.

Additionally, the protocol prose does not name:

- Prototype construction
  (`compute_centroids_bimodal:441--456`: per-subject mean of
  training-set embeddings).
- Distance metric for novelty (Euclidean / `torch.cdist(p=2)`).
- Operating-threshold rule (95th percentile of validation
  min-distances; `_compute_threshold_bimodal:270--279`).

A reviewer cannot re-derive the protocol from the manuscript alone.

### Audit 3: Limitations hedge

Limitations (`main.tex`, near line 500--505) currently list:

- small dataset (20 subjects, 4 channels)
- partial cross-day session variability
- V3 single-seed quick run
- AUROC ≈ 0.5 ⇒ verification under-optimized

What is missing, conservative, and reviewer-easy:

- No external EEG-biometric baselines are reported under the same
  protocol; numbers are within-paper ablation comparisons.
- Noise is synthetically injected; the paper does not test on
  recorded environmental artifacts.

## Goals / Non-Goals

**Goals:**
- Make P@5 unambiguous in one sentence.
- Make the verification protocol re-derivable in 2--3 sentences:
  prototype, distance, threshold, and which scoring populates Table~II.
- Add one conservative limitations sentence about external baselines
  and synthetic noise.
- Preserve the 5-page envelope and 0-overfull profile.
- Touch only `main.tex`; rebuild and resync PDFs.

**Non-Goals:**
- No new experiments, tables, figures, citations, or numbers.
- No retitling, no abstract changes, no protocol changes.
- Not re-opening the identification-framing fix
  (`apsipa-claim-polish`), the AEP wording fix
  (`apsipa-aep-component-fix`), or the byte-equality invariant
  (`apsipa-pdf-sync` / `apsipa-build-artifacts`).
- Not adding an EER table or held-out-subject AUROC table — both are
  legitimately out of scope for this prose pass.
- Not addressing the kit-level
  `\ProvidesClass{APSIPA2025}` warning (lives in the official
  template).

## Decisions

### D1: P@5 disambiguation in Metrics paragraph

**Chosen:** Insert one sentence after the existing P@1/P@5 mention
that defines P@5 as the precision-style metric the implementation
computes:

```tex
P@5 is computed as a precision-at-K value: the fraction of the top-5
retrieved gallery epochs (excluding self) that share the query
identity, averaged across queries; consequently P@5 ≤ P@1 by
construction.
```

**Why:** Smallest edit that closes the labeling-bug suspicion. Does
not require altering Table~II, the captions, or the result columns.

**Alternatives considered:**

- **Rename to "P-prec@5" / "Prec@5"**: rejected — non-standard,
  introduces a new symbol the rest of the paper does not use.
- **Replace P@5 with CMC@5 (top-5 hit rate)**: rejected — would
  require recomputing every P@5 entry in Table~II and the V4 results
  artifact; out of scope for a prose pass.
- **Drop P@5 entirely**: rejected — losing the secondary metric for
  the sake of brevity weakens the retrieval evidence.

### D2: Verification protocol disclosure

**Chosen:** Replace the trailing clause "scored against enrolled
prototypes (held-out-subject open-set verification)" with a sentence
group naming prototype, distance, threshold, and the source of
Table~II's AUROC/EER. Concretely:

```tex
For open-set verification on held-out subjects, we form a per-subject
prototype as the mean of training-set fused embeddings, score each
held-out epoch by its minimum Euclidean distance to any prototype,
and choose the operating threshold as the 95th percentile of the
analogous min-distance scores on the validation split. The AUROC and
EER values reported in Table~II are computed separately, from
pairwise cosine similarities over all test-pool embedding pairs (same
subjects as in training but session- and epoch-disjoint), and
therefore characterize closed-set pairwise verification rather than
held-out-subject scoring; the held-out-subject AUROC remains close to
chance (≈ 0.50), confirming the verification gap discussed in
Section~V.
```

**Why:** Lets a reviewer reproduce both protocol heads from the
manuscript alone, and removes the "Table~II = held-out-subject"
ambiguity that the current Discussion paragraph implicitly creates.

**Alternatives considered:**

- **One-line addition only**: rejected — would name prototype but not
  fix the AUROC-source ambiguity, leaving the Discussion paragraph
  hanging.
- **Promote held-out-subject AUROC into Table~II**: rejected — adds
  experimental scope and a new column; out of scope for this prose
  pass.
- **Remove the "verification gap" claim entirely**: rejected —
  removes substantive scientific content the rest of the paper relies
  on; the gap is a real, honest finding.

### D3: Discussion paragraph alignment with D2

**Chosen:** In the "Verification gap" paragraph, replace the unqualified
"AUROC of the known-vs-unknown distance score" phrasing with one that
distinguishes the closed-set pairwise AUROC (Table~II) from the
held-out-subject AUROC. One sentence change.

**Why:** Avoids contradicting the disclosure added by D2.

### D4: Limitations hedge

**Chosen:** Append one sentence to the Limitations block:

```tex
We also do not benchmark against external EEG-biometric baselines
under the same protocol; reported comparisons are within-paper
ablations, and all noise is injected synthetically rather than
captured from the environment.
```

**Why:** Pre-empts the most common reviewer complaint without adding
content scope.

### D5: Rebuild and resync the shipped PDF in the same change

**Chosen:** After the prose edits, run `latexmk` and copy
`docs/apsipa2026/main.pdf` over
`docs/APSIPA2026_NeuroBiometrics.pdf` so the two PDFs remain
byte-for-byte equal.

**Why:** The `apsipa-build-artifacts` capability requires byte-for-byte
equality at every change boundary.

## Risks / Trade-offs

- **Risk:** D2's longer sentence reflows the Evaluation-Protocol
  paragraph and pushes a table float to a different page or pushes
  the paper past 5 pages.
  **Mitigation:** Rebuild and inspect; if reflow occurs, trim wording
  elsewhere (D2 has one redundant clause, "same subjects as in
  training but session- and epoch-disjoint", which can be shortened
  to "session/epoch-disjoint test pool"). Page count must stay 5.

- **Risk:** A reviewer who wanted a numeric held-out-subject AUROC
  table reads D2 and expects one.
  **Mitigation:** D2 explicitly names the held-out-subject AUROC as
  "close to chance (≈ 0.50)" so the reader is not left wondering;
  promoting it to its own table is correctly out of scope.

- **Risk:** D1's "P@5 ≤ P@1 by construction" is technically
  conditional on every query having at least one in-class neighbor
  beyond rank 1. With ~k≥5 same-identity items per query in this
  dataset, the inequality holds in practice. If it ever did not hold
  for some query, the *averaged* P@5 could in principle exceed P@1,
  but that is not what we observe. The phrase is defensible at the
  level of expectation under the dataset.
  **Mitigation:** Wording uses "by construction" rather than
  "always", which is the standard precision-vs-CMC-at-K relation in
  the retrieval literature.

- **Risk:** Editing introduces a typo, dropped LaTeX brace, or
  duplicated phrase.
  **Mitigation:** Diff the rebuilt PDF against the prior 5-page
  output visually; verify SHA-256 equality of the two PDFs after the
  resync.
