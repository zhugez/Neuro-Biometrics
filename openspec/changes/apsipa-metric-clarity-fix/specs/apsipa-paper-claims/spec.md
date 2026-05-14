## ADDED Requirements

### Requirement: Metrics paragraph SHALL define P@5 as precision-at-5
The Metrics paragraph SHALL define P@5 unambiguously as a precision-style metric. The paragraph MUST state that P@5 is the fraction of the top-5 retrieved gallery epochs (excluding self) that share the query identity, averaged across queries, and that this implies P@5 ≤ P@1 by construction. The paragraph MUST NOT leave P@5 named without a definition that distinguishes it from a top-5 hit rate (CMC@5).

#### Scenario: P@5 is unambiguous on first reading
- **WHEN** a reader reads the Metrics paragraph of the Experimental Setup section
- **THEN** the paragraph states that P@5 is precision-at-K, gives the formula in words (fraction of top-5 retrieved that share the query identity, averaged over queries), and notes the P@5 ≤ P@1 relationship

#### Scenario: P@5 is not confused with CMC@5
- **WHEN** a reader compares P@5 to a CMC@5 (top-5 hit rate) interpretation
- **THEN** the paper's definition rules out the hit-rate interpretation, so the reader does not suspect a labeling bug when P@5 < P@1

### Requirement: Evaluation-Protocol paragraph SHALL disclose verification scoring
The Evaluation-Protocol paragraph SHALL disclose how verification scores are computed in enough detail that a reader can re-derive the procedure without reading the source code. The paragraph MUST name (a) prototype construction (per-subject mean of training-set embeddings), (b) distance metric for held-out-subject scoring (Euclidean), (c) the operating-threshold rule (95th percentile of validation min-distances), and (d) the source of the AUROC and EER values printed in Table~II (closed-set pairwise cosine similarities over all test-pool embedding pairs).

#### Scenario: Reader can re-derive the held-out-subject protocol
- **WHEN** a reader reads the Evaluation-Protocol paragraph
- **THEN** the paragraph names prototype = per-subject mean of training embeddings, distance = Euclidean, threshold = 95th percentile of validation min-distances, scored on held-out-subject epochs

#### Scenario: Reader can identify the source of Table~II's AUROC and EER
- **WHEN** a reader looks at the AUROC and EER columns of Table~II
- **THEN** the manuscript text makes clear that these values are computed from pairwise cosine similarities over the test pool (closed-set pairwise verification, session/epoch-disjoint), not from held-out-subject scoring

#### Scenario: Verification gap claim is consistent with disclosed scoring
- **WHEN** a reader reads the Discussion verification-gap paragraph
- **THEN** the paragraph attributes the near-chance AUROC explicitly to held-out-subject scoring, and does not equate Table~II's pairwise AUROC with held-out-subject AUROC

### Requirement: Limitations paragraph SHALL hedge external baselines and synthetic noise
The Limitations paragraph SHALL include a conservative statement that the paper does not report external EEG-biometric baselines under the same protocol and that all noise is injected synthetically rather than captured from the environment. The statement MUST appear in the same paragraph as the existing dataset-size and verification-objective hedges so the reader sees the full hedge envelope at once.

#### Scenario: External baseline limitation is acknowledged
- **WHEN** a reader reads the Limitations paragraph
- **THEN** the paragraph states explicitly that no strong external baselines are reported under the same protocol and that comparisons in the paper are within-paper ablations

#### Scenario: Synthetic noise limitation is acknowledged
- **WHEN** a reader reads the Limitations paragraph
- **THEN** the paragraph states explicitly that the noise families used for evaluation are injected synthetically rather than captured from environmental recordings
