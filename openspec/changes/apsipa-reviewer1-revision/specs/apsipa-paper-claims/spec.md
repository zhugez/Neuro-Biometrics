## ADDED Requirements

### Requirement: Abstract SHALL fit within a 100-word envelope and lead with headline numbers

The Abstract SHALL contain at most 100 words and SHALL lead with the
single-sentence problem statement followed by the headline P@1 and
SI-SNR numbers and the verification-gap caveat. The Abstract MUST NOT
enumerate the (i)/(ii)/(iii) contribution list at full length;
instead it SHALL summarise them in one clause.

#### Scenario: Abstract length stays within envelope
- **WHEN** the Abstract is rendered in the compiled PDF
- **THEN** the word count is ≤ 100 and the rendered height is ≤ 8 lines

#### Scenario: Abstract lists the P@1 and SI-SNR headline numbers
- **WHEN** a reader reads the Abstract
- **THEN** the Abstract contains P@1 numbers for all three noise
  families and the SI-SNR triple, and a one-sentence statement that
  held-out-subject AUROC ≈ 0.50 motivates calibration

### Requirement: Introduction SHALL state input/output formally before listing contributions

The Introduction SHALL open with a paragraph that states the EEG
biometric task formally (input shape, output shape, the closed-set
and open-set evaluation regimes) before any contribution is listed.
Each subsequent contribution paragraph MUST be motivated by an
observation on the prior pipeline rather than presented as an
architecture list. The Introduction MUST also include a brief
V1--V4 introduction giving one line per version so the reader knows
what V1--V4 mean before the Method.

#### Scenario: Reader knows the input and output before reading Method
- **WHEN** a reader reads the first paragraph of the Introduction
- **THEN** the paragraph names the input tensor shape, the output
  embedding shape, and both the closed-set retrieval and open-set
  verification tasks

#### Scenario: V1--V4 are introduced before being referenced
- **WHEN** a reader reaches the end of the Introduction
- **THEN** the reader has read a one-line description of each of
  V1, V2, V3, V4 and knows that V3 is a single-seed quick run

#### Scenario: Each contribution is motivation-first
- **WHEN** a reader reads a contribution paragraph in the Introduction
- **THEN** the paragraph names the limitation it addresses before
  naming the architectural change

### Requirement: Method SHALL gloss every introduced symbol on first use

The Method section SHALL gloss every symbol introduced by an equation
in prose either immediately before or immediately after the equation
that uses it. The Method section SHALL tie the unit-norm property
of the per-branch embeddings to a named L2-normalisation layer,
rather than leaving the property implicit.

#### Scenario: Reader can map every symbol in the gate equation to a glossary
- **WHEN** a reader reads the gate equation g = σ(W_g[e_eeg; e_spec])
- **THEN** σ is defined as elementwise sigmoid and W_g is named as
  a learned linear projection, in the same paragraph

#### Scenario: SI-SNR loss has a one-sentence purpose statement
- **WHEN** a reader reads the SI-SNR equation in Method
- **THEN** the paragraph contains a one-sentence statement that
  scale-invariant reconstruction quality is needed because the
  denoiser output may rescale the target arbitrarily

#### Scenario: Unit-norm vectors are produced by a named layer
- **WHEN** a reader sees the term "unit-norm vectors" in Method
- **THEN** the paragraph names the L2-normalisation layer that
  produces them and references its position in Fig. 2

### Requirement: Method SHALL use `gated self-attention fusion` as the operator name

The fusion module SHALL be named `gated self-attention fusion`
(or `gated self-attention`) everywhere in narrative text, abstract,
captions, contributions, and conclusion. The term `cross-attention`
MAY appear at most once in Method §III.B as a parenthetical synonym
referring to the prior work's terminology, with an explicit citation
to the prior work, for citation continuity. Table row labels for the
ablation MUST use `+ Spec + Fusion`, not `+ Spec + CrossAttn`.

#### Scenario: Fusion module is named consistently in narrative
- **WHEN** a reader scans the manuscript for the fusion operator name
- **THEN** every mention outside the Method §III.B parenthetical uses
  `gated self-attention` or `gated self-attention fusion`

#### Scenario: Ablation table label matches operator name
- **WHEN** a reader reads the ablation table row label
- **THEN** the label reads `+ Spec + Fusion` and matches the Method
  description

### Requirement: Hyperparameters SHALL be reported in Experimental Setup, not Method

The values of λ_e, λ_s, m, and the seed list {0, 1, 2} SHALL be
reported in the Experimental Setup section, not embedded as numbers
in the Method section. The Method section MAY introduce these
symbols but MUST defer the numerical choice to Experimental Setup
with a forward reference to the hyperparameter ablation.

#### Scenario: Method introduces symbols without committing values
- **WHEN** a reader reads the Method paragraph that introduces
  λ_e and λ_s
- **THEN** the paragraph names the symbols but does not pin them to
  a specific value

#### Scenario: Experimental Setup pins hyperparameter values
- **WHEN** a reader reads Experimental Setup
- **THEN** λ_e = 0.3, λ_s = 0.2, m = 4, and seeds {0, 1, 2} are
  reported with a reference to the hyperparameter ablation table

### Requirement: Cross-version P@1 deltas SHALL be reported with bootstrap CIs

The cross-version and ablation tables SHALL report each headline
P@1 delta (V4 minus V1) with a paired bootstrap 95% confidence
interval computed over the existing per-seed embeddings. Cells whose
confidence interval excludes zero MAY be described as
"significantly above V1 (paired bootstrap, 95% CI excludes zero)";
cells whose confidence interval straddles zero MUST be described
as "magnitude observation within the across-seed std" rather than as
a tested claim.

#### Scenario: V4 - V1 delta is reported with a CI
- **WHEN** a reader reads the ablation table caption
- **THEN** each $\Delta$ V4-V1 row is annotated with a 95% paired
  bootstrap CI on the per-noise P@1 delta

#### Scenario: Insignificant deltas are explicitly hedged
- **WHEN** a $\Delta$ V4-V1 row's CI straddles zero
- **THEN** the prose description in Discussion uses
  "magnitude observation" language rather than asserting significance

### Requirement: Hyperparameter ablations SHALL be reported for λ and m

The paper SHALL report at least a 3 × 3 sensitivity grid for
(λ_e, λ_s) and at least a 4-point sweep for m, both in the form of a
small table or figure. The reported grids MAY be single-seed
(sensitivity rather than significance) and MAY be limited to one
strongest-configuration cell (R34+ArcFace + power-line). The chosen
operating values (λ_e = 0.3, λ_s = 0.2, m = 4) MUST appear on the
reported grids and the prose MUST acknowledge whether they coincide
with the grid optimum.

#### Scenario: λ-grid is included as a small heatmap or table
- **WHEN** a reader reads the hyperparameter section
- **THEN** the section includes a 3 × 3 P@1 grid over
  (λ_e, λ_s) and the chosen point (0.3, 0.2) is visible

#### Scenario: m-sweep is included as a small line plot or table
- **WHEN** a reader reads the hyperparameter section
- **THEN** the section includes P@1 over m ∈ {2, 4, 8, 16} and the
  chosen point m = 4 is visible

### Requirement: Component-wise V4 ablation SHALL be reported

The paper SHALL report a component-wise ablation of V4 with at least
the following knockouts: (a) V4 minus gate (concat-only fusion);
(b) V4 minus self-attention (skip MHA); (c) V4 minus auxiliary
losses (λ_e = λ_s = 0); (d) V4 minus Mamba spectrogram sweeps
(CNN-only spec branch). Each knockout SHALL be reported as
mean±std over three seeds on the strongest configuration
(R34+ArcFace), for at least the three noise families.

#### Scenario: Each named component has a corresponding ablation row
- **WHEN** a reader reads the V4 component ablation table
- **THEN** the table has rows for each of: full V4, V4 - gate,
  V4 - self-attn, V4 - aux, V4 - mamba_spec

#### Scenario: Component ablation reports 3-seed statistics
- **WHEN** a reader reads any V4 component ablation row
- **THEN** the row reports mean and std (or 95% CI) over three seeds

### Requirement: Mechanistic Discussion claims SHALL be backed by figures

The Discussion section SHALL either back any mechanistic claim with
a figure referenced in the same paragraph, or rephrase the claim as
a hedged "is consistent with" or "suggests" statement. The Discussion
section MUST NOT contain unhedged mechanistic claims without a
corresponding figure. The mechanistic claims in scope include 50 Hz
line localisation, EMG high-frequency tilt, the receptive-field
effect of the Mamba block, and the gate selecting one branch by
noise family.

#### Scenario: 50 Hz line removal claim is backed by a spectrogram
- **WHEN** the Discussion mentions the 50 Hz line being removed by
  the spectrogram branch
- **THEN** the same paragraph references a spectrogram figure
  (Fig. 3) showing the line before and after denoising

#### Scenario: Gate selects-branch-by-noise claim is backed by a heatmap
- **WHEN** the Discussion mentions the gate downweighting one branch
  for one noise family
- **THEN** the same paragraph references a gate-vector heatmap
  (Fig. 4) computed per noise family

### Requirement: Prior-work comparison table SHALL be included

The paper SHALL include a prior-work comparison table (Table III or
IV) with at least three cited numbers from prior EEG biometric
papers (e.g. MindID, BrainNet, ChronoNet, EEGNet). Cited numbers
MUST be footnoted with a one-line statement of the protocol
difference (clean vs noise-injected, closed- vs open-set, dataset
identity) and MUST NOT be used to claim a "win" against the
proposed V4. The table caption MUST explicitly state that no
external baseline was re-implemented under the same noise protocol
in this paper, and that the comparison is for context only.

#### Scenario: At least three cited rows appear in the table
- **WHEN** a reader reads the prior-work comparison table
- **THEN** the table contains at least three rows of cited numbers
  from prior EEG biometric papers, each with the original paper
  reference

#### Scenario: Cited numbers are footnoted with protocol caveats
- **WHEN** a reader reads a row whose numbers are cited from a prior
  paper
- **THEN** the row has a footnote stating the protocol difference
  (e.g. "clean signal, closed-set, included for context only")

#### Scenario: Caption discloses absence of same-protocol re-implementation
- **WHEN** a reader reads the comparison table caption
- **THEN** the caption states explicitly that no external baseline
  was re-implemented under the same noise protocol, and that the
  comparison is provided for context only

### Requirement: Limitations SHALL pair each item with a future-work direction

Each Limitation listed in the Limitations paragraph SHALL be paired
with a one-clause future-work direction in the same paragraph. The
paragraph MUST cover at least: small dataset, synthetic noise,
verification AUROC near chance, and absence of cross-dataset
transfer evaluation.

#### Scenario: Each limitation has a paired future direction
- **WHEN** a reader reads any Limitation in the Limitations paragraph
- **THEN** the same sentence (or the immediately following clause)
  names a concrete future-work direction addressing it

#### Scenario: All four named limitations are present
- **WHEN** a reader reads the Limitations paragraph
- **THEN** the paragraph mentions: dataset size, synthetic noise,
  AUROC near chance, and cross-dataset transfer
