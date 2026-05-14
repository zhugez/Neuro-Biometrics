## ADDED Requirements

### Requirement: Abstract V4 win sentence SHALL frame SI-SNR as Stage I descriptive
The abstract sentence reporting the V4 model's win condition SHALL state the
P@1 retrieval result as the win condition and SHALL present the corresponding
SI-SNR number as a Stage~I descriptive value, not as part of the V4 win
condition. The sentence MUST NOT use the phrasing "highest scale-invariant
SNR" as a co-equal V4 result alongside Precision@1.

#### Scenario: V4 abstract sentence couples P@1 win with associated Stage I SI-SNR
- **WHEN** a reader reads the abstract sentence that names the V4 model's
  retrieval win
- **THEN** the sentence states that V4 produces the best Precision@1 on every
  noise type and presents the SI-SNR number as the *associated* or
  *corresponding* Stage~I quality, in line with the Ablation paragraph's
  disclaimer that SI-SNR deltas are not attributed to Stage~II additions

#### Scenario: Abstract still anchors a single SI-SNR number for the skim reader
- **WHEN** a reader sees only the abstract and not the Ablation paragraph
- **THEN** the abstract still names one SI-SNR figure so the skim reader
  has a signal-quality anchor, but framed as Stage~I descriptive context
  rather than as a V4 win

### Requirement: Detailed V4 Results SHALL avoid unanchored SI-SNR "gain" wording
The Detailed V4 Results paragraph SHALL describe the denoiser's power-line
SI-SNR using absolute phrasing that matches what Table~II actually reports.
The paragraph MUST NOT use the phrase "large SI-SNR gains" while Table~II
reports only denoised SI-SNR without an unprocessed baseline column.

#### Scenario: Power-line SI-SNR sentence uses absolute "highest" phrasing
- **WHEN** a reader reads the sentence in the Detailed V4 Results paragraph
  that comments on power-line SI-SNR
- **THEN** the sentence reports that the denoiser *reaches its highest
  SI-SNR in the power-line condition*, or equivalent absolute wording, and
  does not assert a "gain" against an absent baseline

#### Scenario: Table~II remains unchanged
- **WHEN** the wording is updated
- **THEN** Table~II is not modified (no new baseline column, no new rows),
  and the prose stays anchored to the SI-SNR values the table actually
  reports

### Requirement: Related Work novelty claim SHALL match the paper title
The Related Work State-Space-Models paragraph's novelty sentence SHALL
describe the paper's contribution using the same task framing as the title:
identification with an open-set verification analysis. The sentence MUST NOT
state that the paper's contribution is "for EEG biometric verification"
when the title is *Multimodal Mamba Fusion for Noise-Robust EEG Biometric
Identification*.

#### Scenario: Novelty claim uses identification framing
- **WHEN** a reader reads the Related Work novelty sentence
- **THEN** the contribution is described as being for EEG biometric
  *identification with open-set verification analysis*, not "for EEG
  biometric verification"

#### Scenario: Title, abstract, and novelty claim agree on task framing
- **WHEN** a reader compares the title, the abstract's verification-gap
  acknowledgement, and the Related Work novelty sentence
- **THEN** all three present the same primary task (identification) and the
  same secondary framing (open-set verification reported as a gap analysis)
