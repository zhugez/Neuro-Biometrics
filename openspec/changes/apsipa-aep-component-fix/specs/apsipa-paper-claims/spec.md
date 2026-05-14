## ADDED Requirements

### Requirement: AEP-explanation sentence SHALL NOT misclassify cortical components as short-latency
The Dataset section's AEP-explanation sentence SHALL NOT describe cortical AEP components as "short-latency". Specifically, the sentence MUST NOT pair the phrase "the cortical response" with the phrase "short-latency components" in the same sentence, because cortical AER components are long-latency (50--300\,ms+) by definition.

#### Scenario: AEP sentence is internally consistent on latency class
- **WHEN** a reader reads the AEP-explanation sentence in the Dataset
  section
- **THEN** the sentence does not assert that cortical AEP components
  are short-latency, and any latency-class adjective applied to those
  components is consistent with the standard AER taxonomy

#### Scenario: AEP sentence avoids the contradiction with "cortical response"
- **WHEN** the AEP-explanation sentence introduces AEPs as "the
  cortical response to a controlled acoustic stimulus"
- **THEN** no later clause in the same sentence calls those same
  components "short-latency"

### Requirement: AEP-explanation sentence SHALL NOT name $P_{300}$ as a paradigmatic component
The AEP-explanation sentence SHALL NOT name $P_{300}$ as an example component of the AEP signal used in this paper. The reasoning is that $P_{300}$ is a late cognitive ERP that is reliably elicited only by an oddball paradigm, while the AEP-EEG-Biometric dataset (alzahab2024aep) used in this paper records passive auditory stimulation, so $P_{300}$ is not a paradigmatic feature of the corpus.

#### Scenario: AEP sentence does not promise $P_{300}$
- **WHEN** a reader reads the AEP-explanation sentence
- **THEN** the sentence does not give $P_{300}$ as an example
  component, either alone or in a list with other peaks

#### Scenario: AEP sentence does not promise components the paper never analyses
- **WHEN** a reader reads the AEP-explanation sentence and then the
  rest of the manuscript
- **THEN** any specific ERP peak named in the AEP sentence is also
  used somewhere in the experimental analysis; otherwise no specific
  peak is named

### Requirement: AEP-explanation sentence SHALL preserve identity-signal intent
The AEP-explanation sentence SHALL retain its purpose of explaining *why* AEPs carry subject-specific identity information, in language accessible to a reviewer who is not an EEG specialist. The sentence MUST attribute the identity signal to individual differences of cortical anatomy and auditory processing pathways, and MUST describe AEPs as an "internal, replay-resistant biometric cue" or equivalent framing.

#### Scenario: Identity-signal rationale is still present
- **WHEN** a reader reads the rewritten AEP-explanation sentence
- **THEN** the sentence still attributes subject-specific structure to
  individual differences of cortical anatomy and auditory processing
  pathways, and still frames AEPs as an internal biometric cue
  resistant to replay attacks

#### Scenario: Sentence remains accessible to non-EEG reviewers
- **WHEN** a non-EEG reviewer reads the rewritten sentence
- **THEN** they understand why the AEP signal carries identity
  information without needing prior knowledge of specific ERP peak
  names or the AER latency taxonomy
