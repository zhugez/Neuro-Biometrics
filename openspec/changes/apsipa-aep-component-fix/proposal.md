## Why

A read-only review of the APSIPA 2026 manuscript flagged a single
remaining scientific-wording risk: the AEP explanation paragraph
(Dataset section) currently calls $P_{300}$ and $N_{100}$
"short-latency components" of the *cortical* auditory evoked response.
This is internally inconsistent — cortical AEP components are by
definition long-latency (50--300\,ms+), and $P_{300}$ in particular is
a late cognitive ERP (~300\,ms) typically elicited by an oddball
paradigm, which the AEP-EEG-Biometric dataset (alzahab2024aep) does
*not* use. An EEG-literate reviewer is likely to bite. The fix is a
one-sentence conservative rewrite that drops the specific peak names
and the wrong latency class.

## What Changes

- Replace the AEP explanation sentence in
  `docs/apsipa2026/main.tex` (Dataset section, near line 288--292) so
  it no longer claims $P_{300}$/$N_{100}$ are "short-latency
  components", and no longer names specific ERP peaks at all. The
  replacement keeps the same intent (explaining why AEP carries
  subject-specific identity signal) but stays at the latency-/
  amplitude-domain level.
- Rebuild `docs/apsipa2026/main.pdf` and resync
  `docs/APSIPA2026_NeuroBiometrics.pdf` so it remains byte-for-byte
  equal to the latexmk output (per `apsipa-build-artifacts`).

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `apsipa-paper-claims`: extend the discipline established by
  `apsipa-claim-polish` to cover one additional class of textual claim
  — neuroscience descriptions of the AEP signal, which MUST stay
  consistent with the dataset paradigm and with the standard AER
  latency taxonomy.

## Impact

- Single-file textual edit: `docs/apsipa2026/main.tex` (Dataset
  section AEP-explanation sentence near line 288--292).
- Rebuild artifacts: `docs/apsipa2026/main.pdf` and the shipped
  `docs/APSIPA2026_NeuroBiometrics.pdf` are regenerated and re-synced
  byte-for-byte.
- No code, no data, no figures, no bib changes; no impact on results.
- Build envelope unchanged: 5 pages, A4, fonts embedded/subset, no
  overfull, no undefined refs.
