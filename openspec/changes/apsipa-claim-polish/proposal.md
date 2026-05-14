## Why

An independent explore-mode review of the APSIPA 2026 manuscript flagged
three residual wording risks that, while non-blocking, can be tightened
to keep the claim stack consistent with the paper's "identification +
verification gap" framing. The review explicitly concluded the paper is
submittable as-is; this change applies the optional 5-minute polish so
the final submission carries no minor "could-be-bitten" wording.

## What Changes

- Soften the abstract sentence claiming the V4 model has the
  "highest scale-invariant SNR" so the stronger SI-SNR is presented as a
  Stage~I descriptive value rather than as part of the V4 win condition.
- Replace "large SI-SNR gains" in the Detailed V4 Results paragraph with
  a phrasing that does not require an unprocessed baseline in-table,
  since the table only reports denoised SI-SNR.
- Reword the Related Work novelty claim from "for EEG biometric
  verification" to "for EEG biometric identification with open-set
  verification analysis" to match the (already corrected) paper title.

## Capabilities

### New Capabilities
- `apsipa-paper-claims`: Discipline for the textual claims that connect
  the title, abstract, and per-section narrative of the APSIPA 2026
  manuscript to the empirical results actually reported in its tables
  and figures.

### Modified Capabilities
<!-- none -->

## Impact

- Single-file textual edit: `docs/apsipa2026/main.tex`
  (Abstract sentence near line 109, Detailed V4 Results paragraph near
  line 380, Related Work State-space-models paragraph near line 154).
- Rebuild artifacts: `docs/apsipa2026/main.pdf` and the shipped
  `docs/APSIPA2026_NeuroBiometrics.pdf` are regenerated.
- No code, no data, no figures, no bib changes; no impact on results.
- Build envelope unchanged: 5 pages, A4, fonts embedded/subset, no
  overfull, no undefined refs.
