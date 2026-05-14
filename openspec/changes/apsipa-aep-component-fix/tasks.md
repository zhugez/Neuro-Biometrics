## 1. Locate the AEP-explanation sentence

- [x] 1.1 Open `docs/apsipa2026/main.tex` and confirm the Dataset
      section sentence near lines 288--292 still reads
      "short-latency components (e.g., $P_{300}$/$N_{100}$ peaks)"

## 2. Apply the wording edit

- [x] 2.1 Replace the offending clause so the sentence reads
      "their latency- and amplitude-domain morphology can carry
      subject-specific structure that originates in individual
      differences of cortical anatomy and auditory processing
      pathways, which makes them attractive as an internal,
      replay-resistant biometric cue." (peak names dropped, latency
      class dropped, identity-signal intent preserved)

## 3. Rebuild canonical and resync shipped PDF

- [x] 3.1 Run `latexmk -pdf -interaction=nonstopmode -halt-on-error
      main.tex` in `docs/apsipa2026/`
- [x] 3.2 Verify the producing log shows `Output written on main.pdf
      (5 pages, ...)`, 0 overfull, 0 undefined refs/citations, 0
      errors
- [x] 3.3 Copy `docs/apsipa2026/main.pdf` over
      `docs/APSIPA2026_NeuroBiometrics.pdf` (canonical-to-shipped
      direction only, per `apsipa-build-artifacts`)
- [x] 3.4 Recompute SHA-256 of both PDFs and confirm they are equal

## 4. Verify envelope and close out

- [x] 4.1 Confirm the shipped PDF still reports 5 pages, A4, fonts
      embedded/subset, not encrypted
- [x] 4.2 Visually spot-check the Dataset paragraph in the rebuilt
      PDF for typos, dropped braces, or duplicated phrases
- [x] 4.3 Run `openspec status --change apsipa-aep-component-fix` and
      confirm `isComplete: true`
