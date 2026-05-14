## ADDED Requirements

### Requirement: Shipped PDF SHALL be byte-for-byte equal to the latexmk output
The shipped copy at `docs/APSIPA2026_NeuroBiometrics.pdf` SHALL be byte-for-byte equal to the latexmk output at `docs/apsipa2026/main.pdf` before the manuscript is locked for submission. The latexmk output is canonical; the shipped copy is a downstream mirror, and the two files MUST have the same SHA-256 hash.

#### Scenario: Post-build sync produces byte-identical PDFs
- **WHEN** `latexmk` finishes producing `docs/apsipa2026/main.pdf` and
  the shipped PDF is then refreshed
- **THEN** `sha256(docs/APSIPA2026_NeuroBiometrics.pdf)` equals
  `sha256(docs/apsipa2026/main.pdf)`, and a byte-level comparison of
  the two files reports no difference

#### Scenario: Stale shipped PDF is detected before submission
- **WHEN** the SHA-256 hashes of the two files differ even though their
  extracted text matches
- **THEN** the shipped PDF is treated as out-of-date and is resynced
  from the latexmk output before the manuscript is locked

### Requirement: Shipped PDF SHALL preserve the APSIPA submission envelope
The shipped PDF SHALL satisfy the APSIPA 2026 submission envelope independently of the byte-equality check: 5 pages, A4, all fonts embedded and subset, no encryption, and the producing latexmk log MUST report 0 overfull boxes, 0 undefined references, and 0 errors.

#### Scenario: Envelope verified on the shipped PDF
- **WHEN** the shipped PDF is inspected after a sync
- **THEN** it has exactly 5 pages, A4 paper size, every font is
  embedded and subset (Type 1, no Type 3), and the file is not
  encrypted

#### Scenario: Producing log is clean
- **WHEN** the latexmk build that produced the canonical
  `docs/apsipa2026/main.pdf` is inspected
- **THEN** the log reports `Output written on main.pdf (5 pages, ...)`,
  zero overfull `\hbox` warnings, zero undefined citations or
  references, and zero errors (cosmetic underfull warnings are
  permitted)

### Requirement: Sync direction SHALL flow from latexmk output to shipped PDF
Resyncing SHALL overwrite `docs/APSIPA2026_NeuroBiometrics.pdf` with the contents of `docs/apsipa2026/main.pdf`; the reverse direction MUST NOT be used because it would desynchronise the build artifact from its producing log.

#### Scenario: Sync overwrites only the shipped PDF
- **WHEN** the sync step is executed
- **THEN** only `docs/APSIPA2026_NeuroBiometrics.pdf` is modified;
  `docs/apsipa2026/main.pdf` is read-only with respect to the sync

#### Scenario: Reverse-direction sync is rejected
- **WHEN** a workflow attempts to overwrite `docs/apsipa2026/main.pdf`
  with the contents of `docs/APSIPA2026_NeuroBiometrics.pdf`
- **THEN** the workflow is treated as incorrect and the build is
  re-run from `main.tex` instead, restoring the canonical direction
