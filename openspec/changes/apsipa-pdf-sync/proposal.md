## Why

A read-only review of the APSIPA 2026 build artifacts confirmed the paper
content and format are submission-ready (5 pages, A4, fonts embedded/subset,
0 overfull, 0 undefined refs/citations, 0 errors), but flagged a single
repository hygiene mismatch: `docs/APSIPA2026_NeuroBiometrics.pdf` (the
shipped/submit copy) and `docs/apsipa2026/main.pdf` (the latexmk build
output) are **text-equal but not byte-equal** — their SHA-256 hashes
differ even though both are 331{,}808 bytes and extract to identical text.
The checklist for this paper requires the two PDFs to match byte-for-byte
so reviewers and the submission portal see the exact same artifact the
build produced. Without a sync, the only risk is artifact-hygiene drift,
not content; the paper itself is unaffected.

## What Changes

- Resync `docs/APSIPA2026_NeuroBiometrics.pdf` so it is a byte-for-byte
  copy of `docs/apsipa2026/main.pdf` after the most recent successful
  `latexmk` build (the canonical PDF is the latexmk output).
- Verify post-sync that the SHA-256 of the two files match and that the
  shipped PDF still satisfies the format envelope (5 pages, A4, fonts
  embedded/subset, no encryption).
- Document the sync step so future rebuilds do not re-introduce the drift
  (single-line entry in tasks.md / build notes; no new automation).

## Capabilities

### New Capabilities
- `apsipa-build-artifacts`: Discipline for the relationship between the
  latexmk build output (`docs/apsipa2026/main.pdf`) and the shipped /
  submission PDF (`docs/APSIPA2026_NeuroBiometrics.pdf`) for the APSIPA
  2026 manuscript.

### Modified Capabilities
<!-- none -->

## Impact

- File copy only: `docs/APSIPA2026_NeuroBiometrics.pdf` is overwritten
  with the current `docs/apsipa2026/main.pdf` contents.
- No `.tex`, no `.bib`, no figures, no code, no specs, no results.
- Build envelope unchanged: still 5 pages, A4, fonts embedded/subset, no
  overfull, no undefined refs.
- Submission risk after sync: zero — both PDFs are byte-identical and
  carry the exact text the latexmk build produced.
