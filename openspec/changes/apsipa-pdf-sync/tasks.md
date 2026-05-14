## 1. Confirm the drift before fixing

- [x] 1.1 Compute SHA-256 of `docs/apsipa2026/main.pdf` and
      `docs/APSIPA2026_NeuroBiometrics.pdf` and confirm they currently
      differ
- [x] 1.2 Confirm both files are 331{,}808 bytes and extract to
      identical text (sanity check: this is artifact drift, not
      content drift)

## 2. Rebuild canonical and sync

- [x] 2.1 Run `latexmk -pdf -interaction=nonstopmode -halt-on-error
      main.tex` in `docs/apsipa2026/` to make sure
      `docs/apsipa2026/main.pdf` is the freshest possible canonical
      output
- [x] 2.2 Verify the producing log shows `Output written on main.pdf
      (5 pages, ...)`, 0 overfull, 0 undefined refs/citations, 0
      errors
- [x] 2.3 Copy `docs/apsipa2026/main.pdf` over
      `docs/APSIPA2026_NeuroBiometrics.pdf` (canonical-to-shipped
      direction only)

## 3. Verify byte-equality and envelope

- [x] 3.1 Recompute SHA-256 of both files and confirm they are equal
- [x] 3.2 Confirm the shipped PDF still reports 5 pages, A4, fonts
      embedded/subset, not encrypted
- [x] 3.3 Run `openspec status --change apsipa-pdf-sync` and confirm
      `isComplete: true`
