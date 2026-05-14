## Context

`docs/apsipa2026/main.tex` builds to `docs/apsipa2026/main.pdf` via
`latexmk`. The shipped / submission copy lives at
`docs/APSIPA2026_NeuroBiometrics.pdf`. After the
`apsipa-claim-polish` change, both PDFs are 331{,}808 bytes and extract
to identical text, but their SHA-256 hashes differ:

```
docs/APSIPA2026_NeuroBiometrics.pdf  92f0f7101a89c8fc...
docs/apsipa2026/main.pdf             f90b92c6b18b4be9...
```

The most likely cause is that `cp main.pdf -> APSIPA2026_NeuroBiometrics.pdf`
was performed at one point in the build sequence, after which `latexmk`
re-ran and produced a fresh `main.pdf` (different `/CreationDate` /
`/ID` PDF metadata, same visible text). The submission checklist treats
the two PDFs as "the same artifact"; byte-equality is the cheapest way
to enforce that.

## Goals / Non-Goals

**Goals:**
- After this change, `sha256(docs/APSIPA2026_NeuroBiometrics.pdf) ==
  sha256(docs/apsipa2026/main.pdf)` is true.
- The shipped PDF still satisfies the APSIPA envelope (5 pages, A4,
  fonts embedded/subset, no encryption, 0 overfull / 0 undefined refs
  in the producing log).
- The fix is a single, documented step that any future rebuild can
  repeat without thinking.

**Non-Goals:**
- No new build automation, no Makefile target, no `latexmk` post-hook.
- No `.tex` / `.bib` / figure / code / spec changes.
- Not addressing the kit-level `\ProvidesClass{APSIPA2025}` warning
  (lives in the official template, not in our source).
- Not promoting the shipped PDF to be the canonical artifact — the
  latexmk output remains canonical.

## Decisions

### D1: Direction of the sync — main.pdf is canonical

**Chosen:** Treat `docs/apsipa2026/main.pdf` as the canonical artifact
and overwrite `docs/APSIPA2026_NeuroBiometrics.pdf` with its bytes.

**Why:** `main.pdf` is the deterministic output of the documented build
(`latexmk` on `main.tex`). Anyone with the repo can reproduce it; the
shipped copy is a downstream artifact. Always prefer "rebuild
canonical, then mirror".

**Alternative considered:** Make the shipped PDF canonical and write a
note to never rebuild without re-syncing. Rejected — that inverts the
provenance (the producing log lives next to `main.pdf`, not next to the
shipped copy) and is hostile to anyone who runs `latexmk` later.

### D2: Sync mechanism — direct file copy

**Chosen:** `cp docs/apsipa2026/main.pdf docs/APSIPA2026_NeuroBiometrics.pdf`
(or the platform-equivalent), executed once after the final
`latexmk` build.

**Why:** Smallest possible change. No new tooling, no script, no
post-build hook to maintain. The build sequence becomes "rebuild → sync
→ verify SHA-256", which is one extra command.

**Alternative considered:** Add a `latexmk` post-build hook (`-pretex` /
`-aux-directory` tricks or a `Makefile` target) that always copies on
success. Rejected for this change — adding automation is out of scope
for the sync, and a future change can wire it up if the manual step
becomes a recurring problem.

### D3: Verification — SHA-256 equality after sync

**Chosen:** After the copy, compute SHA-256 of both files and confirm
they match. Also confirm the shipped PDF still has 5 pages, A4, fonts
embedded/subset, and is not encrypted.

**Why:** Byte-equality is the only check that catches the failure mode
the reviewer flagged. The envelope check protects against a corrupted
copy (e.g., partial write, wrong source file).

**Alternative considered:** Compare extracted text only. Rejected —
that is exactly the check that already passed before this change and
that the reviewer flagged as insufficient.

## Risks / Trade-offs

- Risk: Future `latexmk` rebuild silently breaks the byte-equality.
  Mitigation: tasks.md adds a "post-rebuild sync" reminder; the
  verification step in the same task list catches the drift before
  submission.

- Risk: Wrong source file copied (e.g., a stale `main.pdf` from a
  different worktree).
  Mitigation: verify SHA-256 *and* envelope (page count, fonts) on the
  shipped PDF after the sync.

- Trade-off: No automation means the next rebuild requires the same
  manual sync. Acceptable for a one-shot submission; the alternative
  (post-build hook) is out of scope for this change.
