## Context

`docs/apsipa2026/main.tex` (Dataset section, lines 288--292) currently
reads:

```tex
Auditory evoked potentials are the cortical response to a controlled
acoustic stimulus; their short-latency components (e.g., $P_{300}$/$N_{100}$
peaks) carry subject-specific morphology that originates in individual
differences of cortical anatomy and auditory processing pathways, which
makes them attractive as an internal, replay-resistant biometric cue.
```

There are two issues in one sentence:

1. **Latency-class mismatch.** The standard auditory evoked response
   (AER) latency taxonomy is:

   | Latency       | Generator        | Components               |
   | ------------- | ---------------- | ------------------------ |
   | 1--10 ms      | brainstem (ABR)  | wave I--V                |
   | 10--50 ms     | thalamic / early | Pa, Nb (MLR)             |
   | 50--300 ms    | cortex (LLR)     | N100, P200, N200         |
   | 250--500 ms+  | cortex (cognitive) | P300, N400             |

   $N_{100}$ is *long-latency cortical*, not short-latency.
   $P_{300}$ is *late cognitive*, also not short-latency. Calling
   "the cortical response" *and* "short-latency components" in the same
   sentence is self-contradicting.

2. **Paradigm mismatch.** $P_{300}$ is reliably elicited by an oddball
   paradigm; the AEP-EEG-Biometric dataset (alzahab2024aep) uses
   passive listening, so $P_{300}$ is not even a paradigmatic feature
   of this corpus. Naming it as the example component therefore
   doubles the reviewer-bite risk.

## Goals / Non-Goals

**Goals:**
- Remove the latency-class error and the paradigm-mismatch by dropping
  specific ERP peak names from the AEP-explanation sentence.
- Keep the sentence's intent: explain *why* AEP carries identity
  signal, in terms accessible to a reviewer who is not an EEG
  specialist.
- Preserve the existing 5-page envelope and 0-overfull build profile.
- Touch only `main.tex`; rebuild and resync PDFs.

**Non-Goals:**
- No new experiments, no new tables, no figure changes, no new
  citations.
- No retitling, no abstract changes.
- Not addressing the kit-level `\ProvidesClass{APSIPA2025}` warning
  (lives in the official template).
- Not adding the optional "real-time-compatible on this desktop-GPU
  evaluation platform" hedge (already covered by the explicit
  RTX~5090 mention in the preceding clause).

## Decisions

### D1: Drop specific ERP peak names entirely (Option A)

**Chosen:** Replace the offending clause with a peak-free,
latency-class-free description that anchors the identity signal in
*latency- and amplitude-domain morphology*:

```tex
Auditory evoked potentials are the cortical response to a controlled
acoustic stimulus; their latency- and amplitude-domain morphology can
carry subject-specific structure that originates in individual
differences of cortical anatomy and auditory processing pathways,
which makes them attractive as an internal, replay-resistant biometric
cue.
```

**Why:** This is the smallest edit that removes both errors at once.
It does not commit the paper to a specific component family, does not
require a paradigm claim (oddball vs passive), and keeps the
reviewer-friendly framing the rest of the sentence already establishes.

**Alternatives considered:**

- **Option B — reclassify and rename** ("their long-latency cortical
  components, e.g., N1/P2 peaks ~100--200 ms"). Rejected for this
  change: technically accurate but commits to a specific component
  pair the paper does not later analyse, and adds a numeric latency
  range that is more detail than the abstract narrative needs.

- **Option C — keep peak names, drop only the latency class** ("their
  event-related components, e.g., N100/P200 peaks"). Rejected:
  $N_{100}$/$P_{200}$ are reasonable for passive AEP, but the paper
  never uses them as features, so naming them creates an expectation
  the rest of the manuscript does not pay off.

### D2: Rebuild and resync the shipped PDF in the same change

**Chosen:** After the wording edit, run `latexmk` and copy
`docs/apsipa2026/main.pdf` over
`docs/APSIPA2026_NeuroBiometrics.pdf` so the two PDFs remain
byte-for-byte equal.

**Why:** The `apsipa-build-artifacts` capability (added by
`apsipa-pdf-sync`) requires byte-for-byte equality before lock. Doing
the resync in the same change keeps the repository invariant true at
every change boundary.

**Alternative considered:** Defer the resync to a later "release"
change. Rejected — leaves the byte-equality invariant violated
between changes, which is exactly the failure mode
`apsipa-pdf-sync` was created to prevent.

## Risks / Trade-offs

- Risk: Wording change reflows the Dataset paragraph and pushes a
  table float to a different page.
  Mitigation: Rebuild and inspect; if reflow occurs, accept it
  provided the page count stays at 5 and there are no new overfull
  boxes.

- Risk: The replacement is *too* generic — a reviewer who wanted to
  see specific ERP peaks named may find the sentence vague.
  Mitigation: Accept this as a deliberate trade-off; the rest of the
  paper does not analyse specific peaks anyway, so the previous
  specificity was unsupported.

- Risk: Edits introduce a subtle typo, dropped LaTeX brace, or
  duplicated phrase.
  Mitigation: Diff the rebuilt PDF against the prior 5-page output
  visually around the Dataset paragraph; verify SHA-256 equality of
  the two PDFs after the resync.
