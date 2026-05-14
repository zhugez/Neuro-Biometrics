## Context

`docs/apsipa2026/main.tex` currently builds a 5-page submission whose
title — "Multimodal Mamba Fusion for Noise-Robust EEG Biometric
Identification" — has been carefully scoped to identification rather
than verification, and whose results section already separates Stage~I
SI-SNR from V4 retrieval wins (see the explicit disclaimer at
`main.tex:420-428`). Three sentences elsewhere in the manuscript
escaped that scoping pass and now read more strongly than the surrounding
narrative supports. None of them is a blocker, and an explore-mode
review confirmed the paper is submittable as-is.

## Goals / Non-Goals

**Goals:**
- Make the abstract, Detailed V4 Results paragraph, and Related Work
  novelty claim consistent with the (already corrected) title and the
  Ablation discussion's "SI-SNR is Stage~I descriptive only" disclaimer.
- Preserve the existing 5-page envelope and 0-overfull build profile.
- Touch only `main.tex`; do not modify figures, tables, bib, or code.

**Non-Goals:**
- No new experiments, no new tables, no figure changes.
- No retitling or restructuring (the title is already final).
- No expansion of the Discussion or Limitations sections.
- Not addressing the kit-level `\ProvidesClass{APSIPA2025}` warning
  (that lives in the official template, not in our source).

## Decisions

### D1: Soften the abstract "highest scale-invariant SNR" sentence

**Chosen:** Reword the V4 win sentence so SI-SNR is presented as the
*corresponding* Stage~I quality, not as part of the V4 win condition.
The replacement sentence is built around the existing sub-clause
"the corresponding Stage~I reconstruction quality is SI-SNR …".

**Why:** Matches the ablation paragraph's explicit "we do not attempt to
attribute SI-SNR deltas to the Stage~II additions" and the Conclusion's
phrasing. Keeps the abstract self-contained.

**Alternative considered:** Drop the SI-SNR numbers from the abstract
entirely. Rejected — readers who skim only the abstract still want a
single signal-quality anchor.

### D2: Replace "large SI-SNR gains" with absolute phrasing

**Chosen:** Use "the denoiser reaches its highest SI-SNR in the
power-line condition" instead of "the denoiser produces large SI-SNR
gains in the power-line condition".

**Why:** Table~II only reports denoised SI-SNR (no unprocessed
baseline column), so the word "gains" is unanchored. "Highest" describes
exactly what the table shows.

**Alternative considered:** Add an unprocessed-baseline column to
Table~II. Rejected — that requires recomputing baselines and would risk
overflowing the already-tight 7-column table.

### D3: Match the novelty claim to the title

**Chosen:** Change "for EEG biometric verification" to "for EEG
biometric identification with open-set verification analysis" in the
Related-Work State-Space-Models paragraph.

**Why:** The title says *Identification*; the abstract is honest about
the verification gap; the novelty claim is the only place that still
says verification. This is a one-word fix.

**Alternative considered:** Drop the claim entirely. Rejected — it is
the paper's positioning sentence and is fine once aligned.

## Risks / Trade-offs

- Risk: Wording change reflows a paragraph and pushes a table float to
  a different page.
  Mitigation: Rebuild and inspect; if reflow occurs, accept it provided
  the page count stays at 5 and there are no new overfull boxes.

- Risk: Edits introduce a subtle typo or duplicated phrase.
  Mitigation: Diff the rebuilt PDF against the prior 5-page output and
  spot-check the three patched regions visually.

- Trade-off: The reviewer's optional Minor 4 ("real-time-compatible on
  this desktop-GPU evaluation platform") is not applied here, since the
  current sentence already names the RTX~5090 explicitly.
