## 1. Locate the three target sentences

- [x] 1.1 Open `docs/apsipa2026/main.tex` and confirm the abstract V4 win
      sentence still reads "highest scale-invariant SNR" near line 109
- [x] 1.2 Confirm the Detailed V4 Results paragraph still reads "large
      SI-SNR gains" near line 380
- [x] 1.3 Confirm the Related Work State-space-models paragraph still
      reads "for EEG biometric verification" near line 154

## 2. Apply the three wording edits

- [x] 2.1 Replace the abstract V4 win sentence so the V4 result is "best
      Precision@1 on every noise type" and the SI-SNR number is presented
      as the *corresponding* / *associated* Stage~I value
- [x] 2.2 Replace "the denoiser produces large SI-SNR gains in the
      power-line condition" with "the denoiser reaches its highest SI-SNR
      in the power-line condition" (or equivalent absolute phrasing)
- [x] 2.3 Replace "for EEG biometric verification" with "for EEG biometric
      identification with open-set verification analysis" in the Related
      Work novelty sentence

## 3. Rebuild and verify the envelope

- [x] 3.1 Run `latexmk` (or the project's standard build) on
      `docs/apsipa2026/main.tex`
- [x] 3.2 Verify the rebuilt PDF is still 5 pages, A4, fonts
      embedded/subset, and that the log shows 0 overfull boxes and 0
      undefined refs/citations
- [x] 3.3 Visually spot-check the three patched regions (abstract,
      Detailed V4 Results, Related Work) for typos or duplicated phrases
- [x] 3.4 Refresh the shipped copy at `docs/APSIPA2026_NeuroBiometrics.pdf`
      (byte-for-byte equal to `docs/apsipa2026/main.pdf`)

## 4. Close out

- [x] 4.1 Run `openspec status --change apsipa-claim-polish` and confirm
      `isComplete: true`
- [x] 4.2 Archive or hand off the change per the project's standard
      `/opsx:apply` workflow
