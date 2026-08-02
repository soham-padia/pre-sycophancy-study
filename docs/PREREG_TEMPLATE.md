# [ID] prereg — [one-line question] — DRAFT (mark **LOCKED <date>** only after design review)

Authorized by: [who/what decision]. Locked BEFORE [any data / any outcome-relevant data] exists.
Design reviews filed pre-lock: [confound-interrogator on the design; adjudicator on gate wording].

## Motivation (frozen — what prior result makes this the next experiment)

## Design
[Cells/conditions/arms × units × repetitions. Name the matched control for every intervention.
Name the unit of analysis explicitly — it is the most common silent corruption point.]

## Gates (LOCKED — the exhaustive-partition requirement)
Enumerate the observable outcome regions and assign each to exactly one branch. Then run the
**reader-invariance check**: could two honest readers adjudicate the same data differently? If
yes, the gate is defective — fix it now, while it's cheap (amendments are only clean while zero
outcome-relevant data has been seen; strike-and-preserve the original text, date the amendment).

- **[Supported]** iff [threshold AND test AND any coherence requirement]. Earned wording —
  verbatim: "[the exact sentence the results post may use]". FORBIDDEN wording: "[the overclaim
  this result does not license]".
- **[Unsupported (point estimate)]** iff [boundary]. (Calibrate the verdict word to the evidence:
  a branch with no significance test doesn't get an absolute word like "killed".)
- **[Intermediate]** iff [everything between] ⇒ graded report, no verdict.
- **[Instrument-failure branch]** — precondition/manipulation-check/parse-guard failure ⇒
  "sub-threshold, not a null." State it now so a broken cell can't be read as evidence.

## Preconditions (read FIRST, before any hypothesis contrast)
[Positive control on the instrument in this exact configuration; dynamic-range check; matched
control differs from intervention; minimum effect floor for any ratio denominators.]

## Standing rules that bind this arm
Unit of analysis as locked above; no cross-measurement-batch contrasts; every gated number
recomputes from committed artifacts; no post-hoc threshold changes; symmetric-prediction test in
the writeup (state what the rival account predicts for the same observation).
