---
name: verdict-integrity
description: Run this checklist BEFORE any result becomes a verdict — before recording a number in CITABLE_NUMBERS, citing it externally, calling a null a null, or adjudicating a preregistered gate. It is the cross-cutting layer the per-domain skills don't cover — instrument positive controls, measurement-batch discipline, unit-of-analysis, multiple-comparison families, gate exhaustiveness, and record consistency. Case history and rationale live in docs/FAILURE_TAXONOMY.md; route doubts to results-auditor (computation), confound-interrogator (meaning), or falsifier (the record).
---

# verdict-integrity

A result is not a verdict until it survives this list. In the origin project's catalog the most
expensive failure mode — plausibly general, unproven beyond n=1 — was the **false null protected
by a missing positive control**, so the checklist leans hardest on nulls.

## 1. Instrument first
- [ ] **Positive control on THIS instrument in THIS configuration.** An instrument validated in
      one configuration licenses nothing about another. Every new subject/system × instrument
      pairing gets its own positive control before its nulls count.
- [ ] **Null on a judged/measured channel?** Confirm the instrument detects the ground-truth
      effect in this exact config before trusting that it can't see the induced one.
- [ ] **Headroom:** where does baseline sit? Ceiling/floor-pinned readouts earn only
      non-specificity, never inertness.
- [ ] **Panel counts:** report effective dimensionality (halo/correlation collapse); state the
      correction family and the surviving subset; uncorrected counts are not citable.
- [ ] **Measurement batches:** no contrast may cross measurement runs (judges, annotators, even
      re-runs of a stochastic scorer). New cells ⇒ re-measure comparison cells in the SAME batch.
      Single-run results get a stability replication before external citation.

## 2. Statistics
- [ ] **Unit of analysis:** the locked unit, not the convenient one. Pooling with shared
      seeds/items is pseudo-replication — it never gates.
- [ ] **Independent reproduction before freezing:** recompute from raw committed artifacts by a
      path that isn't the analyzer's own summary.
- [ ] **Freeze with a data-snapshot identifier**, not just a code version — pools grow.
- [ ] **Seed/sample hygiene:** any pool top-up asserts uniqueness against the existing pool.

## 3. Interpretation
- [ ] **Decodable ≠ causal ≠ load-bearing:** which is actually shown, by intervention evidence?
      Degradation is damage, not the construct. Broken cells are "sub-threshold, not a null."
- [ ] **Orthogonal axes never revise each other:** results on one axis (e.g. addition-side) don't
      rewrite claims on another (removal-side). New finding ⇒ dated addendum, never an edit.
- [ ] **Symmetric-prediction test:** state what the rival account predicts for this exact
      observation. If identical, the result is not evidence between accounts — the wording must
      say so.
- [ ] **Wording is part of the result:** use the earned phrasing for the branch that occurred;
      the forbidden over-readings were named in the prereg — honor them.

## 4. Gates & preregs
- [ ] Branches form an **exhaustive partition**: enumerate observable regions; each maps to
      exactly one branch; two honest readers cannot disagree (reader-invariance).
- [ ] Kill/support symmetry: verdict words calibrated to the evidence behind them.
- [ ] Amendments only while zero outcome-relevant data has been seen; strike-and-preserve, dated.
- [ ] **Attack the prereg before results exist** (falsifier) — the only time gate defects are cheap.

## 5. Record
- [ ] **Process layer intact:** `python bin/skills-lock.py verify` passes — the agents and skills
      adjudicating this verdict are the reviewed versions, not drifted ones. Drift ⇒ stop; revert
      or review-and-regenerate the lock before any verdict.
- [ ] Verify condition **fields in artifacts**, never prose labels.
- [ ] Cross-doc sweep: every document stating this verdict agrees. A stale copy IS a live error.
- [ ] New failure discovered here? **Append to docs/FAILURE_TAXONOMY.md with a date**; if it
      generalizes, add the check to this file. A failure recorded once is paid for once.
