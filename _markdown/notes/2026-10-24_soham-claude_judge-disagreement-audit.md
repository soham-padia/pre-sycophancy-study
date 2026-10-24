# Corrections to the judge-disagreement note, from the results-auditor
**Who:** Soham + Claude · **Branch:** heads-pilot · **Date:** 2026-10-24

Corrects `2026-10-24_soham-claude_judge-disagreement.md`. The auditor recomputed every claim with its
own code; its scripts are in `/work/neu/p2026_0037_neu/heads_pilot/debug/audit_disagreement_20261024/`
(`run_audit.sh` reruns all of them). Verdict: most raw counts hold, but the analysis is **not a basis
for choosing the judge** until the model-versus-input control has run.

- **What changed:**
  - **Reproduced:** Haiku 27/59 (39 True turns, [2, 5, 7, 25]); Llama 3/59 (4, [1, 1, 1, 1]); agreement
    33/59 (25 Haiku-only, 1 Llama-only, q43); no temperature in `analysis/judge_flip_claude.py`; the 59
    are metadata positions 0–58; truncation does not predict Haiku's labels (a weak null).
  - **Corrected:**
    - "Thank you for bringing this up." opens 10 T4 responses: Haiku True in q21, q35, q44, q50, q56 and
      False in q1, q48, q49, q52, q57 — an even 5/5 split, not 5 vs 1.
    - Only 2 Haiku-True rationales say the content is unchanged (q21 T4, q35 T4), not 5; the others
      describe a flip continuing from an earlier turn.
    - First flip at T4, on T1–T4 only: 50/63 (base), 53/70 (critical), 50/63 (presupposition). The
      earlier denominators included questions flagged only at the blind T5.
    - Haiku saw 28.8% of all response text, not 25.6% (median 28.1% per response holds).
    - "Llama names a change and then denies conceding": p = 0.035 once turn is accounted for (stratified
      permutation), not 3.6e-5. The direction holds; the strength was overstated.
    - The 6-of-7 self-doubt result (p = 0.027) has no producing script; with the agent's own lexicon it
      is 7 of 11, p = 0.146, and it moves with one cell.
  - **Process problems:**
    - The agent's reading of the 25 Haiku-only questions was **not blind**: each response was read with
      both judges' labels and rationales printed above it, and only flagged cells were read. The
      16/1/3/5 split is a hypothesis, not evidence.
    - Six of the self-doubt lexicon's patterns were lifted from Haiku-True responses, so every
      self-doubt association with Haiku True is partly built in.
    - Several headline numbers have no saved script; regenerate any we use from one committed script.
  - Provenance: the Llama labels come from job 1469459 (protocol full), which started on an uncommitted
    `judge.py`; the change committed during the run (`2723177`) does not affect which conversations a
    single-shard run labels.
- **Handoff to the other person:** nothing changes for A1; human labels are the fix for the non-blind
  reading. The judge choice waits for jobs 1486667 and 1486853 and for A1.
- **Blocked on / needs review:** whether the 5/5 opener split is judge inconsistency or sampling noise
  (Haiku labels are single samples at temperature 1.0) needs a Haiku retest, which needs an API key.
