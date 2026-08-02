# RESEARCH MAP — destination, open frontier, decisions index

*Wayfinder-style planning (adapted from [mattpocock/skills](https://github.com/mattpocock/skills)
`wayfinder`, resized for research): the map is an INDEX, not a store. Decisions live where they
were made (CITABLE_NUMBERS, DECISION_LOG); this file holds only the destination, the OPEN
frontier, and pointers. Maintained by the research-director on every memo and by the coordinator
when a thread opens or closes. **Every session orients here first** — the open frontier is the
list that otherwise lives in one agent's working memory and silently drops threads.*

## Destination
Acceptance of "Before the Model Caves" at EACL 2027, with every externally cited number
reproducing from committed artifacts by independent recomputation.
Status 2026-08-02: NOT yet submitted — ARR submission due tonight (Aug 3 deadline); all authors
register as ARR reviewers by Aug 5; commitment to EACL Oct 11.

## Open frontier (decision tickets — one line each; close with a date + pointer, never delete)
- Full-record verification pass — FIRST PASS DONE 2026-08-02
  (`_falsifier/2026-08-02_{behavioral,hiddenstate}_audit.md`; 9 confirmed issues fixed in
  paper-jul-6 `2b43858`). Still open: commit the Opus judge CSV + arbitration sheet (the κ
  appendix is currently unverifiable from the repo); Fisher-combined p seed-sensitivity rerun
  (3 seeds × 10k shuffles, quote the range); confirm the 4open mirror carries the behavioral CSVs.
- Positive control on the probe instrument: can the identical probe pipeline decode a construct
  it SHOULD decode (turn index, prompt condition) from the same hidden states? Bears directly on
  the "linear probes fail" null (constitution rule M1). Blocked on hidden-state tensor access.
- Advisor experiment queue for the ARR author response (opened 2026-08-02, from the advisor's
  17-point review; all three are named in the paper as planned checks): (a) T0-correctness
  fraction + rerun restricted to verified-correct T0 (~1 day, Haiku judge over 2,290 T0
  responses, CREPE gold corrections as reference); (b) pressure-blind judge seeing only T0/Tk
  responses (~1 day, re-judge 11,450 turns, compare kappa); (c) neutral non-pressure control
  arm (the causal control; pilot Qwen2.5 T0-to-T1 first, ~2-3 days cluster; full arm 1-2 weeks).
- Populate `docs/CITABLE_NUMBERS.md` from the verification pass; freeze with data-snapshot SHAs.
- Hidden-state release upon acceptance (~22GB): hosting plan + de-anonymization transition.
- Global-workspace (J-space) follow-up study — post-submission.

## Decisions so far
- Frozen chain: `docs/CITABLE_NUMBERS.md` (empty until the verification pass lands).
- Session log + rulings to date: `_markdown/AGENT_NOTES.md` (append-only, predates this map).
- Submission status + revision-letter tracking: `_markdown/PAPER_STATUS.md`.
- Standing failure rules: `docs/FAILURE_TAXONOMY.md` + the off-by-one trap in
  `_markdown/ONBOARDING.md`.

## Out of scope (ruled, don't reopen without a director memo)
- `debate_setting/` and `ethical-setting/` — excluded from the paper and the anonymized repo
  (user ruling, 2026-07; see `_markdown/AGENT_NOTES.md`).
- Re-opening SYCON-Bench attribution wording — settled: we EXTEND, never "introduce"
  (`_markdown/AGENT_NOTES.md` 2026-07-27).
