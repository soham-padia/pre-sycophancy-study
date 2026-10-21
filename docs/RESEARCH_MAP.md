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
Status 2026-09-16 (SUPERSEDES the line above): SUBMITTED as ARR August #1220 and REVIEWED — overall
2 / 2 / 2.5, soundness 2.5 / 2.5 / 3, Software 1 from all three reviewers. Two of three say resubmit
next cycle. Full adjudication and rulings: `docs/DECISION_LOG.md` 2026-09-16; 16 new taxonomy rows;
ERRATA E1-E9. The EACL commitment decision is now a live call, not a formality.

## Open frontier (decision tickets — one line each; close with a date + pointer, never delete)
- Full-record verification pass — FIRST PASS DONE 2026-08-02
  (`_falsifier/2026-08-02_{behavioral,hiddenstate}_audit.md`; 9 confirmed issues fixed in
  paper-jul-6 `2b43858`). Still open: commit the Opus judge CSV + arbitration sheet (**correction
  2026-09-16, ERRATA E9: "unverifiable" was overstated — the entire Haiku side reproduces exactly
  from committed artifacts and the 2x2 is arithmetically constrained; commit the Opus CSV to close
  this cleanly, not to rescue it**); Fisher-combined p seed-sensitivity rerun
  (3 seeds × 10k shuffles, quote the range); confirm the 4open mirror carries the behavioral CSVs.
- Positive control on the probe instrument: can the identical probe pipeline decode a construct
  it SHOULD decode (turn index, prompt condition) from the same hidden states? Bears directly on
  the "linear probes fail" null (constitution rule M1). ~~Blocked on hidden-state tensor access.~~
  **UNBLOCKED 2026-09-16** — all 12 files for the four probed models load clean (zero `None`
  tensors, 6 turns per question); only Gemma-base is corrupt and Gemma is behavioural-only.
  **PARTLY ANSWERED, and the answer is bad:** turn index is decodable at ~45 SEs of L2 separation
  and a turn-index-only classifier matches the best of nine probes (ERRATA E3). The instrument
  works; what it decodes is position, not stance. Remaining piece: prompt-condition decoding.
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
- Standing failure rules: `docs/FAILURE_TAXONOMY.md` + the pre-flip pairing-horizon note in
  `_markdown/ONBOARDING.md` (formerly "the off-by-one trap"; ERRATA E10).

## Out of scope (ruled, don't reopen without a director memo)
- `debate_setting/` and `ethical-setting/` — excluded from the paper and the anonymized repo
  (user ruling, 2026-07; see `_markdown/AGENT_NOTES.md`).
- Re-opening SYCON-Bench attribution wording — settled: we EXTEND, never "introduce"
  (`_markdown/AGENT_NOTES.md` 2026-07-27).

## Open frontier added 2026-09-16 (post-review; one line each, close with a date + pointer)
- **Decide the venue call first.** Two reviewers say resubmit next cycle, and the record now carries
  a fatal ruling on submission finding 3. Resubmitting the same framing is not available. Blocks
  everything below it.
- ~~**Fix `probe_judge_comparison.py:122-135`** to the reference pairing, regenerate
  `probe_judge_sweep.csv`, and route the new numbers through results-auditor before any use.~~
  **Corrected 2026-10-21 (ERRATA E10): under the extraction code there is nothing to fix** (paper
  tensors pending E10's gate) — both pairings are pre-behavioural
  and differ in horizon. Choose and name the §6.3 horizon (next-response as published, or
  one-turn-ahead), regenerate only if it changes, and route new numbers through results-auditor before
  any use. The one-turn-ahead 7.24 pp is a single-seed point estimate; quote a seed range (0/1/42),
  never the number.
- **Re-scope the paper to what survives.** Qwen2.5-base is clean on all seven tested axes
  (permutation, layer coherence, text baseline, question-only baseline, form coupling, zero
  keyword-visible pre-behavioural concessions, class balance). A single-cell n=155 existence result
  is defensible; the four-model claim is not.
- **Add the three missing baselines as first-class results, not rebuttals:** response-text features,
  stimulus-only (CREPE question) features, and turn-index-only. A reviewer can run all three in an
  afternoon; publishing them with the Qwen2.5 contrast is strictly better than being shown them.
- **Run the paired within-cell AUC comparison** (cosine vs best text feature, same questions, same
  labels, DeLong or bootstrap on the AUC difference). The only test that licenses the word
  "internal". ~20 lines.
- **Turn-stratified re-run of the classifier sweep** — within a turn index, turn is constant, so any
  residual accuracy is genuine. Feasible at t=1-3; t=4 strata are degenerate.
- **T0-correctness verification** — CREPE gold joins 100% exactly and every stimulus carries the
  gold `false presupposition` label, so the stimulus side of the DV is already validated; only the
  response side is open. One Haiku pass, then recompute on the T0-correct subset. Until then the DV
  is named "position/stance change".
- **Make the frozen chain satisfiable** — commit the behavioural CSVs (~20 MB) or amend the rule to
  name and pin a snapshot host, and add `revision=` to `download_data.py`. Same fix as the
  Software: 1 scores. Blocks populating `docs/CITABLE_NUMBERS.md`, still empty.
- **Create the anonymous artifact link** — `_markdown/PAPER_STATUS.md`'s submission checklist left it
  unchecked and three reviewers charged for it. Before any export, reconcile CLAUDE.md's exclusion
  list with the out-of-scope ruling above: 13 tracked files carrying author identity pass CLAUDE.md's
  filter today, including `PAPER_MAY_21/latex/acl_latex.tex` with the full author block.
- **Exclude tensor row 0 from every sweep** — it is a constant vector, so layer-0 AUC is exactly
  0.500 and two permutation cells report their argmax there (ERRATA E5).
- **Reviewer-response corrections owed** (all verified): paET's 2/3/9 hold sizes belong to the L2
  appendix loader, not the cosine arm (22-91 there); L9dW's "Krippendorff alpha" is Cohen's kappa;
  L9dW's "disagreement with Haiku 60%" is not in the paper — the nearest is Haiku matching the human
  verdict in 73/120 = 61% of disagreement cases.
