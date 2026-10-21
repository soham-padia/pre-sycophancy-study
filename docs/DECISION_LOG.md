# DECISION LOG — the project's append-only decision trail

Direction memos (research-director), executor acknowledgments, adjudication rulings, and
results posts land here, newest at the BOTTOM, each under a dated header:

`## [AUTHOR/AGENT · YYYY-MM-DD] title`

Rules: append-only (never edit or delete an earlier entry — corrections are new entries that
reference the old); every direction memo is executed only after explicit acknowledgment below it;
every ruling states what it examined and what it did NOT (scope creep in summaries is how stale
verdicts propagate). This file is the "how it got there" companion to CITABLE_NUMBERS.md's
"what we claim."

## [claude-opus-5 coordinator · 2026-09-16] ARR August reviews received; full adjudication routed; 16 taxonomy rows and 9 errata filed

**What happened.** ACL ARR 2026 August submission 1220, "Before the Flip: Hidden-State Precursors of
Multi-Turn Sycophancy", returned three reviews: overall **2 / 2 / 2.5**, soundness **2.5 / 2.5 / 3**,
excitement **3.5 / 4 / 2.5**. Two reviewers recommend resubmit next cycle, one borderline-Findings.
All three scored **Software: 1 — no usable software released**; datasets scored 1 / 2 / 4.

**Producer claims routed, not adjudicated in place.** The coordinator's own code reading produced four
claims (task mismatch between the cosine and probe arms; source-fit scaler/PCA in cross-model
transfer; a matched comparison allegedly free from a committed artifact; transfer coverage/wording).
Per the producer-never-adjudicates rule these went to results-auditor (numbers),
confound-interrogator (meaning) and falsifier (the record). **Two coordinator claims were corrected by
the adjudicators and the corrections are recorded** (ERRATA E4): the "within-family" wording is at
README:13 and appears nowhere in CLAUDE.md, and the proposed free matched comparison does not exist
because the rows in question are at-flip, metric-mismatched and headroom-free.

**Process-layer precondition.** `verdict-integrity` item 5 requires the process layer to verify before
any verdict. It did not: the six agent definitions and the skill were sitting in an untracked `_base/`
directory and `.claude/` held only local settings, so the skill and agents CLAUDE.md mandates could
not be invoked at all. They were moved to `.claude/`, where all seven files matched the existing lock
hashes bit-for-bit. The lock was then regenerated to bless two deliberate changes: the
`CLAUDE.md.template` → `CLAUDE.md` rename, and operational sections (commands, pipeline/artifact
shapes, repo geography) appended to CLAUDE.md in this session. `bin/skills-lock.py verify` now passes.
Because the agent registry is built at session start, the adjudicators ran as general-purpose agents
carrying each contract verbatim rather than as registered agent types — the contracts were
hash-verified against the lock, but this is a deviation worth noting.

**Rulings.** The behavioural results and the judge instrument came out strong; the hidden-state
interpretation did not.
- **Numbers that survived independent recomputation:** all 12 cosine peak AUCs and the T1-exclusion
  variants (≤5e-5 from raw tensors), the 0.596–0.636 range by two independent routes, the entire
  Haiku side of the judge-agreement appendix, and three of CLAUDE.md's own key facts (66–91% chance
  span; T1 flip concentration 3.0% / 0.8%; L2 monotone runs 4/4/3/0). The judge CSVs are clean:
  5 turns per question, no duplicates, no parse failures, ns reconciling cell-for-cell.
- **Interpretations that did not survive:** submission finding 3 ("pre-flip geometry is nonlinear") is
  ruled **fatal** — reproduced in full by a turn-index-only classifier with no hidden states, on top
  of an at-flip artifact. The transfer negative is ruled **uninformative** rather than
  method-limited: constant single-class output in 12/12 pairs and identical accuracy under shuffled
  source labels in 11/12. The "Fisher ratios near zero" negative keeps its conclusion and loses its
  stated evidence.
- **The headline positive result stands narrowed.** The T0→T1 cosine ever-flip arm is structurally
  immune to the turn-index confound (turn index is constant across its examples — confirmed by the
  interrogator against the coordinator's read). What binds it instead: layer-profile incoherence in
  11 of 12 cells, response-text features surviving the project's own sweep correction in 3/12 cells
  against the cosine's 1/12, and a question-only predictor with zero model involvement reaching the
  same AUC band. **Qwen2.5-7B under the base prompt is the one cell clean on every axis tested** —
  a single-model, single-condition, n=155 existence result.

**What this examined and what it did NOT.** Examined: the committed analysis scripts and artifacts,
the local 22 GB of tensors, the committed judge CSVs, the tracked May-21 tex, and the record in
`docs/` and `_markdown/`. **Not examined:** the actual ARR-August submission PDF or tex, which is not
in this repo — the newest local tex is `current_paper_may_21/` (May 22) and `grep` for
"do not transfer" finds no match in any tracked tex, so every wording verdict against the submitted
abstract is provisional and must be re-confirmed against Overleaf. Also not examined: whether the
Opus judge CSV exists outside the repo; the Appendix-G letter (the tracked tex lettering differs);
seed stability of the corrected 7.24 pp figure (single seed only).

**Filed.** 16 dated rows in `docs/FAILURE_TAXONOMY.md` (classes A, B, C, D, E, F) and 9 dated entries
in `docs/ERRATA.md` (E1–E9). Two 2026-08-02 entries were deliberately **not** re-filed — a failure
recorded once is paid for once — and are instead flagged as still-live in the paper text.

**Not done, and deliberately left to the user.** No code was fixed, no README line was corrected, no
paper text was changed, and `docs/CITABLE_NUMBERS.md` remains empty. The one-character pairing fix in
`probe_judge_comparison.py:122-135` plus regeneration is available but would produce new numbers
needing their own audit pass, and correcting public-facing text is the authors' call.

**Standing consequence for the frozen chain.** `CITABLE_NUMBERS.md` cannot be populated as the rule is
written: `/data/` is gitignored so no behavioural CSV or tensor for any paper model is committed, the
only committed raw responses are a Qwen2.5-0.5B pilot that is not in the paper, `analysis_claude/` is
precisely the analyzer-produced summary the rule forbids recomputing from, and `download_data.py`
passes no `revision=` so no data-snapshot identifier exists to freeze against. The rule must either
gain committed behavioural CSVs (~20 MB, not the 22 GB) or name and pin the snapshot host. This is
also the direct fix for the three Software: 1 scores.

## [coordinator · 2026-10-21] Corrections to two 2026-09-16 rulings: pairing timing and the drift dissociation

**Examined.** The hidden-state extraction code and its history on every branch, and the committed
Qwen2.5-0.5B pilot tensors recomputed under the pre-response and generated-token hypotheses
(code-verifier, CPU job 1449390); `analysis_claude/cosine_auc_ci_bands.csv` for the drift–AUC relation
(falsifier); the committed judge CSVs and `permutation_test.csv` for a power analysis (results-auditor).

**Rulings.** (1) The per-turn states are pre-response for the extraction code path (ERRATA E10). The
2026-09-16 characterisations "on top of an at-flip artifact" and "at-flip, metric-mismatched" do not
stand. The ruling that submission finding 3 is fatal is unaffected, because the turn-index baseline ran
on the published artifact. (2) The drift–AUC dissociation is not supported; the narrowest supported
wording is in ERRATA E11. (3) Power analysis: the counts and the single-layer power (0.42–0.70 at a true
AUC of 0.6) were confirmed by independent Monte-Carlo. The producer's Bonferroni bound was corrected: the
permutation thresholds behave like about 6–14 independent layers, giving a sweep-corrected power of
roughly 0.12–0.32 for a one-layer effect and 0.3–0.6 for a 4–8-layer band. The claim that the surviving
cell is the highest-powered one holds by model, not by cell (Qwen2.5-critical has slightly more power
and peaks at 0.531). Not frozen: it enters `docs/CITABLE_NUMBERS.md` with the meta-review work.

**Not examined.** The paper-model tensors (absent from this machine; the layer-0 gate in E10 decides
them), the Overleaf text, and the drift relation in the eight non-best (model, condition) cells.
