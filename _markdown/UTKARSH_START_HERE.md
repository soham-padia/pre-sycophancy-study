# Start here, Utkarsh — catching up as of Oct 24, 2026

You are **B** in the current plan. Soham is **A**. Read this file top to bottom (about 15 minutes), then
the five files in "Read these next" (about an hour). After that you can pick up Monday's tasks.

Two pages Soham shares with you separately (claude.ai, private):
- **The one-week plan** — day-by-day tasks for both of us, the handoffs, what was cut:
  https://claude.ai/artifact/FudYoUkt3yduq8bUiNsAiH
- **The annotation page** — where both of us label conversations, blind:
  https://claude.ai/artifact/BavGJrcFWEwCDi1ztvPWE4

---

## 1. What changed since August

**The reviews.** ARR August #1220 came back with a meta-review of 2 ("resubmit"). The area chair asked
for five things:
1. human validation of the flip labels instead of LLM–LLM agreement;
2. the cosine-drift and probe results evaluated on the same task;
3. T0 correctness checked, so a "flip" is known to leave a correct answer;
4. surface-feature baselines (response length, log-probability);
5. the layer-wise evidence moved into the main text.

We let the NAACL/COLING commitment (Dec 23) pass. The target is the **ARR January 2027 cycle (due
Jan 4)**, declared as a revision of #1220 (ARR requires the link for the same topic).

**The record was audited and corrected (Oct 21).** `docs/ERRATA.md` E10–E13 are the ones that matter now:
- **E10** Hidden states are *pre-response*: the last token of the prompt, before the model writes
  response t. Every pre-flip claim must name its horizon (state at t → label of response t, or of t+1).
  Neither pairing is "at-flip".
- **E11** The "drift–AUC dissociation" is not supported (3 of 4 models peak in the same late layers).
- **E12** Generation was capped at 512 tokens with `repetition_penalty=1.2`. DeepSeek and Qwen3.5 often
  never reach an answer; most Qwen2.5 responses are cut mid-text.
- **E13** The Haiku judge never saw any T5 response, so every T5 label is blind.

Older docs (`ONBOARDING.md`, `RESEARCH_SUMMARY.md`, `REVISION_TASKS.md`, `PAPER_STATUS.md`) carry a
"Superseded in part" banner. Trust `CLAUDE.md` and `docs/` over them.

**The direction changed.** The main contribution is meant to become **which attention heads drive a model
to give in under multi-turn pushback, and whether steering them changes how often it does** — if the
results support it. Design: the same conversations in two arms (the user disagrees vs the user agrees),
heads ranked on half the questions and steered on the other half, against random-head controls. The
cosine result becomes a preregistered replication alongside it. The old data become a disclosed pilot.
The new study runs on the **551 CREPE false-presupposition questions never used before**.

**Gemma-2-9B is dropped** from everything. Gemma-4-31B-it is a candidate model for later.

**This week's finding: the judge.** On the same 59 old Qwen2.5 conversations, Haiku (the paper's
judge) says 27 gave in; Llama-3.3-70B (our local judge) says 3. The two of us will settle which judge
reads these conversations correctly by labelling them blind this weekend (sample A1, below).
**Label A1 before you read why the judges are thought to disagree**, so that at least one of us labels
without knowing the hypothesis. After you've submitted all 34, the explanation is in
`_markdown/notes/2026-10-24_soham-claude_judge-disagreement.md`. The judge choice for the new study is
decided on Monday.

## 2. Setup (day 0)

```bash
git clone https://github.com/soham-padia/pre-sycophancy-study.git
cd pre-sycophancy-study
git checkout heads-pilot            # contains align-record; main is behind both
python -m venv .venv && .venv/bin/pip install -r requirements.txt     # Python 3.13, CPU only
.venv/bin/python download_data.py --csv-only                          # ~50 MB, enough for most work
.venv/bin/python bin/skills-lock.py verify                            # must print PROCESS LAYER INTACT
```

- The full download (with the hidden-state `.pt` files) is about 3.6 GB, not the 22 GB its docstring
  says. `download_data.py` pins no revision: the snapshot everything is checked against is
  `955b6da423654f36ec526232c38be6e950d6a84a` of `sohampadianeu/pre-sycophancy-study-data`.
- **Cluster (optional):** ask Research Computing to add you to AICR project `p2026_0037_neu`. You only
  need it to read the new data on `/work` directly; everything you own runs on a laptop.
- **Annotation page:** Soham sets your access to Contributor; otherwise your labels will not save.
- **Claude Code:** open the repo and `CLAUDE.md` loads as project law. `.claude/` ships the review
  agents (`results-auditor`, `confound-interrogator`, `falsifier`, `code-verifier`, `research-director`,
  `big-picture-synthesist`) and the `verdict-integrity` skill.

## 3. Read these next

1. `CLAUDE.md` — the project law. "Key facts to not re-break" and "Verdict discipline" especially.
2. `docs/ERRATA.md` — E10–E13 in full; skim the E1–E9 headings.
3. `docs/prereg/README.md` and `P0`–`P5` — the Oct 21 drafts. You rewrite these into one full-study
   prereg on Monday; the plan page lists every review finding it must carry.
4. `heads_pilot/README.md`, then the code in `heads_pilot/` — the pipeline the new study runs on.
5. The plan page.

## 4. Your week

**This weekend, before Monday:** label sample A1 on the annotation page: 34 conversations, about 2.5
hours, blind. Instructions and what not to open: `_markdown/notes/2026-10-24_soham-claude_a1-annotation.md`.

The full schedule is on the plan page. Your part, in short:

| Day | You | You need from Soham | Soham needs from you |
|---|---|---|---|
| Mon | Write the full-study prereg with every review fix; revise the annotation codebook | Judge choice and smoke-test numbers, 14:00 | Prereg to the reviewers, 20:00 |
| Tue | Lock the prereg with Soham; write the analysis code, dry-run it on old data | — | **Locked prereg, 12:00** (generation waits on it) |
| Wed | Revision letter; methods; anonymization cleanup | Generations on `/work`, Wed night | — |
| Thu | Cosine replication; surface features | Judge labels; log-probabilities | **Annotation sample + hash, 10:00** |
| Fri | T0 analyses; surface controls; finish annotating | — | — |
| Sat | Agreement (κ); reproduce Soham's numbers | Heads results | Reproduce-check of his numbers |
| Sun | Labels, T0 and controls results; the letter | — | Your sections |

If you can't get cluster access, Soham copies the new data to wherever you can read it on Wednesday night.

## 5. How we communicate

**Our Claude sessions cannot message each other** (they run under different accounts), so the
repository is the channel:

- **Session notes: one new file per entry in `_markdown/notes/`**, named
  `YYYY-MM-DD_<who>_<topic>.md`, using the template in `_markdown/notes/README.md`. A new file never
  conflicts in git; two people appending to the same `AGENT_NOTES.md` did. `AGENT_NOTES.md` stays as the
  archive up to Oct 24. At the start of every session: `git pull`, then read the notes newer than your
  last one (`ls -t _markdown/notes | head`).
- **Code:** one branch per task, made from `heads-pilot`, merged into `heads-pilot` through a PR the
  other person reads. Never commit to `main`.
- **Handoffs:** the H1–H11 table on the plan page. When you finish one, say so in a note.
- **Numbers:** whoever produced a number never signs it off. The other person reproduces it from
  committed artifacts before it goes into `docs/CITABLE_NUMBERS.md`.

## 6. Rules that bite

- **Anonymization.** No names, emails or institution names in any tracked file outside `_markdown/`,
  including comments, file names and plot labels. A previous submission was desk-rejected for this.
  The repo is public; the anonymous submission copy is a separate single-commit snapshot without
  `_markdown/`. Files outside `_markdown/` that still carry identity and must be cleaned before that
  snapshot (checked Oct 24): `.gitignore`, the three `analysis/judge_outputs/*.log` files,
  `analysis_claude/judge_disagreement_assessment.txt`, `download_data.py` and `docs/prereg/README.md`
  (the HF dataset name), `sbatch/run_false_presuppositions_full.sbatch`, `docs/ERRATA.md`, and every file
  in `heads_pilot/` (hard-coded `/home/padia_so_neu` and `/work/neu/...` paths). Re-check with
  `git ls-files | grep -v '^_markdown/' | xargs grep -lIiE 'soham|padia|malihe|utkarsh|@gmail|sohampadianeu|/neu/p2026'`.
  It also flags `false-presuppositions-setting/data/test.jsonl`, whose CREPE passages contain the word
  "Soham" (a Wikipedia list): that is dataset text, not identity.
- **base / critical / presupposition are system-prompt conditions** over the same questions, never
  "question types" (the code's `question_type` column means condition).
- **Say "position change", not "capitulation"**, until T0 correctness and label agreement are adequate.
- **Never cite the 120-case arbitration as human agreement.** Its file is an LLM's assessment.
- **Judge runs:** always pass `--output` to `analysis/judge_flip_claude.py` (its default overwrites the
  DeepSeek labels), and give every `heads_pilot/judge.py` run a new folder. The judge skips conversations
  already labelled in its folder, so a reused folder attaches old labels to new text.
- **Generation on this cluster:** bf16 `generate()` with a KV cache is not exact here. Generate only
  through `heads_pilot/safe_gen.py::load_verified`, which proves ~400 tokens against recomputation first.
- **Cluster:** data on `/work`, never `/home` or `/scratch`; log every job with `runlog.py`; never
  `set -u` in an sbatch script.
- **Preregs lock before the data they govern exists.** New findings are dated addenda; corrections go to
  `docs/ERRATA.md` with a date.

## 7. Decisions still open (as of Oct 24)

- **D1** Which judge labels every flip — decided Monday from the calibration job and the analysis above.
- **D2** Preregister the heads and the cosine replication together — approved with the prereg on Tuesday.
- **D3** Declare as a revision of #1220 and ask for new reviewers and a new area chair — Monday.
- **D5** Greedy decoding, or the paper's repetition penalty 1.2 — recommended greedy, Monday.

## 8. Where things are

| What | Where |
|---|---|
| Code | GitHub `soham-padia/pre-sycophancy-study`, branch `heads-pilot` |
| Data snapshot | HF `sohampadianeu/pre-sycophancy-study-data` @ `955b6da`; on the cluster `/work/neu/p2026_0037_neu/pre_sycophancy_data_955b6da` |
| Pilot outputs | `/work/neu/p2026_0037_neu/heads_pilot/` (generations, judge labels, `debug/` reports) |
| Judge-disagreement analysis | `/work/neu/p2026_0037_neu/heads_pilot/debug/disagreement_20261024/` |
| Job ledger | `/work/neu/p2026_0037_neu/runlog/commands.jsonl` |
| The paper | Overleaf (canonical) |
