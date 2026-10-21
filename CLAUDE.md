# CLAUDE.md — project law for the pre-capitulation sycophancy study

Multi-turn sycophancy study extending SYCON-Bench's false-presupposition setting (Hong et al.
2025): 5 open-weight models, ~177 CREPE questions × 3 system-prompt conditions × 5 escalating
pressure turns; behavioral flip labels (LLM judge) plus layer-wise hidden states. Core question:
does capitulation have a detectable internal signature before any behavioral change? Analysis
scripts in `analysis/` write to `analysis_claude/`; the paper tex is mirrored in the paper-jul-6
repo (Overleaf is canonical). Team/process notes live in `_markdown/`.

## Environment
- Behavioral logs: `data/<Model>/{base,critical,presupposition}_multiturn.csv` + metadata JSON.
  Hidden-state tensors (~22GB `.pt`) are local/cluster only, never committed.
- Judge labels: `analysis_claude/*judgements*.csv` — judge is Claude Haiku 4.5 for ALL models;
  first-flip methodology; baseline corrections are not counted as flips.
- The anonymized public snapshot is a separate single-commit repo; commits there are authored
  Anonymous ONLY. `_markdown/`, `_local/`, `docs/`, `.claude/`, `_falsifier/`, `debate_setting/`,
  `ethical-setting/`, and this file never enter any anonymized export.

## Key facts to not re-break
- base/critical/presupposition are SYSTEM-PROMPT CONDITIONS over the SAME ~177 questions —
  never "question types" (that misdescription shipped once).
- Per the extraction code, hidden states are PRE-RESPONSE: `hs[q][t]` is every layer at the last
  token of the templated prompt through the turn-t user message, before response t is generated (`models.py`; ERRATA E10 —
  verified by recomputation on the committed pilot; paper-model tensors decided by E10's layer-0 gate).
  So there are two pre-behavioural pre-flip pairings, and every claim must name its horizon:
  **next-response** (`hs[q][t]` → label of response t; `probe_judge_comparison.py::build_preflip_features`,
  every §6.3 number) and **one-turn-ahead** (`hs[q][t]` → label of response t+1;
  `train_probes_v2.py::build_preflip_dataset` and the three `b40e3ff` scripts). Neither is "at-flip" —
  ERRATA E1/E2/E8's post-behavioural reading is superseded. Both censor after the first flip and yield
  IDENTICAL n, so an n-check cannot tell them apart — diff the index arithmetic, and audit the producer
  of an artifact, never only its siblings.
- Peak cosine AUCs (0.596–0.636) are layer-sweep-selected: descriptive only. Under the
  sweep-corrected permutation test only Qwen2.5-base survives (p=0.035); Fisher-combined
  p=0.047 with condition choice uncorrected. Never cite individual peaks as significant.
- The drift–AUC "dissociation" is NOT supported: in 3 of 4 models drift and AUC peak in the same
  late layers (ERRATA E11 has the only wording the artifacts license).
- Until T0 correctness is verified, the DV is "position change", not "capitulation".
- Flip-rate class imbalance spans 66–91% chance levels (not 75–85%).
- Qwen3.5 keyword>judge holds ONLY under base/critical; presupposition reverses it.
- The judge does NOT concentrate flips at T1 (Qwen2.5 3%, Qwen3.5 1%); the keyword heuristic
  fires EARLIER than the judge.
- Paper wording: we EXTEND SYCON-Bench — never "introduce"; prevention claims are scoped to
  the two tested prompts and this benchmark.

## Verdict discipline (do not remove)
- **Before any result becomes a verdict** (recorded, cited externally, or a null called a null):
  run the `verdict-integrity` skill.
- Every citable number lives in `docs/CITABLE_NUMBERS.md` and must reproduce from committed raw
  artifacts by an independent recomputation. Freeze with a data-snapshot identifier.
- New findings are dated ADDENDA — frozen claims are never edited in place. Corrections go to
  `docs/ERRATA.md` with a date. New failures go to `docs/FAILURE_TAXONOMY.md` with a date;
  a failure recorded once is paid for once.
- Preregs lock before outcomes; gates must be exhaustive partitions (see docs/PREREG_TEMPLATE.md);
  the falsifier attacks preregs BEFORE data exists and the record AFTER arcs close.
- Producer never adjudicates: results route to results-auditor (numbers), confound-interrogator
  (meaning), falsifier (the record). Code routes to code-verifier before expensive compute.

## Commands
Analyses are CPU-only (`.venv` is Python 3.13). There is no test suite and no lint config;
`bin/skills-lock.py verify` is the only gate that fails.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python download_data.py [--csv-only] [--model Qwen2.5-7B-Instruct]  # data/ is gitignored
python bin/skills-lock.py verify     # process-layer drift; `generate` only after a reviewed change
```

The cosine/geometry/probe/plot scripts take no flags — the model list is a `MODELS` dict at the top
of each file, so a single-model run means editing that dict (or importing the module). The judge
selects models and conditions on the command line (`--models`, `--question-types`).

```bash
python analysis/cosine_disruption.py         # per-layer T0→T1 AUC → analysis_claude/cosine_disruption.{txt,csv}
python analysis/cosine_disruption_checks.py  # T1-flip exclusion, bootstrap CIs, T0 completeness
python analysis/permutation_test.py          # sweep-corrected permutation test (the significance claim)
python analysis/preflip_geometry.py          # Fisher ratios + LDA
python train_probes_v2.py                    # probes + REFERENCE pre-flip pairing
python probe_judge_comparison.py             # classifier sweep + cross-model transfer
python analysis/plot_*.py                    # figures → analysis_claude/
python analysis/judge_flip_claude.py --judge-model claude-haiku-4-5-20251001 \
    --output analysis_claude/<name>.csv      # needs ANTHROPIC_API_KEY; --dry-run --limit 2 to preview
```

ALWAYS pass `--output` to the judge: the default is `analysis_claude/claude_judgements.csv`, which is
the DeepSeek label file — a run resumes (appends) into it and `--overwrite` deletes it. The judge
resumes by skipping keys already present in its output CSV and shards itself with
`--num-shards N --shard-index i`; `analysis/split_judge_csv_shards.py` only re-splits an existing
output CSV so a serial run can resume as shards. Full script→output map: `_markdown/ONBOARDING.md`.
Everything under `analysis_claude/` is generated — regenerate via script, never hand-edit, and note
regenerations in `_markdown/AGENT_NOTES.md`.

GPU work (new inference, hidden-state extraction) runs only on a cluster:
`sbatch/submit_false_presuppositions_qwen35_loop.sh <hf-model-id>` wraps
`false-presuppositions-setting/run_benchmark.py` with 8-hour auto-resubmit (the deepseek7b/llama31
submitters hardcode their model and call it). The scripts were written for the Discovery cluster
(partition `gpu`, `module load python/3.13.5`, `set -euo pipefail`): elsewhere fix partition/account,
and on AICR drop `set -u` — it kills the batch shell there. Only `run_false_presuppositions_full.sbatch`
hardcodes a user home.

## Pipeline and artifact shapes
Four stages, each reading the previous stage's files off disk. There is no orchestrator.

1. **Stimuli** — `analysis/generate_multiturn_dataset.py` expands each CREPE question into T0 plus
   five fixed escalating pressure turns (the ladder is the `PRESSURE_VARIANTS` list; T4 is the
   authority turn, where most flips land).
2. **Generation (cluster)** — `run_benchmark.py` writes
   `false-presuppositions-setting/output/<hf-basename>/<cond>_multiturn.csv` (its default and the
   sbatch default; columns `Question`, `Response_Turn_0..5`) and `<cond>_multiturn_hidden_states.pt`:
   a dict `{question: [T0…T5]}`, each tensor `(n_layers+1, hidden_dim)` — index 0 is the embedding
   layer; one token per turn, the last token of the templated input prompt (pre-response, see above). `data/<Model>/` is what `download_data.py` pulls from the HF
   dataset; its directory names differ from HF basenames (`data/Gemma-2-9B` vs `google/gemma-2-9b-it`).
   Naming trap: the scripts' `MODELS[...]["n_layers"]` (29/33) is the tensor ROW count, not depth, and
   sweeps slice `[:n_layers]`, which includes the constant row 0 — exclude it with `[1:n_layers]`.
3. **Labels** — two independent labelers over the same responses: `flip_labeling.py::response_flipped`
   (keyword heuristic, deliberately stricter than the original SYCON list) and
   `analysis/judge_flip_claude.py`. Judge CSV columns:
   `model, question_type, question, turn, judgement, confidence, rationale`.
   `question_type` in every column name and filename means *condition* (see the law above).
4. **Analysis** — cosine/geometry/probe scripts join `.pt` tensors to a judge CSV **by question
   string**. Two label tasks that must never be mixed: **ever-flip** (question-level; drives the
   cosine AUCs) and **pre-flip** (state at t → label of response t or t+1 — name the horizon, see the
   law above; censored after the first flip).

Model coverage is uneven by design: Qwen3.5 has 112 questions everywhere, DeepSeek up to 176 — for
apples-to-apples claims use the common set: 112 questions in the Qwen3.5 tensors, 111 once all 12
judge-label cells are joined (Qwen2.5-base lacks one). `data/Gemma-2-9B/base_multiturn_hidden_states.pt`
is a corrupt truncated zip; Gemma's other two conditions are intact and never analyzed.

## Repo geography
- **In scope:** `false-presuppositions-setting/`, `analysis/`, `analysis_claude/`, the top-level
  probe scripts, `data/`.
- **Ruled out of scope** (don't revive without a director memo): `debate_setting/`,
  `ethical-setting/` — inherited SYCON-Bench settings, excluded from the paper and the anonymized repo.
- **Inert:** `PAPER_MAY_21/`, `Paper/`, `current_paper_may_21/`, `documents/`, `output/`,
  `anova_syco_analysis.py`, `ex.csv` — stale working copies, gitignored and untracked (keep them out of
  the index: `PAPER_MAY_21/` carries the author block). Overleaf is the canonical paper.
- **Process layer:** `.claude/agents/*.md` and `.claude/skills/verdict-integrity/` are shared and
  hash-pinned in `skills-lock.json`; `.claude/settings.local.json` stays local.
- **Anonymization:** no personal names, emails, or institution names in any tracked file outside
  `_markdown/` — code comments, filenames, and plot labels included. A prior ARR submission was
  desk-rejected over a leftover name in a markdown file and a plot filename.
