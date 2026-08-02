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
  Anonymous ONLY. `_markdown/`, `_local/`, `docs/`, `.claude/`, `_falsifier/`, and this file
  never enter any anonymized export.

## Key facts to not re-break
- base/critical/presupposition are SYSTEM-PROMPT CONDITIONS over the SAME ~177 questions —
  never "question types" (that misdescription shipped once).
- Pre-flip pairing: state at turn t predicts label at t+1, censored at/after the first flip.
  Reference: `train_probes_v2.py::build_preflip_dataset`. Three scripts once used at-flip
  pairing (the off-by-one trap — `_markdown/ONBOARDING.md`).
- Peak cosine AUCs (0.596–0.636) are layer-sweep-selected: descriptive only. Under the
  sweep-corrected permutation test only Qwen2.5-base survives (p=0.035); Fisher-combined
  p=0.047 with condition choice uncorrected. Never cite individual peaks as significant.
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
