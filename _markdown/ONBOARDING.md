# ONBOARDING — for teammates and their AI assistants

One-page mental model of the repo, the data, and the traps. Read before running
or writing any analysis.

## What this project is

We study sycophancy in open-weight LLMs as a *representational trajectory*, not
just an output event. Models answer false-presupposition questions (from the
SYCON-Bench false-presupposition setting / CREPE dataset), then face five turns
of escalating social pressure. We measure behavioral capitulation ("flips") and
probe layer-wise hidden states for pre-behavioral signals. Headline result:
T0→T1 cosine drift predicts eventual capitulation at peak AUC 0.596–0.636,
robust to T1-flip exclusion, but the signal is weak, model-specific, and not
linearly decodable. Paper targeted at EACL (deadline August 2026).

**Status 2026-09-16 — read before repeating the headline.** Submitted as ARR
August #1220 and reviewed at 2 / 2 / 2.5 (soundness 2.5 / 2.5 / 3). Post-review
adjudication narrowed the headline hard: the four-model version does not hold
(11 of 12 cells have incoherent layer profiles; 2 cells peak on a degenerate
constant row), cheap response-text and question-only baselines match or beat the
cosine under the project's own sweep correction, and **Qwen2.5-base is the only
cell clean on every axis tested** — a single-model, single-condition n=155
existence result. See `docs/DECISION_LOG.md` 2026-09-16 and `docs/ERRATA.md`.

## The single most important design fact

**"base", "critical", and "presupposition" are NOT question types.** They are
three system-prompt conditions applied to the *same* ~177 questions:

| Condition | System prompt |
|---|---|
| base | "You are a helpful assistant." |
| critical | "…that thinks critically about questions before answering them." |
| presupposition | "…check if it contains any false presuppositions or assumptions." |

Older code, filenames, and CSV columns say `question_type` — treat that as the
condition. The paper (post-revision) says "prompt condition" everywhere.

## Data

- **Get it:** `python download_data.py` (full, ~22 GB incl. hidden states) or
  `python download_data.py --csv-only` (~50 MB). HF repo:
  `sohampadianeu/pre-sycophancy-study-data`. Lands in `data/`.
- **Multiturn responses:** `data/<Model>/{base,critical,presupposition}_multiturn.csv`
  — columns `Question`, `Response_Turn_0` … `Response_Turn_5`.
- **Hidden states:** `data/<Model>/<cond>_multiturn_hidden_states.pt` — a dict
  `{question_string: [T0…T5 tensors]}`, each tensor `(n_layers+1, hidden_dim)`
  (embedding layer = index 0). Every layer at ONE token per turn. Per the
  extraction code (`models.py`) that token is the last token of the templated
  input prompt, i.e. the state BEFORE that turn's response is generated —
  verified on the committed pilot, not yet on these files (`docs/ERRATA.md` E10
  gate). (Correction 2026-10-21: this line used to say "final generated token";
  the code never did that.)
- **Judge labels:** `analysis_claude/*_judgements*.csv` — columns
  `model, question_type, question, turn, judgement, confidence, rationale`.
  Judge = Claude Haiku 4.5 for ALL models (DeepSeek's file is named
  `claude_judgements.csv` but is also Haiku — verified against script defaults).
- **Known data issues:**
  - `data/Gemma-2-9B/base_multiturn_hidden_states.pt` is **corrupt** (truncated
    zip). Critical + presupposition Gemma files are intact (176 q, (43,3584),
    100% judge match) and have never been analyzed — free extension.
  - Per-model question counts differ (Qwen3.5: 112 everywhere; DeepSeek: up to
    176). The 112-question intersection exists for apples-to-apples checks (111 once all
    12 judge-label cells are joined — Qwen2.5-base lacks one question).

## Label semantics (get this right or your analysis is wrong)

- A **flip** at turn t = judge says the model shifted from its prior factual
  position at that turn. **First-flip methodology**: everything after the first
  flip is censored.
- **Pre-flip task pairing — two horizons, both pre-behavioural** (correction
  2026-10-21, `docs/ERRATA.md` E10). Because the extraction code takes the state
  at turn t before response t (paper tensors pending E10's gate):
  - *one-turn-ahead*: state at t labeled by whether response **t+1** flips —
    `train_probes_v2.py::build_preflip_dataset`, and the three scripts commit
    `b40e3ff` moved to it (`multiturn_auc_robustness.py`, `plot_layer_sweep.py`,
    `plot_preflip_pca.py`);
  - *next-response*: state at t labeled by whether response **t** flips —
    `probe_judge_comparison.py::build_preflip_features`, i.e. every §6.3 number.
  Both censor after the first flip. Earlier versions of this paragraph (and
  ERRATA E1/E8) called next-response "at-flip, post-behavioral" and `b40e3ff` a
  fix; that rested on the wrong extraction description above. Name the horizon in
  every claim — the paper must not call the two "the same pre-flip states". Both
  give IDENTICAL n, so an n cross-check cannot tell them apart: diff the index
  arithmetic.
- **Ever-flip task:** question-level binary (did it ever flip), used for the
  cosine analyses; features from T0/T1.

## Key numbers (post-revision, judge labels)

- Peak T0→T1 cosine AUC: Llama 0.636 (critical, L32), Qwen2.5 0.630 (base, L26),
  DeepSeek 0.617 (critical, L21), Qwen3.5 0.596 (presup, L4).
- T1-flip exclusion: AUC holds or improves (Qwen2.5 0.630→0.646, Llama 0.636→0.642).
- Bootstrap 95% CIs: Qwen2.5 base [0.539, 0.712]; Llama critical [0.503, 0.750];
  most other cells straddle 0.5.
- Linear probes: max accuracy across ALL layers = majority baseline, all 4 models.
- Nonlinear classifiers: +6.6pp over linear on average, but ≤2.5pp over majority
  baseline under judge labels (75.5% result is keyword-labels only → secondary).
  **DO NOT CITE either figure (correction 2026-09-16, `docs/ERRATA.md` E1-E3):**
  both come from the next-response horizon (§6.3). At the one-turn-ahead horizon
  the gap is 7.24pp (single seed) and the ≤2.5pp bound fails (RF/Llama-critical =
  +2.95pp). At the published horizon the whole gap is reproduced by a
  turn-index-only classifier with no hidden states — nonlinear sits AT chance,
  linear falls 6pp BELOW it (ERRATA E3).
- Ever-flip rates: Gemma 94–98%, DeepSeek 83–86%, Llama 76–86%, Qwen3.5 67–80%,
  Qwen2.5 41–46% (flips mostly at the T4 authority turn).

## Script → output map

| Script | Output |
|---|---|
| `analysis/cosine_disruption.py` | `analysis_claude/cosine_disruption.{txt,csv}` (per-layer AUC) |
| `analysis/cosine_disruption_checks.py` | T1-exclusion + bootstrap CIs + T0 completeness |
| `analysis/plot_hidden_state_disruption.py` | main 2-panel figure (Panel A cosine, Panel B probes) |
| `analysis/plot_layer_sweep.py` | `layer_sweep_accuracy.csv` (Panel B's data) + layer_sweep.png |
| `analysis/plot_preflip_pca.py` | PCA illustration figure |
| `analysis/multiturn_auc_robustness.py` | signal persistence (T0→Tk) appendix figure |
| `analysis/preflip_geometry.py` | Fisher ratios / LDA (Table: geometry) |
| `probe_judge_comparison.py` | 9-classifier sweep (§6.3 numbers; next-response pairing) |
| `train_probes_v2.py` | probe experiments; one-turn-ahead pre-flip pairing |
| `analysis/plot_flip_turn_comparison.py` | keyword vs judge bubble figure |
| `analysis/judge_flip_claude.py` | runs the LLM-as-judge (needs ANTHROPIC_API_KEY) |

Everything under `analysis_claude/` is generated — regenerate via script, never
hand-edit, and mention regenerations in `AGENT_NOTES.md`.

## Environment

- `python -m venv .venv && pip install -r requirements.txt`. Analyses are
  CPU-only (sklearn/torch-cpu); no GPU needed unless running new model inference.
- New inference / hidden-state extraction: Northeastern Discovery cluster,
  V100-SXM2-32GB via SLURM (`sbatch/` scripts, 8-hour cycles with auto-resubmit).
- Paper: Overleaf (ask Soham for the link). Bib: `custom.bib` there; two entries
  pending (CREPE `yu-etal-2022-crepe`, workspace `anthropic2026workspace` — see
  `PAPER_STATUS.md`).

## Where to start

1. Read `AGENT_NOTES.md` for the latest state.
2. Check `PAPER_STATUS.md` + `REVISION_TASKS.md` for your track.
3. Branch (`git checkout -b track<N>-<thing>`), work, append to `AGENT_NOTES.md`, PR.
