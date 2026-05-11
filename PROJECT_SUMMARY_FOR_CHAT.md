# Sycophancy Detection Project Summary (Portable Context)

## Project scope
This project extends SYCON-Bench-style multi-turn sycophancy evaluation, with the main implementation in:
- `false-presuppositions-setting/`

Other benchmark folders exist (`debate_setting/`, `ethical-setting/`), but most active work here is on false-presupposition multi-turn pressure testing plus hidden-state logging.

## End goal
Build a practical anti-sycophancy pipeline:
1. Generate multi-turn pressured dialogues.
2. Capture model internals at each turn.
3. Train probes to predict flips before they happen.
4. Add guardrails that reduce sycophantic flips while preserving legitimate corrections.

Core idea:
- Output behavior + internal states + trajectory metadata -> predictive detection -> intervention.

## Proposal framing (from local proposal PDF)
Main hypotheses:
1. Hidden-state precursors exist before output flips.
2. Internal confidence degrades across escalating turns (not always a single abrupt collapse).
3. Simple linear probes can predict next-turn flips above chance.

Pilot observation reported:
- Single-turn embedded pressure is relatively robust.
- Multi-turn committed setting collapses much faster (especially small models), making early-turn detection critical.

## What was implemented

### 1) Benchmark runner upgrades
File: `false-presuppositions-setting/run_benchmark.py`
- Added `--max_batches` for quick local tests and chunked cluster runs.
- Fixed pressure turn handling so turns 1-5 use pressure-only user prompts while conversation history carries prior context.
- Made defaults path-safe (script-directory-based data/output paths).
- Added resume support:
  - Reads existing `*_multiturn.csv`.
  - Skips questions already complete (all 6 turn responses present).
  - Enables reliable stop/resume and Slurm chunk looping.

### 2) Hidden-state extraction upgrades
File: `false-presuppositions-setting/models.py`
- `generate_responses(...)` now returns structured output with:
  - `text`
  - `hidden_state`
  - `hidden_state_type`
- Current hidden-state mode:
  - **all layers, last token** (including embedding layer),
  - stored as `float16` on CPU for manageable size,
  - tensor shape example for Qwen2.5-0.5B: `(25, 896)`.

### 3) New output artifacts
Per prompt type (`base`, `critical`, `presupposition`), outputs now include:
- `*_multiturn.csv`: response text table.
- `*_multiturn_hidden_states.json`: vector JSON (large).
- `*_multiturn_hidden_states_summary.json`: compact numeric stats.
- `*_multiturn_hidden_states.pt`: full tensor states (preferred for training).
- `*_multiturn_metadata.json`: turn-level context:
  - prompt type,
  - turn index,
  - per-turn user prompt,
  - full `input_messages`,
  - assistant response.

## Data format notes (important)
- One question has 6 turns total:
  - turn 0 base,
  - turns 1-5 pressure.
- `.pt` schema:
  - top-level dict keyed by question text,
  - value is list of 6 entries (one per turn),
  - each entry is tensor `(layers, hidden_dim)` or `None`.
- Example counting:
  - `batch_size=3`, `max_batches=2` -> 6 questions.
  - all-turn vectors = 6 x 6 = 36.
  - pressure-only vectors = 6 x 5 = 30.

## HPC/Open OnDemand automation
Files added/used in `sbatch/`:
- `false_presuppositions_qwen35_loop.sbatch`
- `submit_false_presuppositions_qwen35_loop.sh`
- `submit_false_presuppositions_deepseek7b_loop.sh`

Capabilities:
- Chunked benchmark execution per Slurm job (`MAX_BATCHES_PER_JOB`).
- Completion check based on produced CSVs.
- Auto-resubmit until all prompts are complete.
- Optional final output quality report generation.
- Path handling made dynamic (portable across users/folders).

## Commands used most often

### Local smoke test
```bash
python false-presuppositions-setting/run_benchmark.py "Qwen/Qwen2.5-0.5B-Instruct" \
  --prompt_type base \
  --batch_size 1 \
  --max_batches 1 \
  --verbose
```

### Direct Slurm submit (no wrapper required)
```bash
sbatch \
  --chdir="$PWD" \
  --output="$PWD/logs/%x-%j.out" \
  --error="$PWD/logs/%x-%j.err" \
  --export=ALL,ROOT_DIR="$PWD",MODEL_NAME="Qwen/Qwen3.5-9B",PROMPT_TYPE=all,BATCH_SIZE=1,MAX_BATCHES_PER_JOB=8,AUTO_RESUBMIT=1,RUN_REPORT=1 \
  sbatch/false_presuppositions_qwen35_loop.sbatch
```

### Model-specific wrapper example (DeepSeek)
```bash
bash sbatch/submit_false_presuppositions_deepseek7b_loop.sh
```

## Practical constraints encountered
- Newer model families may need newer Python/transformers combinations.
- JSON vectors get huge quickly for larger models; `.pt` is the training source of truth.
- On shared HPC `/home`, storage pressure is real; use scratch HF cache:
  - `export HF_HOME=/scratch/$USER/hf_cache`
- `git pull` can fail due to untracked `__pycache__` artifacts; remove only cache files, not logs/outputs.

## Recommended probe-training inputs
Use this trio:
- `*_multiturn_hidden_states.pt` (features),
- `*_multiturn_metadata.json` (trajectory + labels context),
- `*_multiturn.csv` (text sanity checks).

This is the current foundation for training flip predictors and implementing anti-sycophancy guardrails.

## Reference papers used
- [Measuring Sycophancy of Language Models in Multi-turn Dialogues (arXiv:2505.23840)](https://arxiv.org/abs/2505.23840)
- [Accounting for Sycophancy in Language Model Uncertainty Estimation (Findings NAACL 2025)](https://aclanthology.org/2025.findings-naacl.438/)
- [BASIL: Bayesian Assessment of Sycophancy in LLMs (arXiv:2508.16846)](https://arxiv.org/abs/2508.16846)
