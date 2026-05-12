# LLM Judge Handoff Context - 2026-05-11

This document summarizes the new implementations added today around false-presupposition analysis, flip-rate inspection, and the NDIF/NNSight LLM-as-a-judge pipeline. It is intended as a handoff for continued work, especially on replacing heuristic flip labels with judge-based labels.

## What changed today

The work today had three main goals:

1. make the existing hidden-state metadata easier to analyze across models
2. reduce over-triggering from the original keyword-based flip labels
3. build an LLM judge pipeline that can label per-turn sycophantic concessions from saved metadata without rerunning any base model inference

No benchmark reruns are required for any of this. All analysis uses saved metadata and hidden-state artifacts already present under `data/`.

## Files added or changed and why

### New file: [flip_labeling.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/flip_labeling.py)

Why it was added:
- the old keyword lists were duplicated in multiple scripts
- those lists were too broad and over-labeled models like Gemma because generic apology language counted as a flip

What it does:
- centralizes one stricter heuristic in `match_flip_reason(...)` and `response_flipped(...)`
- counts explicit self-correction / concession language
- only counts apology language when paired with explicit self-error cues

Files now using it:
- [train_probes_v2.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/train_probes_v2.py)
- [probe_model_comparison.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/probe_model_comparison.py)
- [analysis/plot_flip_rates.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plot_flip_rates.py)

### Modified file: [train_probes_v2.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/train_probes_v2.py)

Why it changed:
- to stop using the old embedded keyword list
- to keep probe labels aligned with the shared stricter heuristic

What changed:
- removed the local `SYCOPHANCY_KEYWORDS`
- imported `response_flipped` from `flip_labeling.py`
- label extraction from metadata now uses the shared logic

### Modified file: [probe_model_comparison.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/probe_model_comparison.py)

Why it changed:
- same reason as `train_probes_v2.py`
- classifier comparisons should use the same labels as probe training

What changed:
- removed the local `SYCOPHANCY_KEYWORDS`
- imported `response_flipped` from `flip_labeling.py`

### New file: [analysis/plot_flip_rates.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plot_flip_rates.py)

Why it was added:
- to get an aggregate picture of how often each model appears to flip under the heuristic
- to compare `base`, `critical`, and `presupposition` question types across models

What it does:
- reads `*_multiturn_metadata.json` under `data/<model>/`
- computes:
  - `ever_flip_rate`: fraction of questions with at least one flipped pressured turn
  - `event_rate`: fraction of pressured turns labeled as flips
- supports either direct model folders or one nested subfolder
- writes:
  - [analysis/plots/flip_rate_summary.csv](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_summary.csv)
  - [analysis/plots/flip_rate_by_model_question_type.png](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_by_model_question_type.png)

### New file: [analysis/judge_flip_nnsight.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/judge_flip_nnsight.py)

Why it was added:
- the heuristic flip labels were not sufficient for true sycophancy labeling
- we needed a pipeline that judges each pressured turn from saved metadata using a stronger model, without rerunning the original benchmark models

What it does:
- reads saved `*_multiturn_metadata.json` files under `data/`
- uses NNsight + NDIF remote generation to ask an LLM judge whether each pressured turn is a sycophantic concession
- writes one CSV row per:
  - `model`
  - `question_type`
  - `question`
  - `turn`

Output schema:

```csv
model,question_type,question,turn,judgement,confidence,rationale
```

Key capabilities added:
- resume support against an existing CSV
- `--batch-mode turn` and `--batch-mode question`
- grouped question judging with one row still written per turn
- robust parsing of malformed judge outputs:
  - `json.loads`
  - `ast.literal_eval`
  - regex fallback
- prompt-echo protection by decoding only generated continuation
- NDIF OOM mitigation:
  - truncated prompt fields
  - lower `max_new_tokens`
  - `--turns-per-call`
  - automatic fallback to smaller turn groups on remote OOM
- logging failures to:
  - [analysis/judge_outputs/ndif_parse_failures.log](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/judge_outputs/ndif_parse_failures.log)

### Modified file: [.gitignore](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/.gitignore)

Why it changed:
- `data/` now contains local downloaded model artifacts and hidden-state bundles
- that folder is large enough that it should not be committed to GitHub
- current local size is on the order of tens of GB across multiple model artifact sets

What changed:
- added `/data/`

Important note:
- this does not delete or change local data
- it only prevents accidental Git tracking

## Data layout and current model sources

The local analysis scripts expect model outputs under `data/`.

Current intended folders:
- `data/DeepSeek-R1-Distill-Qwen-7B`
- `data/Llama-3.1-8B-Instruct`
- `data/Gemma-2-9B`
- `data/Qwen2.5-7B-Instruct`
- `data/Qwen3.5-9B`

Current practical state:
- DeepSeek: present
- Llama 3.1 8B: present
- Gemma 2 9B: present
- Qwen 3.5 9B: present
- Qwen 2.5 7B Instruct: empty, so analysis scripts skip it

Relevant Hugging Face dataset repos used today:
- DeepSeek artifacts: [sohampadianeu/multiturn-hidden-states](https://huggingface.co/datasets/sohampadianeu/multiturn-hidden-states)
- Llama artifacts: [sohampadianeu/llama31-sycophancy-artifacts](https://huggingface.co/datasets/sohampadianeu/llama31-sycophancy-artifacts)
- Gemma artifacts: [tomasdavola/sycon-bench-results-gemma](https://huggingface.co/datasets/tomasdavola/sycon-bench-results-gemma)
- Qwen 2.5 7B artifacts: [tomasdavola/Qwen2.5-7B-Instruct](https://huggingface.co/datasets/tomasdavola/Qwen2.5-7B-Instruct)
- Qwen 3.5 9B artifacts: [tomasdavola/Qwen3.5-9B-SYCO](https://huggingface.co/datasets/tomasdavola/Qwen3.5-9B-SYCO)

### Example download commands

Run from repo root:

```bash
hf download sohampadianeu/multiturn-hidden-states \
  --repo-type dataset \
  --local-dir data/DeepSeek-R1-Distill-Qwen-7B

hf download sohampadianeu/llama31-sycophancy-artifacts \
  --repo-type dataset \
  --local-dir data/Llama-3.1-8B-Instruct

hf download tomasdavola/sycon-bench-results-gemma \
  --repo-type dataset \
  --local-dir data/Gemma-2-9B

hf download tomasdavola/Qwen3.5-9B-SYCO \
  --repo-type dataset \
  --local-dir data/Qwen3.5-9B
```

## Why the heuristic path was not enough

The shared heuristic is cleaner than the original keyword list, but it still mostly measures concession / self-correction language, not ground-truth sycophancy.

Observed issue:
- apology-heavy or self-correcting models still get very high rates
- Gemma and Qwen 3.5 remained especially high even after tightening the heuristic

Manual inspection showed that many flagged cases were explicit self-corrections like:
- `I gave inaccurate information`
- `my previous explanation was wrong`
- `you are correct`

These can reflect either:
- true pressured concession
- or legitimate repair of an earlier bad answer

That is why the LLM judge path was added.

## Current flip-rate heuristic outputs

The current aggregate heuristic results are in:
- [analysis/plots/flip_rate_summary.csv](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_summary.csv)
- [analysis/plots/flip_rate_by_model_question_type.png](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_by_model_question_type.png)

Current summary snapshot:

- DeepSeek-R1-Distill-Qwen-7B:
  - base `0.3807`
  - critical `0.3095`
  - presupposition `0.4167`
- Llama-3.1-8B-Instruct:
  - base `0.6250`
  - critical `0.6579`
  - presupposition `0.5890`
- Gemma-2-9B:
  - base `0.9830`
  - critical `0.9773`
  - presupposition `0.9716`
- Qwen3.5-9B:
  - base `0.8929`
  - critical `0.9196`
  - presupposition `0.7589`

Interpretation:
- useful as a concession-style heuristic summary
- not reliable enough as final sycophancy ground truth

## Plotting command used

This command was used because local Matplotlib config/cache needed writable project-local directories:

```bash
mkdir -p .mplconfig .cache/fontconfig analysis/plots
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" ./.venv/bin/python analysis/plot_flip_rates.py
```

## Manual inspection of a single question

For manual judging, the correct file to inspect is the metadata JSON, not the CSV.

Example file:
- [data/DeepSeek-R1-Distill-Qwen-7B/base_multiturn_metadata.json](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/data/DeepSeek-R1-Distill-Qwen-7B/base_multiturn_metadata.json)

Reason:
- the metadata JSON contains:
  - the original question as the key
  - `Turn_0` through `Turn_5`
  - `pressure_prompt`
  - `assistant_response`

Example command to print one full question trajectory:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/DeepSeek-R1-Distill-Qwen-7B/base_multiturn_metadata.json")
question = "What actually happens when we run out of IPv4 Addresses?"

data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data[question], indent=2, ensure_ascii=False))
PY
```

## LLM judge design and current status

### Judge model setup

The judge pipeline uses NNsight remote execution via NDIF. The large judge model does not run locally on the laptop when `remote=True`; it runs on NDIF infrastructure.

Prerequisites:

```bash
./.venv/bin/pip install nnsight
export NDIF_API_KEY=...
```

### Basic judge command

The script supports any NDIF-accessible judge model, but testing today used:
- `meta-llama/Llama-3.1-70B-Instruct`

Safe sample command:

```bash
./.venv/bin/python analysis/judge_flip_nnsight.py \
  --judge-model meta-llama/Llama-3.1-70B-Instruct \
  --batch-mode question \
  --turns-per-call 2 \
  --max-turn0-chars 900 \
  --max-response-chars 700 \
  --max-new-tokens 320 \
  --output analysis/judge_outputs/ndif_judgements_sample.csv \
  --limit 20
```

Recommended full run command:

```bash
./.venv/bin/python analysis/judge_flip_nnsight.py \
  --judge-model meta-llama/Llama-3.1-70B-Instruct \
  --batch-mode question \
  --turns-per-call 2 \
  --max-turn0-chars 900 \
  --max-response-chars 700 \
  --max-new-tokens 320 \
  --output analysis/judge_outputs/ndif_judgements.csv
```

### Why question batching was modified

Initial grouped question batches with all five pressured turns caused NDIF OOM on the 70B judge. The logged failure is in:
- [analysis/judge_outputs/ndif_parse_failures.log](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/judge_outputs/ndif_parse_failures.log)

Mitigations added in code:
- lower default `--max-new-tokens`
- truncate `turn0_response`, `previous_response`, `current_response`, and `pressure_prompt`
- batch only `2` pressured turns per call by default
- recursively split grouped calls when NDIF returns CUDA OOM

### Parsing fixes added

The judge script also had two important implementation fixes:

1. malformed JSON handling
- added layered parsing and retries

2. prompt echo bug
- NDIF responses were initially parsed from echoed prompt text, producing fake outputs like `judgement=True confidence=0`
- fixed by decoding only the generated continuation and extracting the last JSON object

### Current methodology note

The current judge design asks whether each pressured turn is a sycophantic concession rather than a reasonable self-correction.

Important caveat:
- grouped question mode can still create some cross-turn context leakage because a judge chunk can see multiple pressured turns together
- if stricter methodological isolation is needed later, a prefix-mode or pure per-turn mode should be considered

## Commands used for validation

Compile checks:

```bash
python -m py_compile analysis/judge_flip_nnsight.py
```

Dry-run grouped prompt preview:

```bash
./.venv/bin/python analysis/judge_flip_nnsight.py \
  --dry-run \
  --batch-mode question \
  --turns-per-call 2 \
  --limit 1
```

## What exists locally now

Code and generated analysis files currently present:
- [flip_labeling.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/flip_labeling.py)
- [analysis/plot_flip_rates.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plot_flip_rates.py)
- [analysis/judge_flip_nnsight.py](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/judge_flip_nnsight.py)
- [analysis/plots/flip_rate_summary.csv](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_summary.csv)
- [analysis/plots/flip_rate_by_model_question_type.png](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/plots/flip_rate_by_model_question_type.png)
- [analysis/judge_outputs/ndif_parse_failures.log](/Users/sohampadia/workspace/CS6120_SY/CS6120-Sycophancy-Detection/analysis/judge_outputs/ndif_parse_failures.log)

At the time of writing:
- the sample NDIF output CSV is not currently present in the repo tree
- the fail log still contains earlier pre-fix OOM traces and should not be treated as the current behavior of the patched script

## Recommended next steps for the teammate taking over the judge

1. rerun a small sample using the patched `analysis/judge_flip_nnsight.py`
2. manually inspect several judged questions against the metadata JSON
3. decide whether grouped question chunks are acceptable methodologically
4. if not, add prefix-mode judging so turn `k` only sees turns `0..k`
5. once judge quality is acceptable, produce the full `ndif_judgements.csv`
6. update downstream probe training to use judge labels instead of heuristic flip labels

## Bottom line

Today’s work established:
- a shared stricter heuristic labeler
- aggregate flip-rate analysis across available model datasets
- a working NDIF/NNSight judge pipeline for per-turn labels
- practical prompt-size and OOM mitigation for the 70B judge
- local-only handling for large downloaded artifact bundles via `/data/` in `.gitignore`
