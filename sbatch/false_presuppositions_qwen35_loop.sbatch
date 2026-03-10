#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=fp_qwen35_loop

set -euo pipefail

ROOT="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"

module load python/3.13.5
if [[ -f "${ROOT}/.venv313/bin/activate" ]]; then
  source "${ROOT}/.venv313/bin/activate"
else
  source "${ROOT}/.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
mkdir -p "${HF_HOME}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
MODEL_ID="${MODEL_NAME##*/}"
PROMPT_TYPE="${PROMPT_TYPE:-all}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES_PER_JOB="${MAX_BATCHES_PER_JOB:-8}"
RUN_REPORT="${RUN_REPORT:-1}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
PROMPT_LIST="${PROMPT_LIST:-base,critical,presupposition}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/false-presuppositions-setting/output/${MODEL_ID}}"

echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "Model: ${MODEL_NAME}"
echo "Prompt type: ${PROMPT_TYPE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max batches/job: ${MAX_BATCHES_PER_JOB}"
echo "Output dir: ${OUTPUT_DIR}"
echo "HF cache: ${HF_HOME}"
nvidia-smi || true

python false-presuppositions-setting/run_benchmark.py \
  "${MODEL_NAME}" \
  --prompt_type "${PROMPT_TYPE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_batches "${MAX_BATCHES_PER_JOB}" \
  --output_dir "${OUTPUT_DIR}" \
  --verbose

export ROOT OUTPUT_DIR PROMPT_LIST

STATUS_JSON="$(python - <<'PY'
import csv
import json
import os

root = os.environ["ROOT"]
output_dir = os.environ["OUTPUT_DIR"]
prompt_list = [p.strip() for p in os.environ.get("PROMPT_LIST", "base,critical,presupposition").split(",") if p.strip()]
data_dir = os.path.join(root, "false-presuppositions-setting", "data")

status = {"complete": True, "details": {}}

for prompt in prompt_list:
    expected_file = os.path.join(data_dir, f"{prompt}_multiturn.csv")
    output_file = os.path.join(output_dir, f"{prompt}_multiturn.csv")

    expected = 0
    if os.path.exists(expected_file):
        with open(expected_file, "r", encoding="utf-8") as f:
            expected = sum(1 for _ in csv.DictReader(f))

    completed = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if all((row.get(f"Response_Turn_{i}") or "").strip() for i in range(6)):
                    completed += 1

    done = expected > 0 and completed >= expected
    status["details"][prompt] = {
        "expected": expected,
        "completed": completed,
        "done": done,
    }
    if not done:
        status["complete"] = False

print(json.dumps(status))
PY
)"

echo "Completion status: ${STATUS_JSON}"
export STATUS_JSON

IS_COMPLETE="$(python - <<'PY'
import json
import os
status = json.loads(os.environ["STATUS_JSON"])
print("1" if status.get("complete") else "0")
PY
)"

if [[ "${IS_COMPLETE}" == "1" ]]; then
  echo "All prompt files are complete."
  if [[ "${RUN_REPORT}" == "1" ]]; then
    REPORT_TXT="${OUTPUT_DIR}/report_output_analysis.txt"
    python false-presuppositions-setting/report_output_analysis.py "${OUTPUT_DIR}" --model-name "${MODEL_ID}" > "${REPORT_TXT}" || true
    echo "Report written to ${REPORT_TXT}"
  fi
  exit 0
fi

echo "Run incomplete."
if [[ "${AUTO_RESUBMIT}" == "1" ]]; then
  echo "Re-submitting next chunk..."
  sbatch \
    --chdir="${ROOT}" \
    --output="${ROOT}/logs/%x-%j.out" \
    --error="${ROOT}/logs/%x-%j.err" \
    --export=ALL,ROOT_DIR="${ROOT}",MODEL_NAME="${MODEL_NAME}",PROMPT_TYPE="${PROMPT_TYPE}",BATCH_SIZE="${BATCH_SIZE}",MAX_BATCHES_PER_JOB="${MAX_BATCHES_PER_JOB}",RUN_REPORT="${RUN_REPORT}",AUTO_RESUBMIT="${AUTO_RESUBMIT}",PROMPT_LIST="${PROMPT_LIST}",OUTPUT_DIR="${OUTPUT_DIR}" \
    "${ROOT}/sbatch/false_presuppositions_qwen35_loop.sbatch"
else
  echo "AUTO_RESUBMIT=0, stopping."
fi
