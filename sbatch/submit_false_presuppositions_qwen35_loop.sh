#!/bin/bash
set -euo pipefail

ROOT="/home/padia.so/Workspace/CS6120-Sycophancy-Detection"
JOB_FILE="${ROOT}/sbatch/false_presuppositions_qwen35_loop.sbatch"

MODEL_NAME="${1:-Qwen/Qwen3.5-9B}"
PROMPT_TYPE="${PROMPT_TYPE:-all}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES_PER_JOB="${MAX_BATCHES_PER_JOB:-8}"
RUN_REPORT="${RUN_REPORT:-1}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
PROMPT_LIST="${PROMPT_LIST:-base,critical,presupposition}"

MODEL_ID="${MODEL_NAME##*/}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/false-presuppositions-setting/output/${MODEL_ID}}"

echo "Submitting ${JOB_FILE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "PROMPT_TYPE=${PROMPT_TYPE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_BATCHES_PER_JOB=${MAX_BATCHES_PER_JOB}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

sbatch --export=ALL,MODEL_NAME="${MODEL_NAME}",PROMPT_TYPE="${PROMPT_TYPE}",BATCH_SIZE="${BATCH_SIZE}",MAX_BATCHES_PER_JOB="${MAX_BATCHES_PER_JOB}",RUN_REPORT="${RUN_REPORT}",AUTO_RESUBMIT="${AUTO_RESUBMIT}",PROMPT_LIST="${PROMPT_LIST}",OUTPUT_DIR="${OUTPUT_DIR}" "${JOB_FILE}"
