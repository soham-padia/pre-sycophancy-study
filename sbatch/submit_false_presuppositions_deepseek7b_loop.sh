#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional overrides:
#   PROMPT_TYPE=all|base|critical|presupposition
#   BATCH_SIZE=1
#   MAX_BATCHES_PER_JOB=8
#   AUTO_RESUBMIT=1
#   RUN_REPORT=1
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

bash "${SCRIPT_DIR}/submit_false_presuppositions_qwen35_loop.sh" "${MODEL_NAME}"
