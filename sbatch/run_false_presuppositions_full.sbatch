#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=fp_full
#SBATCH --chdir=/home/padia.so/Workspace/CS6120-Sycophancy-Detection
#SBATCH --output=/home/padia.so/Workspace/CS6120-Sycophancy-Detection/logs/%x-%j.out
#SBATCH --error=/home/padia.so/Workspace/CS6120-Sycophancy-Detection/logs/%x-%j.err

set -euo pipefail

mkdir -p /home/padia.so/Workspace/CS6120-Sycophancy-Detection/logs

module load python/3.13.5
source /home/padia.so/Workspace/CS6120-Sycophancy-Detection/.venv313/bin/activate

export HF_HOME="${HF_HOME:-/home/padia.so/.cache/huggingface}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

MODEL_NAME="${1:-Qwen/Qwen3.5-9B}"
PROMPT_TYPE="${PROMPT_TYPE:-all}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "Model: ${MODEL_NAME}"
echo "Prompt type: ${PROMPT_TYPE}"
echo "Batch size: ${BATCH_SIZE}"
echo "HF cache: ${HF_HOME}"

nvidia-smi

CMD=(
  python
  false-presuppositions-setting/run_benchmark.py
  "${MODEL_NAME}"
  --prompt_type "${PROMPT_TYPE}"
  --batch_size "${BATCH_SIZE}"
  --verbose
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
