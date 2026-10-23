#!/bin/bash
#SBATCH -A p2026_0037_neu
#SBATCH -p rtx-batch,rtx-devel,preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J heads_gen
#SBATCH -o /work/neu/p2026_0037_neu/heads_pilot/logs/%x-%A_%a.out
# No `set -u`: it kills the batch shell on this cluster.

source ~/startload.sh
cd /home/padia_so_neu/pre-sycophancy-study
echo "host=$(hostname) job=$SLURM_JOB_ID args=$*"
python heads_pilot/generate.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:-1}" "$@"
