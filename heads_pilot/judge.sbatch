#!/bin/bash
#SBATCH -A p2026_0037_neu
#SBATCH -p preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:b200:2
#SBATCH -c 16
#SBATCH --mem=320G
#SBATCH -t 04:00:00
#SBATCH -J heads_judge
#SBATCH -o /work/neu/p2026_0037_neu/heads_pilot/logs/%x-%A_%a.out
# No `set -u`: it kills the batch shell on this cluster.

source ~/startload.sh
cd /home/padia_so_neu/pre-sycophancy-study
echo "host=$(hostname) job=$SLURM_JOB_ID array=${SLURM_ARRAY_TASK_ID:-none} args=$*"
python heads_pilot/judge.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:-1}" "$@"
