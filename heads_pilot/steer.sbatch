#!/bin/bash
#SBATCH -A p2026_0037_neu
#SBATCH -p rtx-batch,rtx-devel,preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J heads_steer
#SBATCH -o /work/neu/p2026_0037_neu/heads_pilot/logs/%x-%A_%a.out
# No `set -u`: it kills the batch shell on this cluster.
# Array ids 0..(5 conditions x N_SHARDS - 1); see CONDITIONS in steer.py.

source ~/startload.sh
cd /home/padia_so_neu/pre-sycophancy-study
echo "host=$(hostname) job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID} args=$*"
python heads_pilot/steer.py --task "${SLURM_ARRAY_TASK_ID}" --n-shards "${N_SHARDS:-8}" "$@"
