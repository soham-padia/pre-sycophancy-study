#!/bin/bash
#SBATCH -A p2026_0037_neu
#SBATCH -p rtx-batch,rtx-devel,preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J heads_gpu
#SBATCH -o /work/neu/p2026_0037_neu/heads_pilot/logs/%x-%j.out
# Runs one single-GPU stage of the heads pilot: `sbatch gpu.sbatch heads_pilot/<script>.py [args]`.
# No `set -u`: it kills the batch shell on this cluster.

source ~/startload.sh
cd /home/padia_so_neu/pre-sycophancy-study
echo "host=$(hostname) job=$SLURM_JOB_ID args=$*"
python "$@"
