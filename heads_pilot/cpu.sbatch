#!/bin/bash
#SBATCH -A p2026_0037_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J heads_cpu
#SBATCH -o /work/neu/p2026_0037_neu/heads_pilot/logs/%x-%j.out
# Runs one CPU stage of the heads pilot: `sbatch cpu.sbatch heads_pilot/<script>.py [args]`.
# No `set -u`: it kills the batch shell on this cluster.

source ~/startload.sh
cd /home/padia_so_neu/pre-sycophancy-study
echo "host=$(hostname) job=$SLURM_JOB_ID args=$*"
python "$@"
