#!/bin/bash
# Submit the whole heads pilot as one Slurm dependency chain. Every stage starts only if all of its
# inputs finished successfully (afterok); a failure cancels everything downstream (--kill-on-invalid-dep).
# Every text-generating stage verifies its own generation first (safe_gen.py) and fails if it cannot.
# Usage: bash heads_pilot/run_chain.sh      (prints the job ids; logs in /work/.../heads_pilot/logs)

cd /home/padia_so_neu/pre-sycophancy-study
G=/work/neu/p2026_0037_neu/heads_pilot/Qwen2.5-7B-Instruct/base
K=16
CONDS="baseline sel_off sel_up rand_off rand_up"
DEP="--kill-on-invalid-dep=yes"
sub() { sbatch --parsable $DEP "$@"; }

GEN=$(sub -p preemptable --requeue -t 04:00:00 --array=0-15 --export=ALL,N_SHARDS=16 heads_pilot/generate.sbatch)
CAP=$(sub --dependency=afterok:$GEN -J heads_capture heads_pilot/gpu.sbatch heads_pilot/capture_heads.py)
JUD=$(sub --dependency=afterok:$GEN -t 06:00:00 --array=0-7 --export=ALL,N_SHARDS=8 heads_pilot/judge.sbatch)
JT0=$(sub --dependency=afterok:$GEN -t 03:00:00 --array=0-1 --export=ALL,N_SHARDS=2 heads_pilot/judge.sbatch --mode t0 --arms disagree)
RANK=$(sub --dependency=afterok:$CAP:$JUD -J heads_rank heads_pilot/cpu.sbatch heads_pilot/rank_heads.py)
STEER=$(sub --dependency=afterok:$RANK -t 04:00:00 --array=0-39 --export=ALL,N_SHARDS=8 heads_pilot/steer.sbatch --k $K)
ST0=$(sub --dependency=afterok:$RANK -t 02:00:00 --array=0-9 --export=ALL,N_SHARDS=2 heads_pilot/steer.sbatch --k $K --from-turn 0 --turns 1 --arms disagree)
FINAL=""
for c in $CONDS; do
  j=$(sub --dependency=afterok:$STEER -t 08:00:00 --array=0-3 --export=ALL,N_SHARDS=4 heads_pilot/judge.sbatch --gen-dir $G/steer/k$K/$c)
  t=$(sub --dependency=afterok:$ST0 -t 02:00:00 --export=ALL,N_SHARDS=1 heads_pilot/judge.sbatch --mode t0 --arms disagree --gen-dir $G/steer/k${K}_t0/$c)
  FINAL="$FINAL:$j:$t"
done
SUM=$(sub --dependency=afterok$FINAL -J heads_summary heads_pilot/cpu.sbatch heads_pilot/summarize.py --k $K)
echo "generate=$GEN capture=$CAP judge=$JUD judge_t0=$JT0 rank=$RANK steer=$STEER steer_t0=$ST0 summary=$SUM"
