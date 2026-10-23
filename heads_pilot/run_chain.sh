#!/bin/bash
# Submit the whole heads pilot as one Slurm dependency chain. Every stage starts only if all of its
# inputs finished successfully (afterok); a failure cancels everything downstream (--kill-on-invalid-dep).
# Every text-generating stage verifies its own generation first (safe_gen.py) and fails if it cannot.
# One-GPU stages (generate, capture, steer) are pinned to RTX PRO 6000 in their sbatch headers, so every
# comparison stays on one GPU type (fp32 output can differ across types). The 70B judge runs on 4 x RTX PRO
# 6000 (judge.sbatch): its queue is days shorter than 2 x B200, at 4 instead of 6 fairshare units per hour.
# Steering decodes exactly as stage 1 did (it reads stage 1's run_meta files).
#
# Usage: bash heads_pilot/run_chain.sh <new-output-root> [generate.py options, e.g. --repetition-penalty 1.2]
# The root must not exist yet: the judge skips conversations already labelled in its folder, so reusing
# a folder would attach old labels to new text. Prints the job ids; logs in /work/.../heads_pilot/logs.

cd /home/padia_so_neu/pre-sycophancy-study
OUT=$1
[ -n "$OUT" ] || { echo "usage: $0 <new-output-root> [generate.py options]"; exit 1; }
[ -e "$OUT" ] && { echo "refusing: $OUT already exists"; exit 1; }
shift
G=$OUT/Qwen2.5-7B-Instruct/base
K=16
CONDS="baseline sel_off sel_up rand_off rand_up"
DEP="--kill-on-invalid-dep=yes"
sub() { sbatch --parsable $DEP "$@"; }

GEN=$(sub -t 04:00:00 --array=0-15 --export=ALL,N_SHARDS=16 heads_pilot/generate.sbatch --out "$OUT" "$@")
CAP=$(sub --dependency=afterok:$GEN -J heads_capture heads_pilot/gpu.sbatch heads_pilot/capture_heads.py --gen-dir $G)
JUD=$(sub --dependency=afterok:$GEN -t 06:00:00 --array=0-7 --export=ALL,N_SHARDS=8 heads_pilot/judge.sbatch --gen-dir $G)
JT0=$(sub --dependency=afterok:$GEN -t 03:00:00 --array=0-1 --export=ALL,N_SHARDS=2 heads_pilot/judge.sbatch --mode t0 --arms disagree --gen-dir $G)
RANK=$(sub --dependency=afterok:$CAP:$JUD -J heads_rank heads_pilot/cpu.sbatch heads_pilot/rank_heads.py --gen-dir $G)
STEER=$(sub --dependency=afterok:$RANK -t 04:00:00 --array=0-39 --export=ALL,N_SHARDS=8 heads_pilot/steer.sbatch --k $K --gen-dir $G)
ST0=$(sub --dependency=afterok:$RANK -t 02:00:00 --array=0-9 --export=ALL,N_SHARDS=2 heads_pilot/steer.sbatch --k $K --from-turn 0 --turns 1 --arms disagree --gen-dir $G)
FINAL=""
for c in $CONDS; do
  j=$(sub --dependency=afterok:$STEER -t 08:00:00 --array=0-3 --export=ALL,N_SHARDS=4 heads_pilot/judge.sbatch --gen-dir $G/steer/k$K/$c)
  t=$(sub --dependency=afterok:$ST0 -t 02:00:00 --export=ALL,N_SHARDS=1 heads_pilot/judge.sbatch --mode t0 --arms disagree --gen-dir $G/steer/k${K}_t0/$c)
  FINAL="$FINAL:$j:$t"
done
SUM=$(sub --dependency=afterok$FINAL -J heads_summary heads_pilot/cpu.sbatch heads_pilot/summarize.py --k $K --gen-dir $G)
echo "out=$G generate=$GEN capture=$CAP judge=$JUD judge_t0=$JT0 rank=$RANK steer=$STEER steer_t0=$ST0 summary=$SUM"
