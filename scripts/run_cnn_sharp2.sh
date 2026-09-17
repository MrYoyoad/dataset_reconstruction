#!/bin/bash
# HIGH-k STALLS ARE AN OPTIMISATION FAILURE, NOT AN ALIAS. At k=192 the solver reaches 4.7e-06 where the
# working k=32 cells reach 1e-29 -- seven orders short of the floor, with nothing landing. The search is
# 8*192 = 1536-dimensional against 8*32 = 256, and 300 LM iterations do not get there. So: spend the budget
# on ITERATIONS per start instead of on the NUMBER of starts. We do not need 200 starts, we need a few that
# converge. Also probes intermediate k, where the basin is wider and the chart is still much better than 32.
set +u
CLS=$1; R=$2; K=$3; STARTS=$4; ITERS=$5
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/cifar_newclass/sharp2_${LSB_JOBID}.jsonl
mkdir -p results/cifar_newclass
echo "=== $CLS r=$R k=$K starts=$STARTS iters=$ITERS (line k < $((R-8)))  $(date) ==="
python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass "$CLS" \
       --r $R --k $K --starts $STARTS --iters $ITERS --out $OUT
echo "=== DONE $(date) ==="
