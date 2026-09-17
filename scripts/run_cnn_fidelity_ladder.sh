#!/bin/bash
# THE SUPERVISOR FIGURE. The CNN certificate attack already works (job 293438: flowers102, 66/200 starts,
# 8/8 images, residual 3e-29). The recovered row is IDENTICAL to the chart-projection row -- so the attack is
# exact and EVERY bit of the visible blur is the chart, not the attack.
#
# The chart's fidelity is set by k, and the theory says exactly how much k is allowed: k < r - N', with
# N' = rank B_T <= min(m-1, r, N) = 8 here. So a bigger adapter BUYS a sharper picture, and the capacity law
# is what tells you how much. That is the figure: fidelity climbing the ladder the theorem permits.
#   r=64  -> k <= 55   (the published cell used k=32, far below its own ceiling)
#   r=128 -> k <= 119
#   r=256 -> k <= 247
set +u
CELL=$1; R=$2; K=$3; STARTS=${4:-300}; EXTRA=${5:-}
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/cifar_newclass/fidelity_${CELL}_${LSB_JOBID}.jsonl
mkdir -p results/cifar_newclass
echo "=== $CELL r=$R k=$K starts=$STARTS $EXTRA  (line k < r-N' = $((R-8)))  $(date) ==="
python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass flowers102 \
       --r $R --k $K --starts $STARTS $EXTRA --out $OUT
echo "=== DONE $(date) ==="
