#!/bin/bash
# SHARPNESS, honestly. The attack is exact (1e-29); the blur is the chart's. Flowers are high-frequency
# texture and a linear chart cannot represent a SPECIFIC flower -- k=32->128 only moves repr err 0.30->0.22.
# Low intrinsic-dimension classes are the opposite: SVHN at k=32 already lands 196/200, so there is enormous
# margin to spend on chart dimension. The capacity law says how much: k < r - N', N'=8.
#   r=128 -> k <= 119 ;  r=256 -> k <= 247
# The chart is PUBLIC (PCA of public images of that class) -- attacker-available, no oracle.
set +u
CLS=$1; R=$2; K=$3; STARTS=${4:-200}
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/cifar_newclass/sharp_${LSB_JOBID}.jsonl
mkdir -p results/cifar_newclass
echo "=== $CLS r=$R k=$K starts=$STARTS (line k < $((R-8)))  $(date) ==="
python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass "$CLS" \
       --r $R --k $K --starts $STARTS --out $OUT
echo "=== DONE $(date) ==="
