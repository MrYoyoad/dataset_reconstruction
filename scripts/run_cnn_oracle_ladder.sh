#!/bin/bash
# THE QUESTION THAT DECIDES THE PICTURE. The attack is exact -- it lands on the chart's own projection to
# 1e-29. So fidelity is entirely the chart's, and a PUBLIC PCA chart cannot represent a SPECIFIC flower:
# k=32 -> repr err 0.30, and quadrupling to k=128 only reaches 0.22. More k is the wrong lever.
#
# So ask what chart WOULD suffice, by sweeping how close to the private subspace the chart has to be.
# eps=0 is the oracle (chart spans the privates: an upper bound, NOT attacker-available). eps>0 perturbs
# it. The useful output is the TOLERANCE: if the attack survives eps=0.02, then an attacker needs a chart
# only within 2% of the data manifold, and supplying that is a generative-prior problem, not a theory gap.
set +u
TAG=$1; EPS=$2; STARTS=${3:-200}
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/cifar_newclass/oracle_${TAG}_${LSB_JOBID}.jsonl
mkdir -p results/cifar_newclass
echo "=== oracle eps=$EPS starts=$STARTS  $(date) ==="
python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass flowers102 \
       --r 128 --k 96 --starts $STARTS --chart oracle --eps $EPS --private raw --out $OUT
echo "=== DONE $(date) ==="
