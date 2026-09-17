#!/bin/bash
# THE CONTROL THE FIGURE NEEDS. Same pipeline, same chart, same starts -- but the certificate is built from a
# release trained on DIFFERENT private images. If this recovers nothing, the recoveries in the main cell are
# about THIS release and not about the chart being able to draw digits.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
mkdir -p results/cifar_newclass
echo "=== WRONG-RELEASE CONTROL: svhn:3 r=128 k=64 starts=150 iters=800  $(date) ==="
python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass svhn:3 \
  --r 128 --k 64 --starts 150 --iters 800 --wrong_release \
  --out results/cifar_newclass/sharp3_ctl_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
