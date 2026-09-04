#!/bin/bash
# =====================================================================
# The graded-imprint comparison REDONE where the baseline is not already perfect. In job 280255 the adapter
# memorised 8 images to loss 4e-3, so the loss-threshold baseline scored AUC 1.000 and no method could beat it --
# the comparison could only be tied or lost, and ours lost. That is a fault in the test, not evidence about the
# score, and the honest fix is a release that is fine-tuned rather than memorised.
# Larger private set, fewer steps, gentler learning rate, so the trivial attack is informative. The HEAD arm is
# the positive control: it is the one module where the certificate itself is non-vacuous, so the graded score must
# reproduce it there or the statistic is wrong.
# PRE-REGISTERED, unchanged: the graded score must BEAT the loss baseline to be worth anything. If it does not,
# the recipe-free channel on weight-shared modules is finished in every form we have tried, and that is the result.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei102_gradedfair ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START graded fair-baseline $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for CELL in "64 20 0.01" "64 10 0.005" "128 20 0.01"; do
  set -- $CELL
  echo "=== N=$1 members, T=$2 steps, lr=$3 (fine-tuned, not memorised) ==="
  python -u -m experiments.exact_inversion.graded_imprint --block 6 --r 8 16 64 \
      --N $1 --n-nonmember 128 --T $2 --lr $3 --seed 1 \
      --out $OUT/step102_gradedfair_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
