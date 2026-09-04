#!/bin/bash
# =====================================================================
# A SOLUTION ATTEMPT, not a new claim: when the recorded span saturates the certificate dies, because the
# certificate is a NULL-SPACE test and a saturated release has no null space. The singular SPECTRUM of B_T is
# still not flat, so the graded continuation of the same test -- the singular-value-weighted share of a
# candidate's adapter-space energy -- is the only form in which the recipe-free test can survive on a transformer.
# It reduces to the certificate exactly when the release is unsaturated.
# Real pretrained ViT-B/16, frozen, public-seeded head, LoRA on one block's qkv at DEPLOYED ranks, FP64, real
# photographs, 8 members against 64 non-members never trained on.
# PRE-REGISTERED: DEAD (AUC ~ 0.5) is the likely outcome and ends this line honestly. GRADED (members above
# non-members) means the channel survives as a MEMBERSHIP signal, never as reconstruction.
# MANDATORY CONTROL: the same AUC from a plain loss-threshold membership attack. Our score must BEAT it, or it is
# a worse way of doing something already standard and must be reported as such.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei101_graded \
#        -o scripts/wexac_logs/ei101_graded_%J.out -e scripts/wexac_logs/ei101_graded_%J.err \
#        bash scripts/run_graded_imprint_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START graded imprint $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for BLK in 6 11; do
  echo "=== block $BLK ==="
  python -u -m experiments.exact_inversion.graded_imprint --block $BLK --r 8 16 64 \
      --N 8 --n-nonmember 64 --T 100 --lr 0.05 --seed 1 \
      --out $OUT/step101_graded_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
