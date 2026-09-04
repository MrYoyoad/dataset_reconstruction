#!/bin/bash
# =====================================================================
# TRUNCATED CERTIFICATE (yoado-cd): the only route I can see to the saturated architectures, and cleaner than the
# graded-energy statistic that failed, because it is a PROJECTOR with a computable error rather than a score.
# The exact certificate dies at rank B_T = r because the projector annihilates A_T. But the span was measured FULL
# RANK, not FLAT. Project out only the top-k directions: C_k = P_{top-k}^perp A_T gives r-k conditions with a
# member residual of about the DISCARDED TAIL's energy instead of zero. It exists at every k < r, including where
# the exact certificate is identically zero. The trade is exactness for existence.
# MEASURE FIRST, ATTACK SECOND: report B_T's normalised spectrum and the tail beyond every k, and the member vs
# non-member residuals at every k, on the saturated ViT modules.
# PRE-REGISTERED: tail at useful k orders below the non-member residual (0.1-1) => a truncated certificate
# separates and the channel is not closed on transformers, only its exact version is. Flat spectrum => truncation
# buys nothing and the closure stands as reported. Neither answer can come back void.
# If it works, the counting rule must be restated as a condition on EXACTNESS rather than on existence.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei110_trunc ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START truncated certificate $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.truncated_certificate --blocks 0 6 11 --r 16 64 \
    --N 8 --n-nonmember 64 --T 100 --lr 0.02 --seed 1 \
    --out $OUT/step110_trunc_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
