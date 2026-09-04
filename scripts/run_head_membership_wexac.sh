#!/bin/bash
# =====================================================================
# THE RECIPE-FREE TEST ON A REAL PRETRAINED TRANSFORMER'S HEAD -- the one surviving surface.
# A TENSION PRE-REGISTERED BEFORE THE RUN, because it constrains what is reachable: the head records ONE vector
# per image, so its span is N and the certificate needs r > N. The baseline validity band (0.6-0.9) meanwhile
# wants MANY members, since a handful of memorised images makes the trivial attack perfect. Those pull opposite
# ways, so the reachable window is N below r = 64 with the release fine-tuned rather than memorised. N = 8, 16, 32
# at r = 64 is that window; N = 64 and 128 are included precisely to show the certificate going vacuous when the
# batch reaches the rank, which is the head's own version of the saturation result.
# Every row reports the membership AUC TWO ways -- over ALL members and over RECORDED members only (N' by rank,
# gated on the certificate residual at each member's own truth). The GAP between them is the imprint law showing
# up in a membership metric: near-perfect for the images the model had to learn, nothing for the rest. Reporting
# either alone misleads in opposite directions. N', the head width m and the cap N' <= m-1 are logged per row, as
# are each member's certificate residual and final margin, so the imprint law is visible in the same cell.
# REGIME LABEL, stated not hidden: a fresh public-seeded head with labels outside the base model's output space --
# the new-class regime, where N' = N and the channel looks strongest, and also the canonical reason to fine-tune.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei104_headmem ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START head membership, real ViT-B/16 $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for CELL in "8 5 0.002" "16 5 0.002" "16 10 0.005" "32 5 0.002" "32 10 0.002" "64 5 0.002" "128 5 0.002"; do
  set -- $CELL
  echo "=== head: N=$1 members, T=$2 steps, lr=$3 ==="
  python -u -m experiments.exact_inversion.graded_imprint --module head --r 64 \
      --N $1 --n-nonmember 128 --T $2 --lr $3 --seed 1 --baseline-band 0.6 0.9 \
      --out $OUT/step104_headmem_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
