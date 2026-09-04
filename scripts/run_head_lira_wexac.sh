#!/bin/bash
# =====================================================================
# A REAL membership comparator for the head claim (yoado-cd: "if the trajectory lands positive this stops being
# optional"). A loss threshold is the weakest baseline in the membership literature; this measures the certificate
# against a shadow-model likelihood-ratio attack (LiRA) on the SAME releases, across the same training-length
# trajectory.
# Affordable only because the encoder is FROZEN: the pool's features are cached with one forward pass and every
# shadow release is head-only training on cached features, so 128 shadows cost less than one Jacobian run.
# Report AUC *and* COST side by side -- the certificate needs the release and the public model; LiRA needs a
# shadow-training budget, the recipe, and a sample from the data distribution. A tie on AUC is not a tie.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei106_headlira ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START head LiRA $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.head_lira --pool 256 --N 32 --r 64 --shadows 128 \
    --Ts 5 20 50 100 200 400 --lr 0.02 --seed 1 --out $OUT/step106_headlira_${LSB_JOBID}.jsonl
echo "=== vacuous control: N = r, where the certificate must read nothing ==="
python -u -m experiments.exact_inversion.head_lira --pool 256 --N 64 --r 64 --shadows 128 \
    --Ts 5 50 200 --lr 0.02 --seed 1 --out $OUT/step106_headlira_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
