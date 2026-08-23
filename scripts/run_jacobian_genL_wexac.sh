#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_genL_%J.out
#BSUB -e scripts/wexac_logs/jacobian_genL_%J.err
#BSUB -J jacobian_genL

# =====================================================================
# Gen-L (on-manifold tangents) vs random tangents: does using the REAL axes of
# image variation change the leakage picture?
#   qr         = random orthonormal pixel directions (the current baseline)
#   pca        = top-k PRINCIPAL directions of the data, unit-norm (matched
#                orthonormality -> isolates principal-vs-random)
#   pca_scaled = principal directions scaled by the REAL data spectrum (Gen-L
#                that carries the true variance profile)
#
# Compares, per (N,k): J0 eff_rank(J) + recovery-vs-eps, and J1 col(J) leakage
# LOWER BOUND q_eff|col(J) + iso_ratio + the Σ_J/μ mode count.
#
# NOTE (honesty): Gen-L is a PARTIAL realism step — real directions + real
# variance profile, but still LINEAR/orthonormal. True manifold curvature /
# non-orthogonal local tangents (the collinearity effect) needs Gen-G (VAE/
# StyleGAN); that is a later phase.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

for TAN in qr pca pca_scaled; do
  for NK in "2 8" "4 8"; do
    set -- $NK
    echo ""; echo "########## $TAN  N=$1 k=$2 ##########"; date
    echo "-- J0 (spectrum + recovery vs eps) --"
    python -u -m experiments.jacobian_spectrum --j0 \
        --N $1 --k $2 --T 5 --rank 8 --tangent $TAN \
        --eps_list 0.001 0.01 0.1 0.3 1.0 --save --device cuda
    echo "-- J1 (col(J) leakage lower bound + mode count) --"
    python -u -m experiments.jacobian_spectrum --j1 \
        --N $1 --k $2 --T 5 --rank 8 --tangent $TAN \
        --S_list 64 128 --shrink_list 0.01 \
        --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
