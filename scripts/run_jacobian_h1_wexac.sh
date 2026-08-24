#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_h1_%J.out
#BSUB -e scripts/wexac_logs/jacobian_h1_%J.err
#BSUB -J jacobian_h1

# =====================================================================
# H1 — discriminative tangents vs PCA-shared-modes (plan rev 2/3).
# Does an image-specific basis genuinely change col(J) (not just relabel it)?
#   difference = top-(N-1) PCs of the private set's own {x_i-xbar} — ON-manifold,
#     privacy-relevant, rank<=N-1 BY CONSTRUCTION (the finding). pca_tail/residual
#     = OFF-manifold contrasts. Guardrail = principal-angle overlap vs matched-k
#     PCA in BOTH input-space AND col(J)/Y-space (invariance theorem lives in col J).
# Report: fraction hard_rank/Nk + iso_ratio (INVARIANT), eff_rank/q_eff (coord-dep),
# in_ovlp + colJ_ovlp vs pca. Save tangent-direction image grids.
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

echo ""; echo "########## STAGE 1: H1 smoke (mnist N=2 seed=42) ##########"; date
python -u -m experiments.jacobian_spectrum --h1 --dataset mnist --N 2 --k 8 \
    --T 5 --rank 8 --seed 42 --eps_list 0.1 1.0 10.0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: H1 smoke failed. Aborting."; exit 1; fi

echo ""; echo "########## STAGE 2: H1 sweep (dataset x N x seed) ##########"; date
for DS in mnist fashion flowers; do
  for N in 2 4; do
    for SEED in 42 1; do
      echo ""; echo "-- H1 $DS N=$N seed=$SEED --"; date
      python -u -m experiments.jacobian_spectrum --h1 \
          --dataset $DS --N $N --k 8 --T 5 --rank 8 --seed $SEED \
          --eps_list 0.1 1.0 10.0 --save --device cuda
    done
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
