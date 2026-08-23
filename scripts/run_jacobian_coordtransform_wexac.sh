#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_coordtf_%J.out
#BSUB -e scripts/wexac_logs/jacobian_coordtf_%J.err
#BSUB -J jacobian_coordtf

# =====================================================================
# "Subtract linear parts of the PCA tangents from each other, then try again."
# Applies coordinate reparametrizations to the pca J and re-measures:
#   response_white      : rescale so J's columns are orthonormal (flat spectrum)
#   crossimg_sumdiff    : common+difference across the N=2 images (relabel)
#   crossimg_diffONLY   : keep ONLY the difference directions (subspace restriction)
# Shows hard_rank(col J) (the true, invariant leakage) vs eff_rank / q_eff
# (coordinate-dependent). Run on the masking cell (mnist pca seed42) and a
# non-masking cell (mnist pca seed1); plus flowers seed42 (the other elevated one).
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

echo ""; echo "########## COORD TRANSFORMS ##########"; date
for CELL in "mnist 42" "mnist 1" "flowers 42"; do
  set -- $CELL
  echo ""; echo "-- coord transforms $1 pca seed=$2 --"
  python -u -m experiments.jacobian_spectrum --coord_transforms \
      --dataset $1 --tangent pca --N 2 --k 8 --T 5 --rank 8 --seed $2 \
      --eps_list 0.1 0.3 1.0 3.0 10.0 --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
