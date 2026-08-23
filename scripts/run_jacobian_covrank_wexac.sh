#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_covrank_%J.out
#BSUB -e scripts/wexac_logs/jacobian_covrank_%J.err
#BSUB -J jacobian_covrank

# =====================================================================
# Decisive test for the J1 "orthogonality" caveat (yoado-29): is the ~0%
# J-energy-in-noise-subspace real orthogonality, or a dimensionality artifact
# of undersampling a high-dim isotropic noise?
#
# eff_rank(Σ_seed) vs S = 16/32/64/128:
#   saturates / sharp sv decay  ⇒ low-dim noise captured ⇒ orthogonality REAL
#                                  ⇒ "random init is not a defense" holds.
#   grows ~linearly / flat spec ⇒ high-dim undersampled ⇒ q_eff INDETERMINATE;
#                                  use the isotropic fallback q_eff vs √μ.
# run_j1 now prints eff_rank(Σ_seed), the chance baseline, and q_eff|iso.
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

echo ""; echo "########## covrank-vs-S: eff_rank(Σ_seed) + q_eff|iso ##########"; date
for NK in "2 8" "4 8"; do
  set -- $NK
  echo ""; echo "---- N=$1 k=$2 (S=16/32/64/128) ----"
  python -u -m experiments.jacobian_spectrum --j1 \
      --N $1 --k $2 --T 5 --rank 8 --tangent qr \
      --S_list 16 32 64 128 --shrink_list 0.01 \
      --eps_list 0.1 0.3 1.0 3.0 10.0 \
      --save --tag "covrank_N$1_k$2" --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
