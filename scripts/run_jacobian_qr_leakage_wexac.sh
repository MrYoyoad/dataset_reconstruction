#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_qr_leakage_%J.out
#BSUB -e scripts/wexac_logs/jacobian_qr_leakage_%J.err
#BSUB -J jac_qr_leakage

# =====================================================================
# The pca-additions leakage suite, but with ORTHONORMAL (qr) additions as the
# SECRET (not a defense), on the HONEST theta0. run_j1 = the col(J) leakage
# analysis (eff_rank, iso_ratio, q_eff, Sigma_J/mu mode count) that we used for
# the Gen-L/pca comparison. datasets {mnist,fashion} x acts {gelu,relu} x
# T {5 (old comparison point), 50 (near-memorized)}. N=4 k8 r8, tangent qr.
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

echo ""; echo "########## qr leakage (orthonormal secret, honest theta0) ##########"; date
for DS in mnist fashion; do
  for ACT in gelu relu; do
    for T in 5 50; do
      echo ""; echo "-- j1 $DS $ACT T=$T tangent=qr --"; date
      python -u -m experiments.jacobian_spectrum --j1 \
          --dataset $DS --activation $ACT --tangent qr \
          --N 4 --k 8 --T $T --rank 8 --S_list 64 --shrink_list 0.01 \
          --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda
    done
  done
done

echo ""; echo "=== DONE $(date) ==="