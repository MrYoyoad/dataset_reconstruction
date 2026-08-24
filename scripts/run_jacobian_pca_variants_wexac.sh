#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_pca_variants_%J.out
#BSUB -e scripts/wexac_logs/jacobian_pca_variants_%J.err
#BSUB -J jac_pca_variants

# =====================================================================
# The pca-VARIATIONS leakage suite on HONEST theta0 (companion to the qr run).
# run_h1 compares ALL tangent families with the col(J) leakage metrics
# (eff_rank, iso_ratio, q_eff, input+col(J) overlap): pca, difference (=private
# inter-image diffs, k=N-1), pca_tail, residual, and qr (orthonormal baseline).
# datasets {mnist,fashion} x acts {gelu,relu} x T {5 (comparison), 50 (memorized)}.
# N=4 k8 r8. (run_h1 handles difference's k=N-1 correctly; run_j1 does not.)
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

echo ""; echo "########## pca-variants leakage (honest theta0) ##########"; date
for DS in mnist fashion; do
  for ACT in gelu relu; do
    for T in 5 50; do
      echo ""; echo "-- h1 $DS $ACT T=$T (pca variants + qr) --"; date
      python -u -m experiments.jacobian_spectrum --h1 \
          --dataset $DS --activation $ACT --N 4 --k 8 --T $T --rank 8 \
          --h1_methods pca difference pca_tail residual qr \
          --eps_list 0.1 1.0 10.0 --seed 42 --save --device cuda
    done
  done
done

echo ""; echo "=== DONE $(date) ==="