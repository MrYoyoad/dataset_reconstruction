#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_schemes_%J.out
#BSUB -e scripts/wexac_logs/jacobian_schemes_%J.err
#BSUB -J jacobian_schemes

# =====================================================================
# R2 — leakage vs perturbation ASSIGNMENT (DIFFERENT/SAME/MIXTURE) on honest theta0.
# Uses a near-memorized T=50. mnist x {gelu, relu}, N=4 k8.
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

echo ""; echo "########## R2: leakage vs assignment scheme ##########"; date
for ACT in gelu relu; do
  echo ""; echo "-- schemes mnist $ACT --"; date
  python -u -m experiments.jacobian_spectrum --schemes \
      --dataset mnist --activation $ACT --N 4 --k 8 --T 50 --rank 8 --tangent qr \
      --seed 42 --save --device cuda
done

echo ""; echo "=== DONE $(date) ==="