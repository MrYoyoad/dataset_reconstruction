#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_rigor2_%J.out
#BSUB -e scripts/wexac_logs/jacobian_rigor2_%J.err
#BSUB -J jacobian_rigor2

# =====================================================================
# Rigor round 2 (honest theta0, mnist gelu): reach FULL memorization + OVERTRAINING
# and lr-sanity. Round 1 (lr=0.01) stayed underfit through T=100 (max_bce ~3e-3).
# Sweep lr {0.03, 0.1} across T {5,20,50,100,200}: does higher lr memorize
# (max per-sample BCE < 1e-3), and how does eff_rank(J) behave through the
# converged -> overtrained regime? Answers "is the lr right?" + "what happens
# training more?". long-gpu (deep T=200 2nd-order unroll).
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

echo ""; echo "########## RIGOR2: lr-sanity + memorization + overtraining ##########"; date
for LR in 0.03 0.1; do
  echo ""; echo "-- rigor2 mnist gelu lr=$LR --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset mnist --activation gelu --N 4 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 200 --lr $LR --seed 42 --save --tag "mnist_gelu_lr${LR}" \
      --device cuda
done

echo ""; echo "=== DONE $(date) ==="