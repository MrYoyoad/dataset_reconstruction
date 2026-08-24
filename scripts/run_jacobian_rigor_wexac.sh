#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_rigor_%J.out
#BSUB -e scripts/wexac_logs/jacobian_rigor_%J.err
#BSUB -J jacobian_rigor

# =====================================================================
# R3+R4 on the HONEST theta0: leakage (eff_rank/hard_rank of J) AND memorization
# (per-sample BCE on the actual private images) + private accuracy, across
# T = underfit -> converged -> overtrained. mnist x {gelu (exact J), relu
# (within-cell J)}; modifiedrelu is accuracy-only (guarded out of J). Uses the
# retrained weights-mnist_{gelu,relu}.pth. long-gpu (deep unroll at large T).
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

echo ""; echo "########## RIGOR: leakage + memorization vs T ##########"; date
for ACT in gelu relu; do
  echo ""; echo "-- rigor mnist $ACT --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset mnist --activation $ACT --N 4 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 --seed 42 --save --device cuda
done

echo ""; echo "=== DONE $(date) ==="