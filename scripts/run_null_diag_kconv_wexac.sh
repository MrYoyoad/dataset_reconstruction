#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/null_diag_kconv_%J.out
#BSUB -e scripts/wexac_logs/null_diag_kconv_%J.err
#BSUB -J null_diag_kconv
# DECISIVE TEST for arm-B's residual K-growth: run the 3-way metric on NO-SIGNAL data
# (v_j = reseed_B - reseed_A, no swap) at K=50/100/200. If sensitivity reads ~0 FLAT across K,
# the real-data K-growth is genuine signal-direction resolution (d² = K-tightening lower bound).
# If null sensitivity ALSO grows with K, there is a residual bias the sign-flip null misses.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
for K in 50 100 200; do
  echo ""; echo "########## NULL-DIAG K=$K (N in 4,16) ##########"; date
  python -u -m experiments.dataset_sensitivity.arm_b_null_diag --N_list 4 16 --K $K --lr 0.5 --T 1000 --rank 8 --device cuda
done
echo ""; echo "=== DONE $(date) ==="
echo "READ: null sensitivity ~0 & FLAT across K => real-data growth is signal resolution (benign);"
echo "      null sensitivity GROWS with K => residual estimator bias (must fix before quoting absolutes)."
