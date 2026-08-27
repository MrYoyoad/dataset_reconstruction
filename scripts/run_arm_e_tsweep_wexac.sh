#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_e_tsweep_%J.out
#BSUB -e scripts/wexac_logs/arm_e_tsweep_%J.err
#BSUB -J arm_e_tsweep
# ARM E follow-up — DUPLICATION x TRAINING-BUDGET. At convergence (T=1000) duplication imprint was
# sub-linear (β~0.24). Does it BITE at a fixed/short budget (the memorization regime)? Sweep T at the
# privacy-relevant bottleneck rank=8. Read: does β(sensitivity vs k) RISE as T shrinks?
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
for T in 50 200 1000; do
  echo ""; echo "########## T=$T : rank 8, k in {1,2,4,8}, N=16, K=50 ##########"; date
  python -u -m experiments.dataset_sensitivity.arm_e_duplication \
      --rank_list 8 --k_list 1 2 4 8 --N 16 --K 50 --n_targets 4 --lr 0.5 --T $T --device cuda
done
echo ""; echo "=== DONE $(date) ==="
echo "READ: β(sens vs k) across T. β rising as T falls => duplication bites at fixed budget (memorization);"
echo "      β flat across T => saturation is intrinsic, not a convergence effect."
