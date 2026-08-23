#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/crux_softplusb_%J.out
#BSUB -e scripts/wexac_logs/crux_softplusb_%J.err
#BSUB -J crux_softplusb
# THE CRUX's decisive missing test: softplus_b(beta) = a clean smooth->kinked knob (beta 0.5 smoothest
# -> beta 50 ~= relu), everything else fixed. beta x LR grid at T=1 -> matched-weight_change analysis of
# lin-error / NTK-survival / leakage vs smoothness on a CONTROLLED axis (not confounded activation identity).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
for BETA in 0.5 1 2 5 10 50; do
  for LR in 0.005 0.01 0.02 0.05 0.1; do
    echo "#### softplus_b${BETA} lr=${LR} ####"
    python -u -m experiments.run_experiment_b \
      --n_steps 1 --rank 8 --seed 42 --lr ${LR} \
      --finetune_activation softplus_b${BETA} \
      --no_baseline --save_results --skip_if_exists --device cuda 2>&1 | \
      grep -iE "weight.?change|lin.?error|feature.?stab|margin|ntk|ssim_norm" | head -8
  done
done
echo "=== DONE $(date) ==="
