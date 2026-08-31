#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -o scripts/wexac_logs/freec_showcase_N4lr_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_N4lr_%J.err
#BSUB -J freec_showcase_N4lr
# CONFOUND CONTROL for the N-sweep: job 497350 reuses the lr tuned at N=2, so an N=4 drop could be
# under-tuning rather than superposition. Sweep lr per N=4 cell (r=8 and full, MNIST leaky-ReLU T=5).
# If the best lr at N=4 still fails the mean-image baseline, the drop is the mixing symmetry, not the lr.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
for LR in 0.00018 0.00108 0.0027; do
  echo ""; echo "########## LORA N=4 r8 lr=$LR ##########"; date
  python -u -m experiments.run_experiment_b --n_steps 5 --rank 8 --lr "$LR" --finetune_activation leaky_relu \
      --n_per_class 2 --seed 42 --free_coefficients --no_baseline --save_results --skip_if_exists --device cuda
done
for LR in 0.0002 0.002 0.006; do
  echo ""; echo "########## FULL N=4 lr=$LR ##########"; date
  python -u -m experiments.run_experiment_b --n_steps 5 --lr "$LR" --finetune_activation leaky_relu \
      --n_per_class 2 --seed 42 --free_coefficients --save_results --skip_if_exists --device cuda
done
echo ""; echo "=== ALL DONE $(date) ==="
