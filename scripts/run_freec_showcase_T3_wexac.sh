#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -o scripts/wexac_logs/freec_showcase_T3_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_T3_%J.err
#BSUB -J freec_showcase_T3
# Priority comparison rows for the showcase (jobs 323866/323867 run ~20 min/cell and queue these last):
# full fine-tune free-c at T=5/10, LoRA r16/r32 leaky_relu at T=5, relu r8 at T=5, leaky_relu r8 at T=10.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
run() { echo ""; echo "########## $* ##########"; date; python -u -m experiments.run_experiment_b "$@" --seed 42 --free_coefficients --no_baseline --save_results --skip_if_exists --device cuda; }
run --n_steps 5  --lr 0.002   --finetune_activation leaky_relu
run --n_steps 5  --lr 0.0006  --finetune_activation leaky_relu
run --n_steps 5  --rank 16 --lr 0.00054 --finetune_activation leaky_relu
run --n_steps 5  --rank 32 --lr 0.00054 --finetune_activation leaky_relu
run --n_steps 5  --rank 8  --lr 0.00054 --finetune_activation relu
run --n_steps 10 --rank 8  --lr 0.00027 --finetune_activation leaky_relu
run --n_steps 10 --lr 0.001   --finetune_activation leaky_relu
run --n_steps 5  --lr 0.006   --finetune_activation leaky_relu
run --n_steps 5  --rank 16 --lr 0.00178 --finetune_activation leaky_relu
run --n_steps 5  --rank 32 --lr 0.00178 --finetune_activation leaky_relu
run --n_steps 10 --rank 8  --lr 0.00089 --finetune_activation leaky_relu
run --n_steps 10 --lr 0.003   --finetune_activation leaky_relu
echo ""; echo "=== ALL DONE $(date) ==="
