#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o scripts/wexac_logs/freec_showcase_T4_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_T4_%J.err
#BSUB -J freec_showcase_T4
# FULL fine-tune free-c rows for the showcase. NOTE: with --rank omitted the full-FT run IS the
# "baseline" branch, so --no_baseline must NOT be passed (it raised "Nothing to run" in jobs
# 323866/336206). Same seed/images as the LoRA cells so the rows are comparable.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
run() { echo ""; echo "########## FULL $* ##########"; date; python -u -m experiments.run_experiment_b "$@" --seed 42 --free_coefficients --save_results --skip_if_exists --device cuda; }
run --n_steps 5  --lr 0.002  --finetune_activation leaky_relu
run --n_steps 5  --lr 0.0006 --finetune_activation leaky_relu
run --dataset flowers32 --n_steps 5  --lr 0.002  --relu_alpha 10000 --optimizer sgd
run --n_steps 10 --lr 0.001  --finetune_activation leaky_relu
run --dataset flowers32 --n_steps 10 --lr 0.001  --relu_alpha 10000 --optimizer sgd
run --n_steps 5  --lr 0.006  --finetune_activation leaky_relu
run --dataset flowers32 --n_steps 5  --lr 0.0006 --relu_alpha 10000 --optimizer sgd
run --n_steps 10 --lr 0.003  --finetune_activation leaky_relu
echo ""; echo "=== ALL DONE $(date) ==="
