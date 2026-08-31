#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -o scripts/wexac_logs/freec_showcase_N_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_N_%J.err
#BSUB -J freec_showcase_N
# N-sweep for the free-coefficient showcase (user: "did we do only N=2? why not more?").
# MNIST leaky-ReLU T=5 (the best T=5 cells) at N = 4, 6, 10 (n_per_class 2, 3, 5): LoRA r=8, r=32 and full fine-tune.
# Flowers ReLU T=5 at N = 4, 6: LoRA r=8 and full. Free coefficients, tensors saved. Expect the superposition problem to
# grow with N (the deck's direct-inversion wall) — this measures where the free-c LoRA attack breaks with N.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
lora() { echo ""; echo "########## LORA $* ##########"; date; python -u -m experiments.run_experiment_b "$@" --seed 42 --free_coefficients --no_baseline --save_results --skip_if_exists --device cuda; }
full() { echo ""; echo "########## FULL $* ##########"; date; python -u -m experiments.run_experiment_b "$@" --seed 42 --free_coefficients --save_results --skip_if_exists --device cuda; }
for NPC in 2 3 5; do
  lora --n_steps 5 --rank 8  --lr 0.00054 --finetune_activation leaky_relu --n_per_class $NPC
  full --n_steps 5           --lr 0.0006  --finetune_activation leaky_relu --n_per_class $NPC
  lora --n_steps 5 --rank 32 --lr 0.00054 --finetune_activation leaky_relu --n_per_class $NPC
done
for NPC in 2 3; do
  lora --dataset flowers32 --n_steps 5 --rank 8 --lr 0.0006 --relu_alpha 10000 --optimizer sgd --n_per_class $NPC
  full --dataset flowers32 --n_steps 5          --lr 0.002  --relu_alpha 10000 --optimizer sgd --n_per_class $NPC
done
echo ""; echo "=== ALL DONE $(date) ==="
