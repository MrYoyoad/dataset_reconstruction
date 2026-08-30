#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -o scripts/wexac_logs/freec_showcase_T2_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_T2_%J.err
#BSUB -J freec_showcase_T2

# =====================================================================
# Free-coefficient showcase sweep, part 2/2 (user: NO T=1 — find configs that
# LOOK GOOD with LoRA at non-trivial T; free-c only).
#   A) flowers32 (the visually strong dataset; its T=1 free-c LoRA hit ssim
#      0.60-0.70 vs ctrl 0.50 at every rank): a10000 recipe (relu_alpha=10000
#      extraction, sgd, N=2), T in {5,10,20}, r in {8,32}, lr = {0.01,0.03,0.003}/T.
#   B) MNIST cells missing from job 322766: r in {16,32} at T in {5,20} for
#      leaky_relu/relu, + selu r16 T10; lr = {0.0027,0.0089}/T.
# All --free_coefficients --save_results --skip_if_exists (tensors saved).
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

for T in 5 10 20; do
  for R in 8 32; do
    for BASELR in 0.003 0.01 0.03; do
      LR=$(python -c "print($BASELR/$T)")
      echo ""; echo "########## flowers32 T=$T r=$R lr=$LR relu_alpha=10000 free-c ##########"; date
      python -u -m experiments.run_experiment_b \
          --dataset flowers32 --n_steps "$T" --rank "$R" --seed 42 --lr "$LR" \
          --relu_alpha 10000 --optimizer sgd \
          --free_coefficients \
          --no_baseline --save_results --skip_if_exists --device cuda
    done
  done
done

for T in 5 20; do
  for R in 16 32; do
    for ACT in leaky_relu relu; do
      for BASELR in 0.0027 0.0089; do
        LR=$(python -c "print($BASELR/$T)")
        echo ""; echo "########## mnist T=$T r=$R act=$ACT lr=$LR free-c ##########"; date
        python -u -m experiments.run_experiment_b \
            --n_steps "$T" --rank "$R" --seed 42 --lr "$LR" \
            --finetune_activation "$ACT" \
            --free_coefficients \
            --no_baseline --save_results --skip_if_exists --device cuda
      done
    done
  done
done
for BASELR in 0.0027 0.0089; do
  LR=$(python -c "print($BASELR/10)")
  python -u -m experiments.run_experiment_b \
      --n_steps 10 --rank 16 --seed 42 --lr "$LR" \
      --finetune_activation selu --free_coefficients \
      --no_baseline --save_results --skip_if_exists --device cuda
done
echo ""; echo "=== ALL DONE $(date) ==="
