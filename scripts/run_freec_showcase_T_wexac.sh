#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='hgn45' && hname!='lgn28' && hname!='lgn13' && hname!='hgn29']"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -o scripts/wexac_logs/freec_showcase_T_%J.out
#BSUB -e scripts/wexac_logs/freec_showcase_T_%J.err
#BSUB -J freec_showcase_T

# =====================================================================
# FREE-COEFFICIENT reconstructions at NON-TRIVIAL T (user ask, 2026-08-31):
# the deck's good LoRA examples are T=1; the meeting needs realistic (free-c)
# LoRA reconstructions at T in {5,10,20} plus the full-fine-tune comparison,
# WITH SAVED TENSORS (experiment output rules). MNIST N=2 seed 42.
# Activations: the leaking cluster (leaky_relu, relu, selu) + smooth contrast
# (sigmoid, softplus). lr grid scales the T=1 free-c winners (~0.003-0.009)
# by 1/T to hold weight-change roughly constant across T.
# r=8 at every T; r=16/32 at T=10 (rank story); full FT (no --rank) at every T.
# --skip_if_exists makes it resumable; free-c tensors get "_free_" names.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

run () {  # ACT T LR RANKFLAGS...
  local ACT=$1 T=$2 LR=$3; shift 3
  echo ""; echo "########## act=$ACT T=$T lr=$LR $* free-c ##########"; date
  python -u -m experiments.run_experiment_b \
      --n_steps "$T" --seed 42 --lr "$LR" \
      --finetune_activation "$ACT" \
      --free_coefficients \
      --no_baseline --save_results --skip_if_exists --device cuda "$@"
}

for T in 5 10 20; do
  for ACT in leaky_relu relu selu sigmoid softplus; do
    for BASELR in 0.0027 0.0089 0.027; do
      LR=$(python -c "print($BASELR/$T)")
      run "$ACT" "$T" "$LR" --rank 8
    done
  done
done
for R in 16 32; do
  for ACT in leaky_relu relu; do
    for BASELR in 0.0027 0.0089 0.027; do
      LR=$(python -c "print($BASELR/10)")
      run "$ACT" 10 "$LR" --rank "$R"
    done
  done
done
for T in 5 10 20; do
  for BASELR in 0.01 0.03 0.1; do
    LR=$(python -c "print($BASELR/$T)")
    run leaky_relu "$T" "$LR"
  done
done
echo ""; echo "=== ALL DONE $(date) ==="
