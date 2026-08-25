#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/softplus_beta_sweep_%J.out
#BSUB -e scripts/wexac_logs/softplus_beta_sweep_%J.err
#BSUB -J sp_beta

# =====================================================================
# R5 softplus-beta bridge for yoado-b1's twin-axis figure: eff_rank(J) vs beta at
# T=200 MEMORIZATION, to overlay on yoado-b1's lin-err-vs-beta at T=1. Honest
# softplus base models (job 203211): weights-mnist_softplus_b<beta>.pth. Softplus
# is smooth (double-backward clean) so the EXACT J is valid (unlike relu/mrelu).
# beta in {0.5,1,2,5,10,50}, mnist, N=4 k8 r8 seed42 qr, lr=0.1, Ts 5..200.
# run_rigor gives eff_rank(J)+memorization vs T -> read eff_rank at the memorized T.
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

echo ""; echo "########## softplus-beta eff_rank(J) vs T (per beta) ##########"; date
for B in 0.5 1 2 5 10 50; do
  ACT="softplus_b${B}"
  echo ""; echo "-- rigor mnist $ACT lr=0.1 --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset mnist --activation $ACT --num_classes 2 \
      --N 4 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 200 --lr 0.1 --seed 42 --save --device cuda \
      --tag mnist_${ACT}_b1sweep
done

echo ""; echo "=== DONE $(date) ==="
