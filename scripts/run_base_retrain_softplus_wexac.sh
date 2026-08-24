#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/base_retrain_softplus_%J.out
#BSUB -e scripts/wexac_logs/base_retrain_softplus_%J.err
#BSUB -J base_softplus

# =====================================================================
# R5 softplus-beta bridge: honest mnist base models per softplus_b<beta>, matching
# yoado-b1's crux config (beta 0.5,1,2,5,10,50; r=8; seed=42) so eff_rank(J)-vs-beta
# at T=200 memorization can overlay its lin-err-vs-beta at T=1 on one axis.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

for B in 0.5 1 2 5 10 50; do
  ACT="softplus_b${B}"
  echo ""; echo "########## BASE TRAIN  mnist x $ACT ##########"; date
  python -u Main.py --run_mode=train --problem=mnist_odd_even \
      --model_train_activation=$ACT --proj_name=mnist_${ACT} \
      --data_per_class_train=250 \
      --model_hidden_list="[1000,1000]" --model_init_list="[0.0001,0.0001]" \
      --train_epochs=200000 --train_lr=0.01 --train_evaluate_rate=2000 \
      --train_threshold=1e-6
  if [ $? -ne 0 ]; then echo "WARN: base train mnist x $ACT failed"; continue; fi
  MN="mnist_odd_even_d250_mnist_${ACT}"
  SRC=$(ls -t runs/*_${MN}/weights-${MN}.pth 2>/dev/null | head -1)
  if [ -n "$SRC" ]; then
    cp "$SRC" "models/weights-mnist_${ACT}.pth"
    echo "COLLECTED -> models/weights-mnist_${ACT}.pth (from $SRC)"
  else
    echo "WARN: no checkpoint found for $MN"
  fi
done

echo ""; echo "=== ALL SOFTPLUS BASE MODELS DONE $(date) ==="