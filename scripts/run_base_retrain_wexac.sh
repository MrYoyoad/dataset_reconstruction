#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/base_retrain_%J.out
#BSUB -e scripts/wexac_logs/base_retrain_%J.err
#BSUB -J base_retrain

# =====================================================================
# R1 — Honest base models per (dataset x activation). Fixes the "GELU-on-ReLU-
# weights" shortcut: train each activation's OWN theta0 and VALIDATE by test
# accuracy (Main.py prints train/test error+loss each eval). No swap.
#   datasets:    mnist, fashion (784, [1000,1000])   [CIFAR/RGB = separate heavy job]
#   activations: relu, gelu, modifiedrelu
# Weights land in runs/<run>/weights-<problem>_d250_<dataset>_<act>.pth; we copy
# each to dataset_reconstruction/models/weights-<dataset>_<act>.pth (canonical theta0).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

for DS in mnist fashion; do
  PROB=${DS}_odd_even
  for ACT in relu gelu modifiedrelu; do
    echo ""; echo "########## BASE TRAIN  $DS x $ACT ##########"; date
    python -u Main.py --run_mode=train --problem=$PROB \
        --model_train_activation=$ACT --proj_name=${DS}_${ACT} \
        --data_per_class_train=250 \
        --model_hidden_list="[1000,1000]" --model_init_list="[0.0001,0.0001]" \
        --train_epochs=200000 --train_lr=0.01 --train_evaluate_rate=2000 \
        --train_threshold=1e-6 --device=cuda
    if [ $? -ne 0 ]; then echo "WARN: base train $DS x $ACT failed"; continue; fi
    # collect the freshest checkpoint for this model_name into models/
    MN="${PROB}_d250_${DS}_${ACT}"
    SRC=$(ls -t runs/*_${MN}/weights-${MN}.pth 2>/dev/null | head -1)
    if [ -n "$SRC" ]; then
      cp "$SRC" "models/weights-${DS}_${ACT}.pth"
      echo "COLLECTED -> models/weights-${DS}_${ACT}.pth (from $SRC)"
    else
      echo "WARN: no checkpoint found for $MN"
    fi
  done
done

echo ""; echo "=== ALL BASE MODELS DONE $(date) ==="
ls -la models/weights-*_*.pth 2>/dev/null | grep -E "mnist_|fashion_" || true