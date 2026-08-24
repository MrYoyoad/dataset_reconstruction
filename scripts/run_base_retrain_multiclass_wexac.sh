#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/base_retrain_mc_%J.out
#BSUB -e scripts/wexac_logs/base_retrain_mc_%J.err
#BSUB -J base_mc

# =====================================================================
# TIER B — 10-class honest base models (mnist_10class / fashion_10class),
# CrossEntropy training (Main.get_loss_ce branches on output_dim>1). Full-ish
# data (dpc=5000 -> ~50k imgs, the 50000-subset cap) + healthy init (0.05) to
# match Tier A. Saved as models/weights-<ds>10_<act>.pth — EXACTLY the name
# _honest_target(num_classes=10) loads (suffix = str(num_classes) = "10").
#   datasets:    mnist_10class, fashion_10class (784, [1000,1000], output_dim=10)
#   activations: relu, gelu, modifiedrelu
# CIFAR-10 is a separate heavier job (3072-in, monster arch) — deferred.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

DPC=5000
INIT="[0.05,0.05]"

for DS in mnist fashion; do
  PROB=${DS}_10class
  for ACT in relu gelu modifiedrelu; do
    echo ""; echo "########## BASE TRAIN 10-class  $DS x $ACT  dpc=$DPC init=$INIT ##########"; date
    python -u Main.py --run_mode=train --problem=$PROB \
        --model_train_activation=$ACT --proj_name=${DS}10_${ACT} \
        --data_per_class_train=$DPC \
        --model_hidden_list="[1000,1000]" --model_init_list="$INIT" \
        --train_epochs=20000 --train_lr=0.01 --train_evaluate_rate=500 \
        --train_threshold=1e-4
    if [ $? -ne 0 ]; then echo "WARN: base train $DS x $ACT failed"; continue; fi
    SRC=$(ls -t runs/*_${DS}10_${ACT}/weights-*.pth 2>/dev/null | head -1)
    if [ -n "$SRC" ]; then
      cp "$SRC" "models/weights-${DS}10_${ACT}.pth"
      echo "COLLECTED -> models/weights-${DS}10_${ACT}.pth (from $SRC)"
    else
      echo "WARN: no checkpoint found for ${DS}10_${ACT}"
    fi
  done
done

echo ""; echo "=== ALL 10-CLASS BASE MODELS DONE $(date) ==="
ls -la models/weights-*10_*.pth 2>/dev/null || true
