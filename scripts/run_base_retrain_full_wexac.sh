#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/base_retrain_full_%J.out
#BSUB -e scripts/wexac_logs/base_retrain_full_%J.err
#BSUB -J base_full

# =====================================================================
# TIER A — "make the base model better" (binary, drop-in). The 88% base was an
# ARTIFACT of copying Haim's RECONSTRUCTION regime onto the base: 500 imgs
# (d250) + tiny first-layer init 1e-4. But the leakage attack targets the LoRA
# fine-tune's private N imgs, NOT the base training set -> the base is free to be
# a normal strong classifier. Fix: full-ish data (10000/class) + healthy init
# (0.05). Still binary odd/even, output_dim=1 -> the leakage pipeline loads it
# UNCHANGED. Expected ~97-98% (mnist) / ~97%+ (fashion).
#   datasets:    mnist, fashion (784, [1000,1000])
#   activations: relu, gelu, modifiedrelu
# Canonical theta0 = models/weights-<dataset>_<act>.pth (OVERWRITTEN; the old
# d250 models are first backed up to weights-<dataset>_<act>_d250.pth so the
# committed 88%-base leakage results stay reproducible).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

DPC=10000          # data_per_class_train (was 250) -> 20k imgs/model, 40x
INIT="[0.05,0.05]" # healthy first-layer init (was [1e-4,1e-4])

# --- back up the current (d250, ~88%) canonical models once ---
for DS in mnist fashion; do
  for ACT in relu gelu modifiedrelu; do
    C="models/weights-${DS}_${ACT}.pth"
    B="models/weights-${DS}_${ACT}_d250.pth"
    if [ -f "$C" ] && [ ! -f "$B" ]; then cp "$C" "$B"; echo "BACKUP $C -> $B"; fi
  done
done

for DS in mnist fashion; do
  PROB=${DS}_odd_even
  for ACT in relu gelu modifiedrelu; do
    echo ""; echo "########## BASE TRAIN (full)  $DS x $ACT  dpc=$DPC init=$INIT ##########"; date
    python -u Main.py --run_mode=train --problem=$PROB \
        --model_train_activation=$ACT --proj_name=${DS}_${ACT}_full \
        --data_per_class_train=$DPC \
        --model_hidden_list="[1000,1000]" --model_init_list="$INIT" \
        --train_epochs=20000 --train_lr=0.01 --train_evaluate_rate=500 \
        --train_threshold=1e-4
    if [ $? -ne 0 ]; then echo "WARN: base train $DS x $ACT failed"; continue; fi
    # collect freshest checkpoint for this proj (run dir ends in _<DS>_<ACT>_full)
    SRC=$(ls -t runs/*_${DS}_${ACT}_full/weights-*.pth 2>/dev/null | head -1)
    if [ -n "$SRC" ]; then
      cp "$SRC" "models/weights-${DS}_${ACT}.pth"
      echo "COLLECTED -> models/weights-${DS}_${ACT}.pth (from $SRC)"
    else
      echo "WARN: no checkpoint found for ${DS}_${ACT}_full"
    fi
  done
done

echo ""; echo "=== ALL FULL BASE MODELS DONE $(date) ==="
ls -la models/weights-*_*.pth 2>/dev/null | grep -E "mnist_|fashion_" || true
