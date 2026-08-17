#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers64_train_%J.out
#BSUB -e scripts/wexac_logs/flowers64_train_%J.err
#BSUB -J flowers64_train

# =====================================================================
# Phase A (D=12288 rung): train the flowers-NATIVE base model theta_0 at 64x64x3.
#   Task: species-index parity, MLP 12288-1000-1000-1 (no bias), train+val pooled, 500/class.
#   NOTE (risk): max-margin MLP training is untested at this dim. If the loss stalls well above
#   1e-8 (never interpolates), widen the net to [2048,2048] and resubmit. Watch the first eval
#   lines: train-error must reach 0; p-val must grow.
# Output: models/weights-flowers64.pth
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
mkdir -p runs models

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

python - <<'PY' || { echo "STAGE 0 FAILED: flowers64 loader"; exit 1; }
import types
from problems.flowers102_parity import _flowers_transform, fetch_flowers
tfm = _flowers_transform(types.SimpleNamespace(flowers_hw=64, flowers_gray=False))
img, lbl = fetch_flowers('./data', train=False, transform=tfm)[0]
assert tuple(img.shape) == (3, 64, 64), img.shape
print("STAGE 0 OK: RGB 64x64 loader")
PY

echo ""; echo "########## TRAIN flowers64 base (D=12288) ##########"; date
python -u Main.py --run_mode=train --problem=flowers102_parity --proj_name=flowers64_parity \
  --flowers_hw=64 --flowers_gray=false --flowers_holdout="[]" \
  --data_per_class_train=500 --data_reduce_mean=true \
  --model_hidden_list="[1000,1000]" --model_use_bias=false \
  --model_init_list="[0.0001,0.0001]" --use_init_scale=true --use_init_scale_only_first=true \
  --model_train_activation=relu --train_epochs=150000 --train_lr=0.01 \
  --train_evaluate_rate=1000 --train_threshold=1e-20 --train_save_model=true --train_save_model_every=30000 \
  --seed=1 --precision=double --wandb_active=false --cuda=true

CKPT=$(ls -t runs/*flowers64_parity*/weights-*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "ERROR: no flowers64 checkpoint found"; exit 2; fi
cp "$CKPT" "models/weights-flowers64.pth"
echo "Copied $CKPT -> models/weights-flowers64.pth"

echo ""; echo "=== FLOWERS64 TRAINING COMPLETE $(date) ==="
