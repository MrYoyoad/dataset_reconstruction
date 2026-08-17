#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_train_%J.out
#BSUB -e scripts/wexac_logs/flowers32_train_%J.err
#BSUB -J flowers32_train

# =====================================================================
# Phase A (D=3072 rung): train the flowers-NATIVE base model theta_0 to a strong max-margin
# fit, then a Phase-D (Q-B) variant that HOLDS OUT species 0..19 from training.
#   Task: species-index parity over 102 species, binary max-margin MLP 3072-1000-1000-1 (no bias).
#   Data: train+val pooled (~2040 imgs), 500/class. Reconstruction fine-tune/control come later
#         from the disjoint TEST split (never seen here).
# Outputs (canonical, consumed by DATASET_SPECS / --pretrained_path):
#   models/weights-flowers32.pth          (main base, all 102 species)
#   models/weights-flowers32_holdout.pth  (Phase-D base, species 0..19 held out)
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado/dataset_reconstruction
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
mkdir -p runs models

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: validate the flowers102_parity data path (RGB, native dims) ----------
python - <<'PY' || { echo "STAGE 0 FAILED: flowers102_parity loader"; exit 1; }
import types
from problems.flowers102_parity import create_labels, _flowers_transform, fetch_flowers
assert create_labels([0,1,2,3]).tolist() == [0,1,0,1]
tfm = _flowers_transform(types.SimpleNamespace(flowers_hw=32, flowers_gray=False))
ds = fetch_flowers('./data', train=False, transform=tfm)   # test split
img, lbl = ds[0]
assert tuple(img.shape) == (3, 32, 32), img.shape
print("STAGE 0 OK: RGB 32x32 loader, parity labels")
PY

# ---------- STAGE 0.5: tiny 2-epoch train -> validate the full Main.py path + checkpoint ----------
echo ""; echo "########## STAGE 0.5: tiny train smoke (2 epochs, 10/class) ##########"
python -u Main.py --run_mode=train --problem=flowers102_parity --proj_name=flowers32_smoke \
  --flowers_hw=32 --flowers_gray=false --flowers_holdout="[]" \
  --data_per_class_train=10 --data_reduce_mean=true \
  --model_hidden_list="[1000,1000]" --model_use_bias=false \
  --model_init_list="[0.0001,0.0001]" --use_init_scale=true --use_init_scale_only_first=true \
  --model_train_activation=relu --train_epochs=2 --train_lr=0.01 --train_evaluate_rate=1 \
  --train_save_model=true --seed=1 --precision=double --wandb_active=false --cuda=true \
  || { echo "STAGE 0.5 FAILED: Main.py train path"; exit 1; }
SMOKE=$(ls -t runs/*flowers32_smoke*/weights-*.pth 2>/dev/null | head -1)
python - <<PY || { echo "STAGE 0.5 FAILED: smoke checkpoint load"; exit 1; }
import torch, torch.nn as nn
from CreateModel import NeuralNetwork
m = NeuralNetwork(input_dim=3072, hidden_dim_list=[1000, 1000], output_dim=1,
                  activation=nn.ReLU(), use_bias=False).double()
ck = torch.load("$SMOKE", map_location='cpu', weights_only=False)
m.load_state_dict(ck['state_dict'])
print("STAGE 0.5 OK: smoke checkpoint loads into NeuralNetwork(3072,[1000,1000])")
PY
echo "STAGE 0.5 PASSED"

train_rung () {
  HW=$1; TAG=$2; HOLDOUT=$3
  echo ""; echo "########## TRAIN base hw=$HW tag=$TAG holdout=$HOLDOUT ##########"; date
  python -u Main.py --run_mode=train --problem=flowers102_parity --proj_name=${TAG}_parity \
    --flowers_hw=$HW --flowers_gray=false --flowers_holdout="$HOLDOUT" \
    --data_per_class_train=500 --data_reduce_mean=true \
    --model_hidden_list="[1000,1000]" --model_use_bias=false \
    --model_init_list="[0.0001,0.0001]" --use_init_scale=true --use_init_scale_only_first=true \
    --model_train_activation=relu --train_epochs=150000 --train_lr=0.01 \
    --train_evaluate_rate=1000 --train_threshold=1e-20 --train_save_model=true --train_save_model_every=30000 \
    --seed=1 --precision=double --wandb_active=false --cuda=true
  CKPT=$(ls -t runs/*${TAG}_parity*/weights-*.pth 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint found for $TAG"; exit 2; fi
  cp "$CKPT" "models/weights-${TAG}.pth"
  echo "Copied $CKPT -> models/weights-${TAG}.pth"
}

# Main base: all species.
train_rung 32 flowers32 "[]"
# Phase-D base: hold out species 0..19 (10 even + 10 odd -> both parities available as 'novel').
train_rung 32 flowers32_holdout "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]"

echo ""; echo "=== ALL FLOWERS32 TRAINING COMPLETE $(date) ==="
