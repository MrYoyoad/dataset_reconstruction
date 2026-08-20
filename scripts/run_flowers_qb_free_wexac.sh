#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_qb_free_%J.out
#BSUB -e scripts/wexac_logs/flowers_qb_free_%J.err
#BSUB -J flowers_qb_free

# =====================================================================
# Phase D (Q-B, deck slide 20): pretrain/finetune OVERLAP contrast on the flowers32 holdout base.
# Base theta_0 held out species 0..19. Fine-tune on:
#   seen  = species theta_0 trained on (20..101)  -> overlap regime (theta_0 already fits them)
#   novel = held-out species 0..19                -> no overlap
# Prediction: overlap -> smaller weight_change + lower absolute-x faithfulness (recovers novelty,
# not the instance). Multi-seed (42,43,44). r=8, T=1.
#
# CLEAN RE-RUN (2026-08-20): two corrections vs the first Q-B pass, so the seen-vs-novel gap is a
# real leakage signal and not a metric artifact:
#   1. VALIDATED free-c recipe (the Haim mode that reproduced ~0.65 on flowers32 r8):
#      sgd extraction + relu_alpha 10000 (~ReLU) + consistency_weight 1.0 + n_restarts 5.
#      (The first pass used --finetune_activation softplus and dropped the recipe -> weaker/uncomparable.)
#   2. PROPER [0,1] PIXEL BOX (--pixel_box, --verify_weight 5.0): the first pass only boxed the centered
#      x in [-1,1], so the novel arm clipped ~50% of the DISPLAYED image (x+ds_mean) and its raw SSIM
#      collapsed while ssim_norm barely moved. --pixel_box boxes the image itself -> clip-free SSIM.
# Re-score both arms on ssim_norm / NCC (scale-robust), not raw SSIM alone.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

HOLDOUT="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
CKPT=dataset_reconstruction/models/weights-flowers32_holdout.pth
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: holdout theta_0 loads, seen/novel disjoint, AND pixel_box path works ----------
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
import torch
from experiments.configs import DATASET_SPECS
from experiments.data_utils import get_finetuning_data
from experiments.run_experiment_b import load_pretrained
from experiments.ntk_extraction import get_pixel_box_loss

# (a) holdout theta_0 loads; seen/novel filters disjoint
spec = DATASET_SPECS['flowers32']
H = list(range(20))
m = load_pretrained(device='cpu', pretrained_path='dataset_reconstruction/models/weights-flowers32_holdout.pth',
                    input_dim=spec['input_dim'], hidden=spec['hidden'])
_, _, seen, _ = get_finetuning_data(2, seed=42, dataset='flowers32', source='seen', holdout_species=H)
_, _, novel, _ = get_finetuning_data(1, seed=42, dataset='flowers32', source='novel', holdout_species=H)
assert all(int(s) not in H for s in seen), seen
assert all(int(s) in H for s in novel), novel

# (b) pixel_box loss: zero inside [0,1], positive when the image leaves it
ds = torch.full((1, 3, 32, 32), 0.5)
x_in = torch.zeros(2, 3, 32, 32)          # image = 0.5 -> inside
x_hi = torch.full((2, 3, 32, 32), 0.9)    # image = 1.4 -> above 1
assert get_pixel_box_loss(x_in, ds).item() == 0.0, "inside box must be 0"
assert get_pixel_box_loss(x_hi, ds).item() > 0.0, "outside box must be >0"
print("STAGE 0 OK: holdout theta_0 loads; seen/novel disjoint; pixel_box path sane")
PY
echo "STAGE 0 PASSED"

for SRC in seen novel; do
  for SEED in 42 43 44; do
    echo ""; echo "########## flowers32 Q-B source=$SRC seed=$SEED (pbox) ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed "$SEED" --lr 0.01 \
      --source "$SRC" --holdout_species $HOLDOUT \
      --pixel_box --verify_weight 5.0 \
      --no_baseline $FREEC --save_results --skip_if_exists --device cuda
  done
done

echo ""; echo "=== FLOWERS Q-B (Phase D, clean pbox) COMPLETE $(date) ==="
