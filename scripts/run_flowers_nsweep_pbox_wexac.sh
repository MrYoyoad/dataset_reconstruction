#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_nsweep_pbox_%J.out
#BSUB -e scripts/wexac_logs/flowers_nsweep_pbox_%J.err
#BSUB -J flowers_nsweep_pbox

# =====================================================================
# RE-DO of the N-sweep with the proper [0,1] pixel box (the audit flagged N>=4 as clip-inflated:
# npc=4 clipped 47% -> raw magnitude depressed). Reproduces the canonical MAIN N-sweep recipe EXACTLY
# and only adds --pixel_box (+ verify_weight 5.0 on every N so the box bites), across 3 seeds for
# honest error bars. Canonical: npc 1,2 = free-c joint extraction; npc 4,8 = --sequential_peel
# (now pixel-boxed too). Direction won't reverse (collapse shows on ssim_norm+margin) but the
# absolute N>=4 numbers become honest.  Axis: npc in {1,2,4,8} == N in {2,4,8,16}.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"
CKPT=dataset_reconstruction/models/weights-flowers32.pth

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: pixel_box now threaded into BOTH extraction paths (joint + peel) ----------
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
import inspect, torch
from experiments.ntk_extraction import run_sequential_peeling, get_pixel_box_loss
sig = inspect.signature(run_sequential_peeling).parameters
assert 'pixel_box' in sig and 'ds_mean' in sig, "peel path missing pixel_box/ds_mean"
ds = torch.full((1,3,32,32), 0.5)
assert get_pixel_box_loss(torch.zeros(2,3,32,32), ds).item() == 0.0
assert get_pixel_box_loss(torch.full((2,3,32,32),0.9), ds).item() > 0.0
print("STAGE 0 OK: pixel_box on joint + peel paths; box loss sane")
PY
echo "STAGE 0 PASSED"

for SEED in 42 43 44; do
  for NPC in 1 2 4 8; do
    PEEL=""; [ "$NPC" -ge 4 ] && PEEL="--sequential_peel"
    echo ""; echo "########## flowers32 N-sweep pbox npc=$NPC seed=$SEED ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed "$SEED" --lr 0.01 --n_per_class "$NPC" \
      --pixel_box --verify_weight 5.0 $PEEL \
      $FREEC --no_baseline --save_results --skip_if_exists --device cuda
  done
done
echo ""; echo "=== FLOWERS N-SWEEP pbox COMPLETE $(date) ==="
