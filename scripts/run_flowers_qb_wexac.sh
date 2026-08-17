#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_qb_%J.out
#BSUB -e scripts/wexac_logs/flowers_qb_%J.err
#BSUB -J flowers_qb

# =====================================================================
# Phase D (Q-B, deck slide 20): pretrain/finetune OVERLAP contrast on the flowers32 holdout base.
# Base theta_0 held out species 0..19. Fine-tune on:
#   seen  = species theta_0 trained on (20..101)  -> overlap regime (theta_0 already fits them)
#   novel = held-out species 0..19                -> no overlap
# Prediction: overlap -> smaller weight_change + lower absolute-x faithfulness (recovers novelty,
# not the instance). Multi-seed (42,43,44). softplus, r=8, T=1.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

HOLDOUT="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
CKPT=dataset_reconstruction/models/weights-flowers32_holdout.pth

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: holdout theta_0 loads + seen/novel filter disjoint ----------
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
from experiments.configs import DATASET_SPECS
from experiments.data_utils import get_finetuning_data
from experiments.run_experiment_b import load_pretrained
spec = DATASET_SPECS['flowers32']
H = list(range(20))
m = load_pretrained(device='cpu', pretrained_path='dataset_reconstruction/models/weights-flowers32_holdout.pth',
                    input_dim=spec['input_dim'], hidden=spec['hidden'])
_, _, seen, _ = get_finetuning_data(2, seed=42, dataset='flowers32', source='seen', holdout_species=H)
_, _, novel, _ = get_finetuning_data(1, seed=42, dataset='flowers32', source='novel', holdout_species=H)
assert all(int(s) not in H for s in seen), seen
assert all(int(s) in H for s in novel), novel
print("STAGE 0 OK: holdout theta_0 loads; seen/novel disjoint")
PY
echo "STAGE 0 PASSED"

for SRC in seen novel; do
  for SEED in 42 43 44; do
    echo ""; echo "########## flowers32 Q-B source=$SRC seed=$SEED ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed "$SEED" --lr 0.01 --finetune_activation softplus \
      --source "$SRC" --holdout_species $HOLDOUT \
      --no_baseline --save_results --skip_if_exists --device cuda
  done
done

echo ""; echo "=== FLOWERS Q-B (Phase D) COMPLETE $(date) ==="
