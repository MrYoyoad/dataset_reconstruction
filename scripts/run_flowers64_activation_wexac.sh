#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers64_activation_%J.out
#BSUB -e scripts/wexac_logs/flowers64_activation_%J.err
#BSUB -J flowers64_activation

# =====================================================================
# Phase C (flowers64, D=12288): activation spectrum x LR grid, N=2, T=1, r=8, oracle.
# The rich rung of the Q-A dimension ladder. Extraction is heavier here (x-hat is 12288-dim/
# sample) so the LR grid is trimmed to 3 points. Read matched-weight_change post-hoc (rescore).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
import argparse
from experiments.configs import DATASET_SPECS
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
from experiments.run_experiment_b import load_pretrained, build_base_name
spec = DATASET_SPECS['flowers64']
xf, yf, sp, _ = get_finetuning_data(1, seed=42, dataset='flowers64')
assert tuple(xf.shape) == (2, 3, 64, 64), xf.shape
get_control_images_in_distribution(sp, dataset='flowers64')
m = load_pretrained(device='cpu', pretrained_path=spec['pretrained'],
                    input_dim=spec['input_dim'], hidden=spec['hidden'])
assert m(xf.double()).shape == (2, 1)
assert len({build_base_name(argparse.Namespace(n_steps=1, dataset='flowers64', rank=8,
            free_coefficients=False, seed=42, relu_alpha=149.87, finetune_activation=a,
            n_per_class=1, loss_type='l2', lr=0.01, anchor_alpha=0.0)) for a in
            ('gelu','softplus')}) == 2
print("STAGE 0 OK")
PY
echo "STAGE 0 PASSED"

echo ""; echo "########## STAGE 0.5: smoke gelu 300ep ##########"
OUT=$(python -u -m experiments.run_experiment_b --dataset flowers64 --n_steps 1 --rank 8 \
      --seed 42 --lr 0.01 --finetune_activation gelu --no_baseline \
      --extraction_epochs 300 --device cuda 2>&1)
echo "$OUT" | tail -12
echo "$OUT" | grep -q "weight_change" || { echo "STAGE 0.5 FAILED"; exit 1; }
echo "STAGE 0.5 PASSED"

ACTS="softplus silu gelu mish gelu_tanh elu celu tanh sigmoid selu hardswish leaky_relu relu"
LRS="0.005 0.01 0.02"

for LR in $LRS; do
  for ACT in $ACTS; do
    echo ""; echo "########## flowers64 N=2 act=$ACT lr=$LR T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers64 --n_steps 1 --rank 8 --seed 42 --lr "$LR" \
      --finetune_activation "$ACT" \
      --no_baseline --save_results --skip_if_exists --device cuda
  done
done

echo ""; echo "=== FLOWERS64 ACTIVATION SWEEP COMPLETE $(date) ==="
