#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_activation_%J.out
#BSUB -e scripts/wexac_logs/flowers32_activation_%J.err
#BSUB -J flowers32_activation

# =====================================================================
# Phase C (flowers32, D=3072): activation spectrum x LR grid, N=2, T=1, r=8, oracle.
# Doubles as the LR-CALIBRATION (Addition 2 + the lr axis): each activation has its own usable
# LR band; read the matched-weight_change comparison POST-HOC (rescore) rather than at fixed LR.
# Question: does softplus >> ... > elu survive on RGB native-dimension flowers?
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: shapes, theta_0 load, forward, filename uniqueness ----------
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
import argparse, torch
from experiments.configs import DATASET_SPECS
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
from experiments.run_experiment_b import load_pretrained, build_base_name
spec = DATASET_SPECS['flowers32']
xf, yf, sp, _ = get_finetuning_data(1, seed=42, dataset='flowers32')
assert tuple(xf.shape) == (2, 3, 32, 32), xf.shape
get_control_images_in_distribution(sp, dataset='flowers32')
m = load_pretrained(device='cpu', pretrained_path=spec['pretrained'],
                    input_dim=spec['input_dim'], hidden=spec['hidden'])
assert m(xf.double()).shape == (2, 1)
def nm(ds, act):
    a = argparse.Namespace(n_steps=1, dataset=ds, rank=8, free_coefficients=False, seed=42,
                           relu_alpha=149.87, finetune_activation=act, n_per_class=1,
                           loss_type='l2', lr=0.01, anchor_alpha=0.0)
    return build_base_name(a)
assert len({nm('flowers32','gelu'), nm('flowers32','softplus'), nm('flowers','gelu')}) == 3
print("STAGE 0 OK: theta_0 loads, forward=(2,1), filenames unique")
PY
echo "STAGE 0 PASSED"

# ---------- STAGE 0.5: one short real config -> sane NTK diagnostics before the full grid ----------
echo ""; echo "########## STAGE 0.5: smoke gelu 300ep ##########"
OUT=$(python -u -m experiments.run_experiment_b --dataset flowers32 --n_steps 1 --rank 8 \
      --seed 42 --lr 0.01 --finetune_activation gelu --no_baseline \
      --extraction_epochs 300 --device cuda 2>&1)
echo "$OUT" | tail -12
echo "$OUT" | grep -q "weight_change" || { echo "STAGE 0.5 FAILED: no NTK diagnostics"; exit 1; }
echo "STAGE 0.5 PASSED"

# ---------- Full grid: activation x LR, N=2 ----------
ACTS="softplus silu gelu mish gelu_tanh elu celu tanh sigmoid selu hardswish leaky_relu relu"
LRS="0.005 0.01 0.02 0.05"

for LR in $LRS; do
  for ACT in $ACTS; do
    echo ""; echo "########## flowers32 N=2 act=$ACT lr=$LR T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --n_steps 1 --rank 8 --seed 42 --lr "$LR" \
      --finetune_activation "$ACT" \
      --no_baseline --save_results --skip_if_exists --device cuda
  done
done

echo ""; echo "=== FLOWERS32 ACTIVATION SWEEP COMPLETE $(date) ==="
echo "Rescore: python -m experiments.recompute_metrics --glob 'results/exp_b_T1_flowers32_*' --out results/flowers32_activation.csv"
