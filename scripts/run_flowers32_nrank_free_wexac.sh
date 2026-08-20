#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_nrank_free_%J.out
#BSUB -e scripts/wexac_logs/flowers32_nrank_free_%J.err
#BSUB -J flowers32_nrank_free

# =====================================================================
# Phase C (flowers32): the remaining user axes, priority-ordered.
#   (1) N size   : n_per_class in {1,2,4,8} -> retrieval top-1 vs 1/N (the N-axis metric);
#                  watch superposition collapse. verify_weight 5.0 at N>=8.
#   (2) NTK/rank : r in {4,8,16,32,64} + full-FT (run_baseline) at N=2 -> feature_stability,
#                  r>=N well-posedness, eff_rank<=N.
#   (3) Optim alg: finetune {sgd,adamw} x extraction {lbfgs,adam} x loss {l2,cosine} at N=2.
# Winner activation = softplus (confirm from the activation sweep; edit ACT if it changes).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
ACT=softplus

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
from experiments.configs import DATASET_SPECS
from experiments.data_utils import get_finetuning_data
from experiments.run_experiment_b import load_pretrained
spec = DATASET_SPECS['flowers32']
xf, yf, sp, _ = get_finetuning_data(1, seed=42, dataset='flowers32')
m = load_pretrained(device='cpu', pretrained_path=spec['pretrained'],
                    input_dim=spec['input_dim'], hidden=spec['hidden'])
assert m(xf.double()).shape == (2, 1)
print("STAGE 0 OK")
PY
echo "STAGE 0 PASSED"

# ---------- (1) N size ----------
for NPC in 1 2 4 8; do
  VW=""; [ "$NPC" -ge 4 ] && VW="--verify_weight 5.0"
  echo ""; echo "########## flowers32 N-sweep npc=$NPC act=$ACT r=8 ##########"; date
  python -u -m experiments.run_experiment_b \
    --dataset flowers32 --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
    --finetune_activation "$ACT" --n_per_class "$NPC" $VW \
    --no_baseline --free_coefficients --save_results --skip_if_exists --device cuda
done

# ---------- (2) NTK / rank (N=2, includes full-FT baseline) ----------
for R in 4 8 16 32 64; do
  echo ""; echo "########## flowers32 rank r=$R act=$ACT N=2 ##########"; date
  python -u -m experiments.run_experiment_b \
    --dataset flowers32 --n_steps 1 --rank "$R" --seed 42 --lr 0.01 \
    --finetune_activation "$ACT" \
    --free_coefficients --save_results --skip_if_exists --device cuda
done

# ---------- (3) Optimization algorithm: finetune x extraction x loss ----------
for FTOPT in sgd adamw; do
  for EXOPT in lbfgs adam; do
    for LOSS in l2 cosine; do
      echo ""; echo "########## flowers32 optim ft=$FTOPT ex=$EXOPT loss=$LOSS ##########"; date
      python -u -m experiments.run_experiment_b \
        --dataset flowers32 --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
        --finetune_activation "$ACT" \
        --finetune_optimizer "$FTOPT" --optimizer "$EXOPT" --loss_type "$LOSS" \
        --no_baseline --free_coefficients --save_results --skip_if_exists --device cuda
    done
  done
done

echo ""; echo "=== FLOWERS32 N/RANK/OPTIM SWEEP COMPLETE $(date) ==="
