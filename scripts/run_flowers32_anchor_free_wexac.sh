#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers32_anchor_free_%J.out
#BSUB -e scripts/wexac_logs/flowers32_anchor_free_%J.err
#BSUB -J flowers32_anchor_free

# =====================================================================
# Phase C (flowers32): Addition-3 anchor alpha-sweep two-curve for candidate winner activations.
# For each activation: alpha in {0,0.25,0.5,0.75,0.9}, T=10, r=8, both full-FT + LoRA. Emits the
# headline lin-error(alpha) vs SSIM/retrieval(alpha) two-curve. Legit win = lin-error bottoms
# where SSIM peaks; red flag = SSIM keeps climbing past the lin-error minimum (anchor leaking x).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: theta_0 loads + forward ----------
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

# Candidate winners (confirm/adjust from the activation sweep rescore).
for ACT in softplus gelu silu; do
  echo ""; echo "########## flowers32 anchor two-curve act=$ACT T=10 r=8 ##########"; date
  python -u -m experiments.run_anchor_sweep \
    --dataset flowers32 --n_steps 10 --rank 8 --seed 42 \
    --finetune_activation "$ACT" --free_coefficients --save --skip_if_exists --device cuda
done

echo ""; echo "=== FLOWERS32 ANCHOR SWEEP COMPLETE $(date) ==="
