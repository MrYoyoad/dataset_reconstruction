#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/activation_spectrum_%J.out
#BSUB -e scripts/wexac_logs/activation_spectrum_%J.err
#BSUB -J activation_spectrum

# =====================================================================
# Step 2a of the coupled activation x anchor x linearization study
# (notes/next_experiment_plan.md): the matched-weight_change activation SPECTRUM.
#
# Step 1's rescore of job 857271 gave a first-pass ranking of the C^inf family (softplus >> silu >
# gelu ~ gelu_tanh > mish > elu) but LEFT GAPS: the kinked controls (relu, leaky_relu, tanh,
# hardswish) never ran, and sigmoid/celu/selu did not exist. This job fills the spectrum so every
# activation sits on a common weight_change axis, ordered by smoothness, at T=1.
#
# Design (why no --target_weight_change LR search): Step 1 showed lr=0.01 already yields
# weight_change ~= 0.04 across the smooth family, so a small lr grid {0.005, 0.01, 0.02} brackets a
# matched weight_change and lets us read every activation at a common weight_change POST-HOC
# (interpolate on the saved per-run weight_change). --skip_if_exists reuses the lr=0.01 tensors
# already on disk for gelu/silu/softplus/mish/gelu_tanh/elu, so only the gaps actually compute.
# Every run uses the same activation for fine-tune AND extraction (consistency principle).
#
# Ordered smooth -> kinked so a preemption still delivers the smooth end. ~33 configs x ~5 min.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# smooth (C^inf) -> C^1 -> kinked (C^0)
ACTS="sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu"
LRS="0.005 0.01 0.02"

# ---------- STAGE 0: validate on the GPU node before spending hours ----------
echo ""; echo "########## STAGE 0: activation construction + smoothness tests ##########"
python - <<'PY' || { echo "STAGE 0 FAILED: activation construction"; exit 1; }
from experiments.run_experiment_b import make_activation
import torch
for a in "sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu".split():
    m = make_activation(a); out = m(torch.linspace(-3, 3, 7, dtype=torch.double))
    assert torch.isfinite(out).all(), a
    print(f"  OK {a}")
print("all activations construct")
PY
python -m pytest experiments/tests/test_activations.py -q || { echo "STAGE 0 FAILED: activation tests"; exit 1; }
echo "STAGE 0 PASSED"

# ---------- STAGE 1: the spectrum sweep (LoRA r=8, N=2, seed 42, T=1) ----------
for ACT in $ACTS; do
    for LR in $LRS; do
        echo ""; echo "########## act=$ACT lr=$LR T=1 r=8 ##########"; date
        python -u -m experiments.run_experiment_b \
            --n_steps 1 --rank 8 --seed 42 --lr "$LR" \
            --finetune_activation "$ACT" \
            --no_baseline --save_results --skip_if_exists --device cuda
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
