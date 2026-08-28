#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 23:00
#BSUB -o scripts/wexac_logs/crux_featstab_T_%J.out
#BSUB -e scripts/wexac_logs/crux_featstab_T_%J.err
#BSUB -J crux_featstab_T

# =====================================================================
# Crux GPU-closure, piece 2/3 — FEATURE-STABILITY-vs-T (NTK survival curve).
#
# The crux mechanism's part 1: smoother activations keep the first-order NTK/anchor
# linearization accurate over MORE fine-tuning steps T. Metric: feature_stability(T)
# = cos(∇f(θ0;x), ∇f(θ_T;x)). "NTK survival" = the largest T at which it stays above
# NTK_FEATURE_COS_THRESHOLD (0.99). Prediction: smoother -> survives to larger T.
#
# The first-pass (job 857271) is ALL T=1 -> feature-stability-vs-T was untested; this
# fills it. Mode matches the template (no --free_coefficients) so the T>1 tensors extend
# the SAME measurement as the first-pass T=1 tensors; feature_stability is coefficient-mode
# independent regardless. Fixed lr=0.01, r=8, N=2, seed 42; only T varies. Ordered
# smooth->kinked so a preemption still delivers the smooth end. --skip_if_exists makes it
# resumable and reuses the T=1 tensors already on disk. ~13 acts x 6 T, ~5 min each.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: validate activations on the GPU node before spending hours ----------
echo ""; echo "########## STAGE 0: activation construction ##########"
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
from experiments.run_experiment_b import make_activation
import torch
for a in "sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu".split():
    out = make_activation(a)(torch.linspace(-3, 3, 7, dtype=torch.double))
    assert torch.isfinite(out).all(), a
print("all activations construct")
PY
echo "STAGE 0 PASSED"

# ---------- STAGE 1: T-sweep at fixed lr (smooth -> kinked) ----------
ACTS="gelu silu softplus mish gelu_tanh sigmoid tanh elu celu selu hardswish leaky_relu relu"
TS="1 2 5 10 20 50"
for ACT in $ACTS; do
    for T in $TS; do
        echo ""; echo "########## act=$ACT T=$T lr=0.01 r=8 ##########"; date
        python -u -m experiments.run_experiment_b \
            --n_steps "$T" --rank 8 --seed 42 --lr 0.01 \
            --finetune_activation "$ACT" \
            --no_baseline --save_results --skip_if_exists --device cuda
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
