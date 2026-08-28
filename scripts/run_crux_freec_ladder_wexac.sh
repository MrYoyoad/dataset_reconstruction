#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 23:00
#BSUB -o scripts/wexac_logs/crux_freec_ladder_%J.out
#BSUB -e scripts/wexac_logs/crux_freec_ladder_%J.err
#BSUB -J crux_freec_ladder

# =====================================================================
# Crux GPU-closure, piece 1/3 — the REALISTIC (free-coefficient) leakage wc-LADDER.
#
# WHY: the first-pass job-857271 ranking is ORACLE (free_coefficients=False, known
# coefficients = upper bound). This project has documented precedent (crux_activation_
# analysis.md:165) that free-c FLIPS the activation ranking. So the "kinked leak most"
# oracle read must be re-tested under the realistic free-c attack (user rule: oracle is
# cheating/upper-bound only).
#
# WHY A LADDER: ntk_passed (wc<0.01 AND feat_stab>0.99) and a "meaningful" wc≈0.1-0.3 are
# DISJOINT BY CONSTRUCTION — the plan's "matched-wc AND ntk_passed" target was ill-posed.
# So instead we match at EXACT per-activation wc across a ladder {0.005 (NTK rung), 0.03,
# 0.1, 0.3} and make the ranking's wc-DEPENDENCE the headline (stable across rungs => robust;
# flips => that IS the finding). feature_stability is reported per rung as a graded covariate.
# The 0.005 NTK rung may be degenerate (feat_stab~1 for all, leakage~noise) — if so, that
# itself shows leakage is a finite-wc phenomenon, not an NTK-regime one.
#
# LRs are pre-computed per (activation, target-wc) by inverting the first-pass wc-vs-LR fit
# (wc is mode-independent) -> results/crux_ladder_lrs.txt (act target_wc lr). Each run records
# its ACTUAL wc, so any prediction error is caught post-hoc (match actual wc to nearest rung).
# free_coefficients=True; r=8, N=2, T=1, seed 42. Ordered smooth->kinked. --skip_if_exists
# makes it resumable (free-c tensors get a distinct "_free_" name, no collision with oracle).
# 52 runs (13 acts x 4 rungs), ~5-8 min each free-c.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: LR file present + activations construct ----------
if [ ! -s results/crux_ladder_lrs.txt ]; then echo "STAGE 0 FAILED: results/crux_ladder_lrs.txt missing"; exit 1; fi
python - <<'PY' || { echo "STAGE 0 FAILED: activation construction"; exit 1; }
from experiments.run_experiment_b import make_activation
import torch
for a in "sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu".split():
    assert torch.isfinite(make_activation(a)(torch.linspace(-3,3,7,dtype=torch.double))).all(), a
print("all activations construct")
PY
echo "STAGE 0 PASSED ($(wc -l < results/crux_ladder_lrs.txt) act x rung pairs)"

# ---------- STAGE 1: free-c ladder ----------
while read ACT WC LR; do
    [ -z "$ACT" ] && continue
    echo ""; echo "########## act=$ACT target_wc=$WC lr=$LR free-c T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
        --n_steps 1 --rank 8 --seed 42 --lr "$LR" \
        --finetune_activation "$ACT" \
        --free_coefficients \
        --no_baseline --save_results --skip_if_exists --device cuda
done < results/crux_ladder_lrs.txt

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
