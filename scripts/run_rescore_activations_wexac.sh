#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/rescore_activations_%J.out
#BSUB -e scripts/wexac_logs/rescore_activations_%J.err
#BSUB -J rescore_activations

# =====================================================================
# Crux Step 1 (≈0 GPU) — rescore ALL 152 surviving job-857271 activation-sweep
# tensors (exp_b_T1_r8_s42_a149_*) with the full metric suite. Job 857271 died at
# RUNLIMIT before its 152 configs were analyzed; only a partial 27-row CSV (Aug 11)
# existed. This finishes the FIRST-PASS activation ranking across the full smoothness
# spectrum — sigmoid/tanh (bounded C^inf), gelu/silu/softplus/mish/gelu_tanh (C^inf),
# the softplus-b{0.5..50} sharpness knob, elu/celu/selu (C^1), and the kinked controls
# relu/leaky_relu/hardswish (which DID run — tensors on disk — just were never scored).
#
# CPU-bound metric recompute from saved tensors (no training/reconstruction); on a
# short-gpu slot only to honor the "always bsub, never local" rule.
# Honest-metric caveat carried downstream: every config is T=1 and (per the Aug-11
# pass) ntk_passed is almost always False -> this ranks activations on the metrics
# available but CANNOT yield the matched-weight_change leakage ranking or the
# feature-stability-vs-T curve; those need new GPU runs with a corrected LR band.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.recompute_metrics \
    --glob 'results/exp_b_T1_r8_s42_a149_*.pth' \
    --out results/rescored_activations_857271_full_2026-08-28.csv
echo "=== DONE $(date) ==="
wc -l results/rescored_activations_857271_full_2026-08-28.csv
