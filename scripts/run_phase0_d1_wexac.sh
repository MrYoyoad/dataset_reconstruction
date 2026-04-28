#!/bin/bash
# Phase 0 D1: Controlled optimizer x TV comparison
# Submits 4 independent GPU jobs (one per config) to run in parallel.
#
# Configs (all on same Flowers102 image, seed=42, full-model gradient):
#   A: Adam      + tv_weight=1e-4  (baseline)
#   B: signAdam  + tv_weight=1e-4  (Geiping et al.)
#   C: Adam      + tv_weight=1e-2  (strong TV)
#   D: signAdam  + tv_weight=1e-2  (signed + strong TV)
#
# Each job: 8 restarts x 10K iters ≈ 50 min/restart ≈ 7 hours
# All 4 run simultaneously → total wall time ≈ 7 hours (not 28).
#
# After all jobs complete, run the comparison script to generate
# the summary figure and CSV (see bottom of this file).
#
# Usage:
#   bash scripts/run_phase0_d1_wexac.sh          # submit all 4 jobs
#   bash scripts/run_phase0_d1_wexac.sh compare   # generate comparison plots

set -e
cd /home/projects/galvardi/yoado
mkdir -p results figures scripts/wexac_logs

COMMON_ARGS="--dataset flowers102 --n_iters 10000 --n_restarts 8 --tv_norm l2 --device cuda --seed 42 --mode full"

if [ "${1:-}" = "compare" ]; then
    echo "=== Generating D1 comparison from saved results ==="
    python -u -m experiments.phase0_d1_compare
    exit 0
fi

echo "=== Submitting 4 D1 jobs in parallel ==="

# Config A: Adam + tv=1e-4
JOBA=$(bsub -q long-gpu \
    -R "rusage[mem=32768] select[ngpus>0]" \
    -gpu "num=1" \
    -W 8:00 \
    -o wexac_logs/phase0_d1_A_%J.out \
    -e wexac_logs/phase0_d1_A_%J.err \
    -J phase0_d1_A \
    bash -c "
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
pip install -q scipy 2>/dev/null
echo '=== D1 Config A: Adam + tv=1e-4 ==='
echo 'Date:' \$(date)
echo 'Host:' \$(hostname)
python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\"
python -u -m experiments.phase0_vit_inversion \
    $COMMON_ARGS \
    --optimizer Adam \
    --tv_weight 1e-4
echo '=== D1 Config A Complete ==='
echo 'Date:' \$(date)
" 2>&1 | grep -oP '\d+')
echo "Config A (Adam, tv=1e-4):     Job $JOBA"

# Config B: signAdam + tv=1e-4
JOBB=$(bsub -q long-gpu \
    -R "rusage[mem=32768] select[ngpus>0]" \
    -gpu "num=1" \
    -W 8:00 \
    -o wexac_logs/phase0_d1_B_%J.out \
    -e wexac_logs/phase0_d1_B_%J.err \
    -J phase0_d1_B \
    bash -c "
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
pip install -q scipy 2>/dev/null
echo '=== D1 Config B: signAdam + tv=1e-4 ==='
echo 'Date:' \$(date)
echo 'Host:' \$(hostname)
python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\"
python -u -m experiments.phase0_vit_inversion \
    $COMMON_ARGS \
    --optimizer signAdam \
    --tv_weight 1e-4
echo '=== D1 Config B Complete ==='
echo 'Date:' \$(date)
" 2>&1 | grep -oP '\d+')
echo "Config B (signAdam, tv=1e-4): Job $JOBB"

# Config C: Adam + tv=1e-2
JOBC=$(bsub -q long-gpu \
    -R "rusage[mem=32768] select[ngpus>0]" \
    -gpu "num=1" \
    -W 8:00 \
    -o wexac_logs/phase0_d1_C_%J.out \
    -e wexac_logs/phase0_d1_C_%J.err \
    -J phase0_d1_C \
    bash -c "
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
pip install -q scipy 2>/dev/null
echo '=== D1 Config C: Adam + tv=1e-2 ==='
echo 'Date:' \$(date)
echo 'Host:' \$(hostname)
python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\"
python -u -m experiments.phase0_vit_inversion \
    $COMMON_ARGS \
    --optimizer Adam \
    --tv_weight 1e-2
echo '=== D1 Config C Complete ==='
echo 'Date:' \$(date)
" 2>&1 | grep -oP '\d+')
echo "Config C (Adam, tv=1e-2):     Job $JOBC"

# Config D: signAdam + tv=1e-2
JOBD=$(bsub -q long-gpu \
    -R "rusage[mem=32768] select[ngpus>0]" \
    -gpu "num=1" \
    -W 8:00 \
    -o wexac_logs/phase0_d1_D_%J.out \
    -e wexac_logs/phase0_d1_D_%J.err \
    -J phase0_d1_D \
    bash -c "
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
pip install -q scipy 2>/dev/null
echo '=== D1 Config D: signAdam + tv=1e-2 ==='
echo 'Date:' \$(date)
echo 'Host:' \$(hostname)
python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\"
python -u -m experiments.phase0_vit_inversion \
    $COMMON_ARGS \
    --optimizer signAdam \
    --tv_weight 1e-2
echo '=== D1 Config D Complete ==='
echo 'Date:' \$(date)
" 2>&1 | grep -oP '\d+')
echo "Config D (signAdam, tv=1e-2): Job $JOBD"

echo ""
echo "=== All 4 D1 jobs submitted ==="
echo "Monitor: bjobs -w | grep phase0_d1"
echo "Peek:    bpeek <JOBID>"
echo "After completion: bash scripts/run_phase0_d1_wexac.sh compare"
