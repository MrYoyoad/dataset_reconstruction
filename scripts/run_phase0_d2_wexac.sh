#!/bin/bash
# Phase 0 D2: Targeted hyperparameter sweep around D1 winner
#
# Grid: tv_weight × lr × n_iters (signAdam fixed)
#   tv_weight: 5e-3, 1e-2, 2e-2, 5e-2, 1e-1  (5 values)
#   lr:        0.01, 0.05, 0.1, 0.5            (4 values)
#   n_iters:   10000, 30000                     (2 values)
#   Total: 40 configs, indexed 0-39
#
# Restarts: 4 for 10K iters (~3.5h), 2 for 30K iters (~5h)
#
# Usage:
#   bash scripts/run_phase0_d2_wexac.sh          # submit all 40 jobs
#   bash scripts/run_phase0_d2_wexac.sh compare   # generate comparison plots
#   bash scripts/run_phase0_d2_wexac.sh status     # check job status

set -e
cd /home/projects/galvardi/yoado
mkdir -p results figures/phase0/d2_sweep scripts/wexac_logs

if [ "${1:-}" = "compare" ]; then
    echo "=== Generating D2 comparison ==="
    source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
    conda activate /home/projects/galvardi/yoado/.conda/envs/rec
    python -u -m experiments.phase0_d2_compare
    exit 0
fi

if [ "${1:-}" = "status" ]; then
    echo "=== D2 Job Status ==="
    bjobs -w 2>&1 | grep phase0_d2 || echo "No running D2 jobs"
    echo ""
    echo "Completed results:"
    ls -1 results/phase0_d2_*.pth 2>/dev/null | wc -l
    echo "out of 40 total configs"
    exit 0
fi

echo "=== Submitting 40 D2 jobs ==="
echo "Grid: tv_weight(5) × lr(4) × n_iters(2) = 40 configs"
echo ""

for IDX in $(seq 0 39); do
    SCRIPT="/tmp/phase0_d2_${IDX}.sh"
    cat > "$SCRIPT" << HEREDOC
#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o wexac_logs/phase0_d2_${IDX}_%J.out
#BSUB -e wexac_logs/phase0_d2_${IDX}_%J.err
#BSUB -J phase0_d2_${IDX}

set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
mkdir -p results figures/phase0/d2_sweep scripts/wexac_logs
pip install -q scipy 2>/dev/null

echo "=== D2 Config ${IDX}/39 ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

python -u -m experiments.phase0_vit_inversion \\
    --d2 \\
    --config_index ${IDX} \\
    --dataset flowers102 \\
    --tv_norm l2 \\
    --device cuda \\
    --seed 42

echo "=== D2 Config ${IDX} Complete ==="
echo "Date: \$(date)"
HEREDOC

    JOB_OUT=$(bsub < "$SCRIPT" 2>&1)
    JOB_ID=$(echo "$JOB_OUT" | grep -oP 'Job <\K\d+')
    # Get config params for display
    PARAMS=$(python3 -c "
from experiments.phase0_vit_inversion import d2_config_from_index
tv, lr, iters, restarts = d2_config_from_index($IDX)
print(f'tv={tv:.0e} lr={lr} iters={iters} restarts={restarts}')
" 2>/dev/null || echo "...")
    printf "  [%2d/39] Job %-8s %s\n" "$IDX" "$JOB_ID" "$PARAMS"
done

echo ""
echo "=== All 40 D2 jobs submitted ==="
echo "Monitor:  bash scripts/run_phase0_d2_wexac.sh status"
echo "After:    bash scripts/run_phase0_d2_wexac.sh compare"
