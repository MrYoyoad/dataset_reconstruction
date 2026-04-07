#!/bin/bash
# CLOSED: Negative result confirmed (2026-04-07). KKT loss 330-350 for all N.
# See LESSONS_LEARNED.md for full analysis. Do NOT resubmit.
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -o wexac_logs/sprint2c_a_split_%J.out
#BSUB -e wexac_logs/sprint2c_a_split_%J.err
#BSUB -J sprint2c_a

# Sprint 2c Track A: Experiment A — KKT with fine-tuning LR × epochs × N sweep
#
# PREVIOUS FAILURE: Job 691143 was killed by KeyboardInterrupt (timeout).
# FIX: Use long-gpu queue with 48h wall time. Track A trains models for
# up to 5M epochs per config, which takes ~30min per config × 48 configs = ~24h.
#
# If this still times out, split further by uncommenting the --lr_subset flag below.

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
mkdir -p scripts/wexac_logs

echo "=== Starting Sprint 2c Track A (KKT LR × N sweep) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Full run: all 48 configs (3 LRs × 2 epoch counts × 8 N values)
# Expected runtime: ~24h on L40S
python -u -m experiments.run_sprint2c_sweep \
    --track A \
    --device cuda \
    --seed 42

echo "=== Sprint 2c Track A Complete ==="
echo "Date: $(date)"
